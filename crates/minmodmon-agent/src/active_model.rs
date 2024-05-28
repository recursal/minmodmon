use std::fs::File;

use anyhow::{bail, Error};
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::{event, Level};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    model::{loader::Loader, ContextAutoLimits},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        model::{Build, ModelBuilder, ModelRuntime, Quant, State},
        softmax::softmax_one,
        v5, JobRuntime,
    },
    tensor::TensorCpu,
    tokenizer::Tokenizer,
};
use wgpu::{Instance, PowerPreference};

use crate::{
    config::ModelConfig,
    sampler::Sampler,
    types::{ChatMessage, ModelInfo},
};

pub struct ActiveModel {
    id: String,
    config: ModelConfig,

    tokenizer: Tokenizer,
    context: Context,
    runtime: JobRuntime<InferInput, InferOutput>,
    state: Box<dyn State + Send + Sync>,
    initial_state: TensorCpu<f32>,
}

impl ActiveModel {
    pub(crate) async fn create(id: String, config: ModelConfig) -> Result<Self, Error> {
        // Load the tokenizer
        let contents = std::fs::read_to_string(&config.vocab)?;
        let tokenizer = Tokenizer::new(&contents)?;

        // Load the model
        let (context, runtime, state) = load_model(&config.weights).await?;

        // Get the initial state if we need to reset
        let initial_state = state.back(0).await?;

        let value = Self {
            id,
            config,

            tokenizer,
            context,
            runtime,
            state,
            initial_state,
        };

        Ok(value)
    }

    /// Get metadata information of the currently loaded model.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.id.clone(),
            object: "model".to_string(),
            created: 1715960329,
            owned_by: "Recursal AI".to_string(),
        }
    }

    /// Reset model state to a clear initial state.
    pub fn reset_state(&self) -> Result<(), Error> {
        self.state.load(0, self.initial_state.clone())?;
        Ok(())
    }

    pub async fn export_state(&self) -> Result<TensorCpu<f32>, Error> {
        let state = self.state.back(0).await?;
        Ok(state)
    }

    pub fn import_state(&self, state: TensorCpu<f32>) -> Result<(), Error> {
        self.state.load(0, state)?;
        Ok(())
    }

    pub async fn process_message(&self, message: &ChatMessage) -> Result<(), Error> {
        event!(
            Level::DEBUG,
            role = message.role,
            len = message.content.len(),
            "processing message"
        );

        // Encode content into tokens
        let content = self.tokenizer.encode(message.content.as_bytes())?;

        // Assemble with prompt format
        let role = match message.role.as_str() {
            "system" => &self.config.role_system,
            "user" => &self.config.role_user,
            "assistant" => &self.config.role_assistant,
            _ => bail!("invalid role"),
        };

        let mut assembled = role.prefix.clone();
        assembled.extend_from_slice(&content);
        assembled.extend_from_slice(&role.suffix);

        // Process the tokens into the active state
        self.process_tokens(assembled).await?;

        Ok(())
    }

    pub async fn generate_message(&self) -> Result<String, Error> {
        event!(Level::DEBUG, "generating message");

        // Start with the prompt format of an assistant message
        let mut tokens = self.config.role_assistant.prefix.clone();
        let mut next_input = tokens.pop().unwrap();
        self.process_tokens(tokens).await?;

        // Generate answer tokens
        let mut sampler = Sampler::default();
        let mut generated = Vec::new();

        while !self.should_stop_generation(&generated) {
            // Run model step
            let batch = InferInputBatch {
                tokens: vec![next_input],
                option: InferOption::Last,
            };
            let input = InferInput::new(vec![batch], 32);
            let (_input, output) = self.runtime.infer(input).await;

            let logits = &output[0].0;

            // Pick output token
            let logits = sampler.apply_penalties(logits)?;
            let probabilities = softmax_one(&self.context, logits).await?;
            next_input = sampler.sample(&probabilities);

            // Accumulate newly generated tokens
            generated.push(next_input);

            sampler.consume_token(next_input);
        }

        let content = self.finalize_generated(generated)?;

        Ok(content)
    }

    fn should_stop_generation(&self, tokens: &[u16]) -> bool {
        // Maximum tokens
        if tokens.len() >= 512 {
            return true;
        }

        // Ending with stop tokens
        if tokens.ends_with(&self.config.stop_sequence) {
            return true;
        }

        false
    }

    fn finalize_generated(&self, mut tokens: Vec<u16>) -> Result<String, Error> {
        // Trim stop tokens, if we got them at the end
        if tokens.ends_with(&self.config.stop_sequence) {
            for _ in 0..self.config.stop_sequence.len() {
                tokens.pop();
            }
        }

        // Decode the tokenized answer
        let answer_bytes = self.tokenizer.decode(&tokens)?;
        let answer = String::from_utf8_lossy(&answer_bytes);

        // Typically the model will output the first token with a prefixing space, remove that
        let value = if let Some(value) = answer.strip_prefix(' ') {
            value.to_string()
        } else {
            answer.to_string()
        };

        Ok(value)
    }

    async fn process_tokens(&self, tokens: Vec<u16>) -> Result<(), Error> {
        // Process initial prompt
        let batch = InferInputBatch {
            tokens,
            option: InferOption::Last,
        };
        let mut input = InferInput::new(vec![batch], 32);

        while !input.batches[0].tokens.is_empty() {
            let (out_input, _output) = self.runtime.infer(input).await;
            input = out_input;
        }

        Ok(())
    }
}

async fn load_model(
    path: &str,
) -> Result<
    (
        Context,
        JobRuntime<InferInput, InferOutput>,
        Box<dyn State + Send + Sync>,
    ),
    Error,
> {
    event!(Level::INFO, path, "loading model");

    // Preload the model
    let file = File::open(path)?;
    let data = unsafe { Mmap::map(&file)? };

    let safetensors = SafeTensors::deserialize(&data)?;
    let model_info = Loader::info(&safetensors)?;

    // Prepare a context for the model
    let instance = Instance::default();
    let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(&model_info)
        .build()
        .await?;

    // Quantize all layers to 8-bit
    let quantize = (0..model_info.num_layer)
        .map(|layer| (layer, Quant::NF4))
        .collect();

    // Configure the model
    let builder = ModelBuilder::new(&context, safetensors).quant(quantize);

    // Build the runtime, actually loading weights
    let model = Build::<v5::Model>::build(builder).await?;
    let builder = v5::ModelRuntime::<f16>::new(model, 1);
    let state = builder.state();
    let runtime = JobRuntime::new(builder).await;

    Ok((context, runtime, Box::new(state)))
}
