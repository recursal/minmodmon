use std::{fs::File, sync::Arc};

use anyhow::{bail, Context as _, Error};
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
        v5, JobRuntime, Submission,
    },
    tensor::TensorCpu,
    tokenizer::Tokenizer,
};
use wgpu::{Instance, PowerPreference};

use crate::{
    config::{Config, ModelConfig},
    sampler::Sampler,
    types::{ChatMessage, ModelInfo},
};

pub struct AgentManager {
    config: Arc<Config>,
    active_model_config: ModelConfig,
    tokenizer: Tokenizer,
    context: Context,
    runtime: JobRuntime<InferInput, InferOutput>,
    state: Box<dyn State + Send + Sync>,
    initial_state: TensorCpu<f32>,
}

impl AgentManager {
    pub async fn create(config: Arc<Config>) -> Result<Self, Error> {
        event!(Level::INFO, "creating agent service");

        // Pick the active model from the config
        event!(Level::INFO, "active model {:?}", config.active_model);
        let active_model_config = config
            .models
            .get(&config.active_model)
            .context("failed to find active model in config")?
            .clone();

        // Load the tokenizer
        let contents = std::fs::read_to_string(&active_model_config.vocab)?;
        let tokenizer = Tokenizer::new(&contents)?;

        // Load the model
        let (context, runtime, state) = load_model(&active_model_config.weights).await?;

        // Get the initial state if we need to reset
        let initial_state = state.back(0).await?;

        let value = AgentManager {
            config,
            active_model_config,
            context,
            tokenizer,
            runtime,
            state,
            initial_state,
        };

        Ok(value)
    }

    /// Get metadata information of the currently loaded model.
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.config.active_model.clone(),
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
        let (prefix, suffix) = match message.role.as_str() {
            "system" => (
                &self.active_model_config.system_prefix,
                &self.active_model_config.system_suffix,
            ),
            "user" => (
                &self.active_model_config.user_prefix,
                &self.active_model_config.user_suffix,
            ),
            "assistant" => (
                &self.active_model_config.assistant_prefix,
                &self.active_model_config.assistant_suffix,
            ),
            _ => bail!("invalid role"),
        };

        let mut assembled = prefix.clone();
        assembled.extend_from_slice(&content);
        assembled.extend_from_slice(suffix);

        // Process the tokens into the active state
        self.process_tokens(assembled).await?;

        Ok(())
    }

    pub async fn generate_message(&self) -> Result<String, Error> {
        event!(Level::DEBUG, "generating message");

        // Start with the prompt format of an assistant message
        let mut tokens = self.active_model_config.assistant_prefix.clone();
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

            let (sender, receiver) = tokio::sync::oneshot::channel();
            let submission = Submission { input, sender };
            self.runtime.send(submission).await?;

            let (_input, output) = receiver.await?;
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
        if tokens.ends_with(&self.active_model_config.stop_sequence) {
            return true;
        }

        false
    }

    fn finalize_generated(&self, mut tokens: Vec<u16>) -> Result<String, Error> {
        // Trim stop tokens, if we got them at the end
        if tokens.ends_with(&self.active_model_config.stop_sequence) {
            for _ in 0..self.active_model_config.stop_sequence.len() {
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
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let submission = Submission { input, sender };
            self.runtime.send(submission).await?;

            let (out_input, _output) = receiver.await?;
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
        .map(|layer| (layer, Quant::Int8))
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
