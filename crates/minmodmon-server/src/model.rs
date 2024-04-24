use std::{collections::HashMap, fs::File, sync::Arc};

use anyhow::Error;
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::{event, Level};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{loader::Loader, ContextAutoLimits},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        model::{Build, ModelBuilder, ModelRuntime, Quant, State},
        softmax::softmax_one,
        v5, JobRuntime, Submission,
    },
    tensor::{TensorCpu, TensorInit, TensorShape},
    tokenizer::Tokenizer,
    wgpu::PowerPreference,
};

#[derive(Clone)]
pub struct ModelService {
    inner: Arc<ModelServiceInner>,
}

struct ModelServiceInner {
    tokenizer: Tokenizer,
    context: Context,
    runtime: JobRuntime<InferInput, InferOutput<f16>>,
    state: Box<dyn State + Send + Sync>,
    initial_state: TensorCpu<f32>,
}

impl ModelService {
    pub async fn load() -> Result<Self, Error> {
        event!(Level::INFO, "loading model service");

        let model_path = "../EagleX_v2/EagleX-v2.st";
        let tokenizer_path = "../EagleX_v2/rwkv_vocab_v20230424.json";

        // Load the tokenizer
        let contents = std::fs::read_to_string(tokenizer_path)?;
        let tokenizer = Tokenizer::new(&contents)?;

        // Load the model
        let (context, runtime, state) = load_model(model_path).await?;

        // Get the initial state if we need to reset
        let initial_state = state.back(0).await?;

        let inner = ModelServiceInner {
            context,
            tokenizer,
            runtime,
            state,
            initial_state,
        };
        let value = Self {
            inner: Arc::new(inner),
        };
        Ok(value)
    }

    pub async fn run_placeholder(&self) -> Result<String, Error> {
        // Reset state to an initial state
        self.inner.state.load(0, self.inner.initial_state.clone())?;

        // Process the prompt
        let prompt = "You are a helpful writing assistant.\n\nUser: Write a funny parable about a fox jumping over a dog.\n\nAssistant:".to_string();
        let mut prompt_tokenized = self.inner.tokenizer.encode(prompt.as_bytes())?;
        let prompt_last_token = prompt_tokenized.pop().unwrap();

        self.process_prompt(prompt_tokenized).await?;

        // Generate tokens
        let generated = self.generate_tokens(prompt_last_token).await?;

        // Decode the tokenized answer
        let answer_bytes = self.inner.tokenizer.decode(&generated)?;
        let answer = String::from_utf8_lossy(&answer_bytes).into_owned();

        Ok(answer)
    }

    async fn process_prompt(&self, tokens: Vec<u16>) -> Result<(), Error> {
        event!(Level::DEBUG, count = tokens.len(), "processing prompt");

        // Process initial prompt (minus last token)
        let batch = InferInputBatch {
            tokens,
            option: InferOption::Last,
        };
        let mut input = InferInput::new(vec![batch], 32);

        while !input.batches[0].tokens.is_empty() {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let submission = Submission { input, sender };
            self.inner.runtime.send(submission).await?;

            let (out_input, _output) = receiver.await?;
            input = out_input;
        }

        Ok(())
    }

    async fn generate_tokens(&self, start: u16) -> Result<Vec<u16>, Error> {
        let sampler = Sampler {
            top_p: 0.5,
            temperature: 0.8,
            presence_penalty: 0.3,
            frequency_penalty: 0.3,
        };

        let mut generated = Vec::new();
        let mut occurrences = HashMap::new();

        let mut token = start;

        for _ in 0..256 {
            // Run model input
            let batch = InferInputBatch {
                tokens: vec![token],
                option: InferOption::Last,
            };
            let input = InferInput::new(vec![batch], 32);

            let (sender, receiver) = tokio::sync::oneshot::channel();
            let submission = Submission { input, sender };
            self.inner.runtime.send(submission).await?;

            let (_input, output) = receiver.await?;
            let logits = &output[0].0;

            // Convert to f32 to work with it
            let shape = logits.shape();
            let mut logits: Vec<_> = logits.iter().map(|x| x.to_f32()).collect();

            // Apply repetition penalties
            logits[0] = f32::NEG_INFINITY;
            for (&token, &count) in &occurrences {
                let penalty = sampler.presence_penalty + count as f32 * sampler.frequency_penalty;
                logits[token as usize] -= penalty;
            }

            // Predict next token
            let logits = TensorCpu::from_data(shape, logits)?;
            let probabilities = softmax_one(&self.inner.context, logits).await?;
            token = sampler.sample(&probabilities);

            // Remember what we got
            generated.push(token);

            let count = occurrences.get(&token).cloned().unwrap_or(0) + 1;
            occurrences.insert(token, count);
        }

        Ok(generated)
    }
}

async fn load_model(
    path: &str,
) -> Result<
    (
        Context,
        JobRuntime<InferInput, InferOutput<f16>>,
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
    let instance = Instance::new();
    let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
    let context = ContextBuilder::new(adapter)
        .with_auto_limits(&model_info)
        .build()
        .await?;

    // Quantize all layers to 8-bit
    let quantize = (0..model_info.num_layer)
        .map(|layer| (layer, Quant::Int8))
        .collect();

    // Configure the model
    let builder = ModelBuilder::new(&context, safetensors).with_quant(quantize);

    // Build the runtime, actually loading weights
    let builder = Build::<v5::ModelJobBuilder<f16>>::build(builder).await?;
    let state = builder.state();
    let runtime = JobRuntime::new(builder).await;

    Ok((context, runtime, Box::new(state)))
}

struct Sampler {
    top_p: f32,
    temperature: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl Sampler {
    pub fn sample(&self, probabilities: &[f32]) -> u16 {
        let sorted: Vec<_> = probabilities
            .iter()
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temperature)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u16
    }
}
