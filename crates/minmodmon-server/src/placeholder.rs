use std::collections::HashMap;

use anyhow::{anyhow, Error};
use itertools::Itertools;
use salvo::{handler, Depot, Response};
use tracing::{event, Level};
use web_rwkv::{
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption},
        softmax::softmax_one,
        Submission,
    },
    tensor::{TensorCpu, TensorInit, TensorShape},
};

use crate::{AppState, AppStateRef};

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let state = depot
        .obtain::<AppStateRef>()
        .map_err(|_| anyhow!("failed to get state"))?;

    // Reset state to an initial state
    state.state.load(0, state.initial_state.clone())?;

    // Process the prompt
    let prompt = "You are a helpful writing assistant.\n\nUser: Write a funny parable about a fox jumping over a dog.\n\nAssistant:".to_string();
    let mut prompt_tokenized = state.tokenizer.encode(prompt.as_bytes())?;
    let prompt_last_token = prompt_tokenized.pop().unwrap();

    process_prompt(state, prompt_tokenized).await?;

    // Generate tokens
    let generated = generate_tokens(state, prompt_last_token).await?;

    // Decode the tokenized answer
    let answer_bytes = state.tokenizer.decode(&generated)?;
    let answer = String::from_utf8_lossy(&answer_bytes).into_owned();

    // Send back the result
    res.render(answer);

    Ok(())
}

async fn process_prompt(state: &AppState, tokens: Vec<u16>) -> Result<(), Error> {
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
        state.runtime.send(submission).await?;

        let (out_input, _output) = receiver.await?;
        input = out_input;
    }

    Ok(())
}

async fn generate_tokens(state: &AppState, start: u16) -> Result<Vec<u16>, Error> {
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
        state.runtime.send(submission).await?;

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
        let probabilities = softmax_one(&state.context, logits).await?;
        token = sampler.sample(&probabilities);

        // Remember what we got
        generated.push(token);

        let count = occurrences.get(&token).cloned().unwrap_or(0) + 1;
        occurrences.insert(token, count);
    }

    Ok(generated)
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
