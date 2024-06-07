use std::collections::HashMap;

use anyhow::Error;
use web_rwkv::tensor::{TensorCpu, TensorInit, TensorShape};

pub struct SamplerSettings {
    pub temperature: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// TODO: Refactor into a cleaner separate `SamplerSettings`/`FrequencyState`.
#[derive(Default)]
pub struct Sampler {
    occurrences: HashMap<u16, u32>,
}

impl Sampler {
    pub fn apply_penalties(
        &self,
        settings: &SamplerSettings,
        logits: &TensorCpu<f32>,
    ) -> Result<TensorCpu<f32>, Error> {
        // Convert to f32 to work with it
        let shape = logits.shape();
        let mut logits: Vec<_> = logits.iter().cloned().collect();

        // Apply repetition penalties
        logits[0] = f32::NEG_INFINITY;
        for (&token, &count) in &self.occurrences {
            let penalty = settings.presence_penalty + count as f32 * settings.frequency_penalty;
            logits[token as usize] -= penalty;
        }
        let logits = TensorCpu::from_data(shape, logits)?;

        Ok(logits)
    }

    pub fn sample(&self, settings: &SamplerSettings, logits: &[f32]) -> u16 {
        let max = logits.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));
        let min = logits.iter().fold(f32::INFINITY, |acc, x| acc.min(*x));

        let mut dart = fastrand::f32();
        let power = 1.0 - f32::powf(dart, settings.temperature * f32::powf(dart, 10.0));
        dart = f32::powf(dart, power);
        dart = min + dart * (max - min);

        let (index, _) = logits
            .iter()
            .enumerate()
            .reduce(|x, y| {
                let xv = (x.1 - dart).abs();
                let yv = (y.1 - dart).abs();

                if xv < yv {
                    x
                } else {
                    y
                }
            })
            .unwrap();

        index as u16
    }

    pub fn consume_token(&mut self, token: u16) {
        let count = self.occurrences.get(&token).cloned().unwrap_or(0) + 1;
        self.occurrences.insert(token, count);
    }
}
