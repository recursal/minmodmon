use std::collections::HashMap;

use anyhow::Error;
use itertools::Itertools;
use web_rwkv::tensor::{TensorCpu, TensorInit, TensorShape};

// TODO: This entire sampling system needs a re-work:
// - should it parse the input too? is it per-message?
// - in general it needs to be redesigned more clear and resilient
// - support different kinds of samplers as a configured option

pub struct Sampler {
    occurrences: HashMap<u16, u32>,
    top_p: f32,
    temperature: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            occurrences: HashMap::new(),
            top_p: 0.5,
            temperature: 0.8,
            presence_penalty: 0.3,
            frequency_penalty: 0.3,
        }
    }
}

impl Sampler {
    pub fn apply_penalties(&self, logits: &TensorCpu<f32>) -> Result<TensorCpu<f32>, Error> {
        // Convert to f32 to work with it
        let shape = logits.shape();
        let mut logits: Vec<_> = logits.iter().cloned().collect();

        // Apply repetition penalties
        logits[0] = f32::NEG_INFINITY;
        for (&token, &count) in &self.occurrences {
            let penalty = self.presence_penalty + count as f32 * self.frequency_penalty;
            logits[token as usize] -= penalty;
        }
        let logits = TensorCpu::from_data(shape, logits)?;

        Ok(logits)
    }

    #[allow(unused)]
    pub fn sample(&self, logits: &[f32]) -> u16 {
        let sorted: Vec<_> = logits
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

    pub fn sample_min_t(&self, logits: &[f32]) -> u16 {
        let max = logits.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));
        let min = logits.iter().fold(f32::INFINITY, |acc, x| acc.min(*x));

        let mut dart = fastrand::f32();
        let power = 1.0 - f32::powf(dart, self.temperature * f32::powf(dart, 10.0));
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
