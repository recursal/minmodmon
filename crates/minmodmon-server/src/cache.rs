use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use anyhow::{Context, Error};
use salvo::Depot;
use tokio::sync::Mutex;
use web_rwkv::tensor::TensorCpu;

use minmodmon_agent::types::ChatMessage;

pub fn cache_service(depot: &Depot) -> Result<Arc<CacheService>, Error> {
    depot
        .obtain::<Arc<CacheService>>()
        .ok()
        .cloned()
        .context("failed to get cache service")
}

pub struct CacheService {
    entry: Mutex<Option<CacheEntry>>,
}

struct CacheEntry {
    length: usize,
    hash: u64,
    state: TensorCpu<f32>,
}

impl CacheService {
    pub fn create() -> Result<Self, Error> {
        let value = Self {
            entry: Mutex::new(None),
        };

        Ok(value)
    }

    pub async fn query(&self, messages: &[ChatMessage]) -> Option<(usize, TensorCpu<f32>)> {
        // Check if we have an entry at all
        let slot = self.entry.lock().await;
        let entry = slot.as_ref()?;

        // If our entry is too long, it can't possibly match
        if entry.length > messages.len() {
            return None;
        }

        // Hash the range of messages that might be in the cache
        let messages = &messages[0..entry.length];
        let hash = hash_messages(messages);

        if entry.hash == hash {
            return Some((entry.length, entry.state.clone()));
        }

        // Cached entry doesn't match
        None
    }

    pub async fn set(&self, messages: &[ChatMessage], state: TensorCpu<f32>) {
        let hash = hash_messages(messages);
        let entry = CacheEntry {
            length: messages.len(),
            hash,
            state,
        };
        let mut slot = self.entry.lock().await;
        *slot = Some(entry);
    }
}

fn hash_messages(messages: &[ChatMessage]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for message in messages {
        message.hash(&mut hasher);
    }
    hasher.finish()
}
