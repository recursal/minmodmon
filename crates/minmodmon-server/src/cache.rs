use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use anyhow::{Context, Error};
use salvo::Depot;
use tokio::sync::{Mutex, MutexGuard};
use web_rwkv::tensor::TensorCpu;

use crate::types::ChatMessage;

#[derive(Clone)]
pub struct CacheService {
    service: Arc<Mutex<CacheManager>>,
}

impl CacheService {
    pub fn create() -> Result<Self, Error> {
        let manager = CacheManager::create()?;
        let value = Self {
            service: Arc::new(Mutex::new(manager)),
        };

        Ok(value)
    }

    pub async fn manager(&self) -> MutexGuard<CacheManager> {
        self.service.lock().await
    }
}

pub fn cache_service(depot: &Depot) -> Result<&CacheService, Error> {
    depot
        .obtain::<CacheService>()
        .ok()
        .context("failed to get cache service")
}

pub struct CacheManager {
    entry: Option<CacheEntry>,
}

struct CacheEntry {
    length: usize,
    hash: u64,
    state: TensorCpu<f32>,
}

impl CacheManager {
    fn create() -> Result<Self, Error> {
        let value = Self { entry: None };

        Ok(value)
    }

    pub fn query(&self, messages: &[ChatMessage]) -> Option<(usize, &TensorCpu<f32>)> {
        // Check if we have an entry at all
        let entry = self.entry.as_ref()?;

        // If our entry is too long, it can't possibly match
        if entry.length > messages.len() {
            return None;
        }

        // Hash the range of messages that might be in the cache
        let messages = &messages[0..entry.length];
        let hash = hash_messages(messages);

        if entry.hash == hash {
            return Some((entry.length, &entry.state));
        }

        // Cached entry doesn't match
        None
    }

    pub fn set(&mut self, messages: &[ChatMessage], state: TensorCpu<f32>) {
        let hash = hash_messages(messages);
        let entry = CacheEntry {
            length: messages.len(),
            hash,
            state,
        };
        self.entry = Some(entry);
    }
}

fn hash_messages(messages: &[ChatMessage]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for message in messages {
        message.hash(&mut hasher);
    }
    hasher.finish()
}
