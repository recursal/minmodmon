use std::sync::Arc;

use anyhow::{Context as _, Error};
use salvo::Depot;
use tokio::sync::{Mutex, MutexGuard};

use crate::{agent::AgentManager, config::Config};

#[derive(Clone)]
pub struct AgentService {
    /// Despite that the service technically can be used without locking, we want to avoid race
    /// conditions in the loaded on-gpu model state.
    service: Arc<Mutex<AgentManager>>,
}

impl AgentService {
    pub async fn create(config: Arc<Config>) -> Result<Self, Error> {
        let service = AgentManager::create(config).await?;
        let value = Self {
            service: Arc::new(Mutex::new(service)),
        };

        Ok(value)
    }

    pub async fn manager(&self) -> MutexGuard<AgentManager> {
        self.service.lock().await
    }
}

pub fn agent_service(depot: &Depot) -> Result<&AgentService, Error> {
    depot
        .obtain::<AgentService>()
        .ok()
        .context("failed to get agent service")
}
