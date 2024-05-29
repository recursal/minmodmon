use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Error};
use salvo::Depot;
use tokio::sync::Mutex;
use tracing::{event, Level};

use crate::{
    active_model::ActiveModel,
    config::{load_model_configs, ModelConfig},
};

pub fn agent_service(depot: &Depot) -> Result<Arc<AgentService>, Error> {
    depot
        .obtain::<Arc<AgentService>>()
        .ok()
        .cloned()
        .context("failed to get agent service")
}

pub struct AgentService {
    model_configs: HashMap<String, ModelConfig>,
    active_model: Mutex<Option<ActiveModelRef>>,
}

pub type ActiveModelRef = Arc<Mutex<ActiveModel>>;

impl AgentService {
    pub async fn create() -> Result<Self, Error> {
        event!(Level::INFO, "creating agent service");

        let model_configs = load_model_configs().context("failed to load model configs")?;

        let value = AgentService {
            model_configs,
            active_model: Mutex::new(None),
        };

        Ok(value)
    }

    pub fn model_configs(&self) -> &HashMap<String, ModelConfig> {
        &self.model_configs
    }

    pub async fn active_model(&self) -> Option<ActiveModelRef> {
        let value = self.active_model.lock().await;
        value.clone()
    }

    pub async fn active_model_id(&self) -> Option<String> {
        let active_model = self.active_model().await?;
        let active_model = active_model.lock().await;
        Some(active_model.info().id)
    }
}

// TODO: Figure out how to better architect shared managers/services for functions like this.
pub async fn start_activate_model(
    service: Arc<AgentService>,
    id: String,
    quant_nf8: bool,
) -> Result<(), Error> {
    event!(Level::INFO, "activating model {:?}", id);

    let model_config = service
        .model_configs
        .get(&id)
        .context("failed to find model in config")?
        .clone();

    let future = async move {
        let result = activate_model_task(service, id, model_config, quant_nf8).await;
        if let Err(error) = result {
            // TODO: Do something with this
            event!(Level::ERROR, "error while activating model:\n{:?}", error);
        }
    };
    tokio::task::spawn(future);

    Ok(())
}

async fn activate_model_task(
    service: Arc<AgentService>,
    id: String,
    config: ModelConfig,
    quant_nf8: bool,
) -> Result<(), Error> {
    let active_model = ActiveModel::create(id, config, quant_nf8).await?;
    let active_model = Arc::new(Mutex::new(active_model));

    let mut slot = service.active_model.lock().await;
    *slot = Some(active_model);

    Ok(())
}
