use std::sync::Arc;

use anyhow::{Context as _, Error};
use tracing::{event, Level};

use crate::{
    active_model::ActiveModel,
    config::{Config, ModelConfig},
    AgentService,
};

pub struct AgentManager {
    config: Arc<Config>,
    active_model: Option<ActiveModel>,
}

impl AgentManager {
    pub async fn create(config: Arc<Config>) -> Result<Self, Error> {
        event!(Level::INFO, "creating agent service");

        let value = AgentManager {
            config,
            active_model: None,
        };

        Ok(value)
    }

    pub fn active_model(&mut self) -> Option<&mut ActiveModel> {
        self.active_model.as_mut()
    }
}

// TODO: Figure out how to better architect shared managers/services for functions like this.
pub async fn activate_model(service: AgentService, id: String) -> Result<(), Error> {
    event!(Level::INFO, "activating model {:?}", id);

    let manager = service.manager().await;
    let model_config = manager
        .config
        .models
        .get(&id)
        .context("failed to find active model in config")?
        .clone();
    drop(manager);

    let future = async move {
        let result = activate_model_task(service, id, model_config).await;
        if let Err(error) = result {
            // TODO: Do something with this
            event!(Level::ERROR, "error while activating model:\n{:?}", error);
        }
    };
    tokio::task::spawn(future);

    Ok(())
}

async fn activate_model_task(
    service: AgentService,
    id: String,
    config: ModelConfig,
) -> Result<(), Error> {
    let active_model = ActiveModel::create(id, config).await?;
    let mut manager = service.manager().await;
    manager.active_model = Some(active_model);
    Ok(())
}
