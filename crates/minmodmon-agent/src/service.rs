use std::{
    collections::HashMap,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use anyhow::{bail, Context as _, Error};
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
    known_models: HashMap<String, KnownModelInfo>,
    active_model: Mutex<Option<ActiveModelRef>>,
    loading: AtomicBool,
}

pub struct KnownModelInfo {
    config: ModelConfig,
    available: bool,
}

pub type ActiveModelRef = Arc<Mutex<ActiveModel>>;

impl AgentService {
    pub async fn create() -> Result<Self, Error> {
        event!(Level::INFO, "creating agent service");

        let model_configs = load_model_configs().context("failed to load model configs")?;
        let known_models = model_configs
            .into_iter()
            .map(|(id, config)| {
                // Check if the safetensors file for this model exists
                let weights_path = format!("data/{}.st", id);
                let weights_path = Path::new(&weights_path);
                let available = weights_path.exists();

                let info = KnownModelInfo { config, available };

                (id, info)
            })
            .collect();

        let value = AgentService {
            known_models,
            active_model: Mutex::new(None),
            loading: AtomicBool::new(false),
        };

        Ok(value)
    }

    pub fn known_models(&self) -> &HashMap<String, KnownModelInfo> {
        &self.known_models
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

    pub fn loading(&self) -> bool {
        self.loading.load(Ordering::SeqCst)
    }
}

impl KnownModelInfo {
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn available(&self) -> bool {
        self.available
    }
}

// TODO: Figure out how to better architect shared managers/services for functions like this.
pub async fn start_activate_model(
    service: Arc<AgentService>,
    id: String,
    quant_nf8: bool,
) -> Result<(), Error> {
    event!(Level::INFO, "activating model {:?}", id);

    let model_info = service
        .known_models
        .get(&id)
        .context("failed to find model in config")?;

    if !model_info.available {
        bail!("model not available")
    }

    let config = model_info.config.clone();
    let future = async move {
        service.loading.store(true, Ordering::SeqCst);

        let result = activate_model_task(service.clone(), id, config, quant_nf8).await;

        if let Err(error) = result {
            // TODO: Do something with this in the dashboard
            event!(Level::ERROR, "error while activating model:\n{:?}", error);
        }

        service.loading.store(false, Ordering::SeqCst);
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
    // Unload any existing model, if there is one
    {
        let mut slot = service.active_model.lock().await;
        *slot = None;
    }

    // Load the new model
    let active_model = ActiveModel::create(id, config, quant_nf8).await?;
    let active_model = Arc::new(Mutex::new(active_model));

    // Store the new model
    {
        let mut slot = service.active_model.lock().await;
        *slot = Some(active_model);
    }

    Ok(())
}
