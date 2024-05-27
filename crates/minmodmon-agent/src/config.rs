use std::{collections::HashMap, sync::Arc};

use anyhow::Error;
use serde::{Deserialize, Serialize};

pub fn load_config() -> Result<Arc<Config>, Error> {
    let config_str = std::fs::read_to_string("./Config.toml")?;
    let value = toml::from_str(&config_str)?;
    Ok(Arc::new(value))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub active_model: String,
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub weights: String,
    pub vocab: String,
    pub role_system: RoleConfig,
    pub role_user: RoleConfig,
    pub role_assistant: RoleConfig,
    pub stop_sequence: Vec<u16>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoleConfig {
    pub prefix: Vec<u16>,
    pub suffix: Vec<u16>,
}
