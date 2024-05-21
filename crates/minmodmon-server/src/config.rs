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
    pub system_prefix: Vec<u16>,
    pub system_suffix: Vec<u16>,
    pub user_prefix: Vec<u16>,
    pub user_suffix: Vec<u16>,
    pub assistant_prefix: Vec<u16>,
    pub assistant_suffix: Vec<u16>,
    pub stop_sequence: Vec<u16>,
}
