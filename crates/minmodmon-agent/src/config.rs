use std::collections::HashMap;

use anyhow::{Context, Error};
use serde::{Deserialize, Serialize};

pub(crate) fn load_model_configs() -> Result<HashMap<String, ModelConfig>, Error> {
    let mut model_configs = HashMap::new();

    for entry in std::fs::read_dir("./data")? {
        let entry = entry?;

        let path = entry.path();
        let is_toml = path.extension().map(|v| v == "toml").unwrap_or(false);

        if !is_toml {
            continue;
        }

        let name = path.file_stem().context("failed to get file name")?;
        let name = name.to_string_lossy().to_string();

        // Parse the file contents
        let config_str = std::fs::read_to_string(path)?;
        let value: ModelConfig = toml::from_str(&config_str)?;

        model_configs.insert(name, value);
    }

    Ok(model_configs)
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
