//! Serializable common API types.
//!
//! Matches completion API standards used by many platforms.

use serde::Serialize;

#[derive(Serialize, Debug)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information metadata.
#[derive(Serialize, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
