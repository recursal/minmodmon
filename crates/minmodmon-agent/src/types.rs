//! Serializable common API types.
//!
//! Matches completion API standards used by many platforms.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information metadata.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize, Hash, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatResponseChoice>,
    pub usage: UsageReport,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatResponseChoice {
    pub index: usize,
    pub message: ChatMessage,

    /// Should only be "stop" or "length" in minmodmon.
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UsageReport {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
