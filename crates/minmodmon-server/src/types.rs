//! Serializable common API types.
//!
//! Matches completion API standards used by many platforms.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information metadata.
#[derive(Serialize, Deserialize, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatResponseChoice>,
    pub usage: UsageReport,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatResponseChoice {
    pub index: usize,
    pub message: ChatMessage,

    /// Should only be "stop" or "length" in midmodmon.
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UsageReport {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}