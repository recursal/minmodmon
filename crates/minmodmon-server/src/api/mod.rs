use std::time::SystemTime;

use anyhow::Error;
use salvo::{handler, writing::Json, Depot, Request, Response, Router};

use crate::cache::cache_service;
use crate::{
    agent::agent_service,
    types::{ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, ModelList, UsageReport},
};

pub fn create_router() -> Result<Router, Error> {
    let router = Router::with_path("api")
        .push(Router::with_path("models").get(handle_models))
        .push(Router::with_path("chat/completions").post(handle_chat_completions));

    Ok(router)
}

#[handler]
async fn handle_models(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;
    let manager = service.manager().await;

    let info = manager.model_info();
    let list = ModelList {
        object: "list".to_string(),
        data: vec![info],
    };

    res.render(Json(list));

    Ok(())
}

#[handler]
async fn handle_chat_completions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), Error> {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let service = agent_service(depot)?;
    let cache = cache_service(depot)?;
    let manager = service.manager().await;
    let mut cache = cache.manager().await;

    // Parse the input
    let request = req.parse_json::<ChatRequest>().await?;

    // Check if we can restore from cache
    let mut skipped = 0;
    if let Some((length, state)) = cache.query(&request.messages) {
        skipped = length;
        manager.import_state(state.clone())?;
    } else {
        manager.reset_state()?;
    }

    // Process remaining messages
    for message in &request.messages[skipped..] {
        manager.process_message(message).await?;
    }

    // Cache current state, after processing given non-cached messages
    let state = manager.export_state().await?;
    cache.set(&request.messages, state);

    // Generate output
    let message = manager.generate_message().await?;

    // Serialize and send back the result
    let message = ChatMessage {
        role: "assistant".to_string(),
        content: message,
    };
    let choice = ChatResponseChoice {
        index: 0,
        message,
        finish_reason: "stop".to_string(),
    };
    let usage = UsageReport {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let response = ChatResponse {
        id: format!("req-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: manager.model_info().id,
        choices: vec![choice],
        usage,
    };
    res.render(Json(response));

    Ok(())
}
