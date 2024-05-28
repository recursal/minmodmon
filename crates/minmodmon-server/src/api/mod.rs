use std::time::SystemTime;

use anyhow::{Context, Error};
use minmodmon_agent::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, ModelList, UsageReport,
};
use salvo::{handler, writing::Json, Depot, Request, Response, Router};
use tracing::{event, Level};

use minmodmon_agent::agent_service;

use crate::cache::cache_service;

pub fn create_router() -> Result<Router, Error> {
    let router = Router::with_path("api")
        .push(Router::with_path("models").get(handle_models))
        .push(Router::with_path("chat/completions").post(handle_chat_completions));

    Ok(router)
}

#[handler]
async fn handle_models(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;
    let mut manager = service.manager().await;

    let mut list = ModelList {
        object: "list".to_string(),
        data: Vec::new(),
    };

    if let Some(model) = manager.active_model() {
        let info = model.info();
        list.data.push(info);
    }

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
    let mut manager = service.manager().await;
    let mut cache = cache.manager().await;

    // Get the current model
    let model = manager
        .active_model()
        .context("failed to get active model")?;

    // Parse the input
    let request = req.parse_json::<ChatRequest>().await?;

    // Check if we can restore from cache
    let mut skipped = 0;
    if let Some((length, state)) = cache.query(&request.messages) {
        event!(Level::INFO, length, "restoring from cached state");
        skipped = length;
        model.import_state(state.clone())?;
    } else {
        event!(Level::INFO, "could not restore from cached state, no match");
        model.reset_state()?;
    }

    // Process remaining messages
    for message in &request.messages[skipped..] {
        model.process_message(message).await?;
    }

    // Cache current state, after processing given non-cached messages
    let state = model.export_state().await?;
    cache.set(&request.messages, state);

    // Generate output
    let content = model.generate_message().await?;

    // Serialize and send back the result
    let message = ChatMessage {
        role: "assistant".to_string(),
        content,
    };
    let choice = ChatResponseChoice {
        index: 0,
        message,
        // TODO: This needs to actually distinguish between if we stopped because we ran to the
        //  token limit, or because the message is done.
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
        model: model.info().id,
        choices: vec![choice],
        usage,
    };
    res.render(Json(response));

    Ok(())
}
