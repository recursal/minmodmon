mod agent;
mod api;
mod placeholder;
mod sampler;
mod types;

use std::collections::HashMap;

use anyhow::Error;
use salvo::{
    affix::AffixList, conn::TcpListener, logging::Logger, Listener, Router, Server, Service,
};
use serde::{Deserialize, Serialize};

use crate::agent::AgentService;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().init();

    // Load configuration
    let config_str = std::fs::read_to_string("./Config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    // Create services
    let model_service = AgentService::create(&config).await?;

    // Configure routes
    let api_router = api::create_router()?;
    let router = Router::new().get(placeholder::handle).push(api_router);

    // Configure the service
    let affix = AffixList::new().inject(model_service);
    let service = Service::new(router).hoop(Logger::new()).hoop(affix);

    // Start the server
    let acceptor = TcpListener::new("127.0.0.1:5000").bind().await;
    let server = Server::new(acceptor);
    server.serve(service).await;

    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub active_model: String,
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelConfig {
    pub weights: String,
    pub vocab: String,
}
