mod api;
mod cache;

use std::sync::Arc;

use anyhow::{Context, Error};
use salvo::{
    affix::AffixList, conn::TcpListener, logging::Logger, Listener, Router, Server, Service,
};

use minmodmon_agent::AgentService;

use crate::cache::CacheService;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().init();

    // Create services
    let model_service = AgentService::create()
        .await
        .context("failed to create agent service")?;
    let cache_service = CacheService::create().context("failed to create cache service")?;

    // Configure routes
    let dashboard_router = minmodmon_dashboard::create_router()?;
    let api_router = api::create_router()?;
    let router = Router::new().push(dashboard_router).push(api_router);

    // Configure the service
    let affix = AffixList::new()
        .inject(Arc::new(model_service))
        .inject(Arc::new(cache_service));
    let service = Service::new(router).hoop(Logger::new()).hoop(affix);

    // Start the server
    let acceptor = TcpListener::new("127.0.0.1:5000").bind().await;
    let server = Server::new(acceptor);
    server.serve(service).await;

    Ok(())
}
