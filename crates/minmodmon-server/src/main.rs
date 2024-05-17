mod api;
mod model;
mod placeholder;
mod sampler;
mod types;

use anyhow::Error;
use salvo::{
    affix::AffixList, conn::TcpListener, logging::Logger, Listener, Router, Server, Service,
};

use crate::model::ModelService;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().init();

    // Create services
    let model_service = ModelService::load().await?;

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
