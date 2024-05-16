mod model;
mod placeholder;
mod sampler;

use anyhow::Error;
use salvo::{affix, conn::TcpListener, logging::Logger, Listener, Router, Server, Service};

use crate::model::ModelService;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().init();

    let model_service = ModelService::load().await?;

    // Configure routes
    let router = Router::new()
        .hoop(affix::inject(model_service))
        .get(placeholder::handle);

    // Configure the service
    let service = Service::new(router).hoop(Logger::new());

    // Start the server
    let acceptor = TcpListener::new("127.0.0.1:5000").bind().await;
    let server = Server::new(acceptor);
    server.serve(service).await;

    Ok(())
}
