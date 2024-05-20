use anyhow::Error;
use salvo::{handler, writing::Json, Depot, Response, Router};

use crate::{agent::agent_service, types::ModelList};

pub fn create_router() -> Result<Router, Error> {
    let router = Router::with_path("api").push(Router::with_path("models").get(handle_models));

    Ok(router)
}

#[handler]
async fn handle_models(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let agent_service_provider = agent_service(depot)?;
    let agent_service = agent_service_provider.manager().await;

    let info = agent_service.model_info();
    let list = ModelList {
        object: "list".to_string(),
        data: vec![info],
    };

    res.render(Json(list));

    Ok(())
}
