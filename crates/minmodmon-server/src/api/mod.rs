use anyhow::Error;
use salvo::{handler, writing::Json, Depot, Response, Router};

use crate::{model::get_model_service, types::ModelList};

pub fn create_router() -> Result<Router, Error> {
    let router = Router::with_path("api").push(Router::with_path("models").get(handle_models));

    Ok(router)
}

#[handler]
fn handle_models(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let model_service = get_model_service(depot)?;

    let info = model_service.model_info();
    let list = ModelList {
        object: "list".to_string(),
        data: vec![info],
    };

    res.render(Json(list));

    Ok(())
}
