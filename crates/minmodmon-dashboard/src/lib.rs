use anyhow::{Context as _, Error};
use salvo::{
    handler,
    writing::{Redirect, Text},
    Depot, Request, Response, Router,
};
use serde::Serialize;
use tinytemplate::TinyTemplate;
use tracing::{event, Level};

use minmodmon_agent::{agent_service, start_activate_model};

pub fn create_router() -> Result<Router, Error> {
    let router = Router::new()
        .get(handle)
        .push(Router::with_path("load_model").post(handle_load_model));

    Ok(router)
}

#[handler]
async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;

    let active_model_id = service.active_model_id().await;

    // Prepare template
    let template = std::fs::read_to_string("./data/dashboard.html")?;
    let mut tt = TinyTemplate::new();
    tt.add_template("dashboard", &template)?;

    // Prepare model context data
    let mut models = Vec::new();
    for (id, info) in service.known_models() {
        let context = ModelContext {
            id: id.clone(),
            available: info.available(),
            download_link: info.config().download_link.clone(),
        };
        models.push(context);
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));

    // Prepare context data
    let context = Context {
        loaded_model: active_model_id.unwrap_or_else(|| "None".to_string()),
        models,
    };

    // Render and reply with the template
    let rendered = tt.render("dashboard", &context)?;
    res.render(Text::Html(rendered));

    Ok(())
}

#[derive(Serialize)]
struct Context {
    loaded_model: String,
    models: Vec<ModelContext>,
}

#[derive(Serialize)]
struct ModelContext {
    id: String,
    available: bool,
    download_link: String,
}

#[handler]
async fn handle_load_model(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), Error> {
    event!(Level::INFO, "dashboard requested load model");

    let service = agent_service(depot)?;

    let model_id = req
        .form::<String>("model-id")
        .await
        .context("failed to get model-id")?;
    let model_quantization = req
        .form::<String>("model-quantization")
        .await
        .context("failed to get model-quantization")?;

    let quant_nf8 = model_quantization == "nf8";
    start_activate_model(service.clone(), model_id, quant_nf8)
        .await
        .context("failed to start model activation")?;

    res.render(Redirect::other("/"));

    Ok(())
}
