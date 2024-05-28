use anyhow::Error;
use salvo::{
    handler,
    writing::{Redirect, Text},
    Depot, Response, Router,
};
use serde::Serialize;
use tinytemplate::TinyTemplate;

use minmodmon_agent::{activate_model, agent_service};

pub fn create_router() -> Result<Router, Error> {
    let router = Router::new()
        .get(handle)
        .push(Router::with_path("load_model").post(handle_load_model));

    Ok(router)
}

#[handler]
async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;
    let mut manager = service.manager().await;

    // Prepare template
    let template = include_str!("dashboard.html");
    let mut tt = TinyTemplate::new();
    tt.add_template("dashboard", template)?;

    // Prepare context data
    let info = manager.active_model().map(|m| m.info());
    let models = vec![
        ModelContext {
            id: "recursal-eaglex-v2".to_string(),
        },
        ModelContext {
            id: "recursal-eaglex-chat-v1".to_string(),
        },
    ];
    let context = Context {
        loaded_model: info.map_or_else(|| "None".to_string(), |i| i.id.clone()),
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
}

#[handler]
async fn handle_load_model(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;

    activate_model(service.clone(), "recursal-eaglex-v2".to_string()).await?;

    res.render(Redirect::other("/"));

    Ok(())
}
