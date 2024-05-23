use anyhow::Error;
use salvo::{handler, writing::Text, Depot, Response};
use serde::Serialize;
use tinytemplate::TinyTemplate;

use crate::agent::agent_service;

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;
    let manager = service.manager().await;

    // Prepare template
    let template = include_str!("dashboard.html");
    let mut tt = TinyTemplate::new();
    tt.add_template("dashboard", template)?;

    // Prepare context data
    let info = manager.model_info();
    let context = Context {
        loaded_model: info.id.clone(),
    };

    // Render and reply with the template
    let rendered = tt.render("dashboard", &context)?;
    res.render(Text::Html(rendered));

    Ok(())
}

#[derive(Serialize)]
struct Context {
    loaded_model: String,
}
