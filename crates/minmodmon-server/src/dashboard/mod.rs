use anyhow::Error;
use salvo::{handler, writing::Text, Depot, Response};

use crate::agent::agent_service;

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;
    let manager = service.manager().await;

    let template = include_str!("dashboard.html");
    let info = manager.model_info();

    let rendered = template.replace("$LOADED_MODEL", &info.id);

    res.render(Text::Html(rendered));

    Ok(())
}
