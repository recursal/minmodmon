use anyhow::Error;
use salvo::{handler, Depot, Response};

use crate::agent::agent_service;

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let agent_service_provider = agent_service(depot)?;
    let agent_service = agent_service_provider.manager().await;

    let answer = agent_service.run_placeholder().await?;

    res.render(answer);

    Ok(())
}
