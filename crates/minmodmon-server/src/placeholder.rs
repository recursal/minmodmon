use anyhow::Error;
use salvo::{handler, Depot, Response};

use crate::agent::agent_service;

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let agent_service_provider = agent_service(depot)?;
    let agent_service = agent_service_provider.manager().await;

    // Run the placeholder inference task
    agent_service.reset_state()?;
    agent_service
        .process_message("System", "You are a helpful writing assistant.")
        .await?;
    agent_service
        .process_message(
            "User",
            "Write a funny parable about a fox jumping over a dog.",
        )
        .await?;
    let answer = agent_service.generate_message().await?;

    // Send the answer back
    res.render(answer);

    Ok(())
}
