use anyhow::Error;
use salvo::{handler, Depot, Response};

use crate::{agent::agent_service, types::ChatMessage};

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let service = agent_service(depot)?;
    let manager = service.manager().await;

    // Run the placeholder inference task
    manager.reset_state()?;

    let message = ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful writing assistant.".to_string(),
    };
    manager.process_message(&message).await?;

    let message = ChatMessage {
        role: "user".to_string(),
        content: "Write a funny parable about a fox jumping over a dog.".to_string(),
    };
    manager.process_message(&message).await?;

    let answer = manager.generate_message().await?;

    // Send the answer back
    res.render(answer);

    Ok(())
}
