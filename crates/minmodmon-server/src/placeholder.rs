use anyhow::Error;
use salvo::{handler, Depot, Response};

use crate::model::get_model_service;

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let model_service = get_model_service(depot)?;

    let answer = model_service.run_placeholder().await?;

    res.render(answer);

    Ok(())
}
