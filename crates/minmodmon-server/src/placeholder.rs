use anyhow::{anyhow, Error};
use salvo::{handler, Depot, Response};

use crate::model::ModelService;

#[handler]
pub async fn handle(depot: &mut Depot, res: &mut Response) -> Result<(), Error> {
    let model_service = depot
        .obtain::<ModelService>()
        .map_err(|_| anyhow!("failed to get model service"))?;

    let answer = model_service.run_placeholder().await?;

    // Send back the result
    res.render(answer);

    Ok(())
}
