mod placeholder;

use std::{fs::File, sync::Arc};

use anyhow::Error;
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use salvo::{affix, conn::TcpListener, logging::Logger, Listener, Router, Server, Service};
use tracing::{event, Level};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{loader::Loader, ContextAutoLimits},
    runtime::{
        infer::{InferInput, InferOutput},
        model::{Build, ModelBuilder, ModelRuntime, Quant, State},
        v5, JobRuntime,
    },
    tensor::TensorCpu,
    tokenizer::Tokenizer,
    wgpu::PowerPreference,
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().init();

    let state = create_app_state().await?;
    let state_ref = AppStateRef::new(state);
    let inject_state = affix::inject(state_ref);

    // Configure routes
    let router = Router::new().hoop(inject_state).get(placeholder::handle);

    // Configure the service
    let service = Service::new(router).hoop(Logger::new());

    // Start the server
    let acceptor = TcpListener::new("127.0.0.1:5000").bind().await;
    let server = Server::new(acceptor);
    server.serve(service).await;

    Ok(())
}

struct AppState {
    tokenizer: Tokenizer,
    context: Context,
    runtime: JobRuntime<InferInput, InferOutput<f16>>,
    state: Box<dyn State + Send + Sync>,
    initial_state: TensorCpu<f32>,
}

type AppStateRef = Arc<AppState>;

async fn create_app_state() -> Result<AppState, Error> {
    let model_path = "../EagleX_v2/EagleX-v2.st";
    let tokenizer_path = "../EagleX_v2/rwkv_vocab_v20230424.json";

    event!(Level::INFO, "creating application state");

    // Load the tokenizer
    let contents = std::fs::read_to_string(tokenizer_path)?;
    let tokenizer = Tokenizer::new(&contents)?;

    // Load the model
    let (context, runtime, state) = load_model(model_path).await?;

    // Get the initial state if we need to reset
    let initial_state = state.back(0).await?;

    let value = AppState {
        context,
        tokenizer,
        runtime,
        state,
        initial_state,
    };
    Ok(value)
}

async fn load_model(
    path: &str,
) -> Result<
    (
        Context,
        JobRuntime<InferInput, InferOutput<f16>>,
        Box<dyn State + Send + Sync>,
    ),
    Error,
> {
    event!(Level::INFO, path, "loading model");

    // Preload the model
    let file = File::open(path)?;
    let data = unsafe { Mmap::map(&file)? };

    let safetensors = SafeTensors::deserialize(&data)?;
    let model_info = Loader::info(&safetensors)?;

    // Prepare a context for the model
    let instance = Instance::new();
    let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
    let context = ContextBuilder::new(adapter)
        .with_auto_limits(&model_info)
        .build()
        .await?;

    // Quantize all layers to 8-bit
    let quantize = (0..model_info.num_layer)
        .map(|layer| (layer, Quant::Int8))
        .collect();

    // Configure the model
    let builder = ModelBuilder::new(&context, safetensors).with_quant(quantize);

    // Build the runtime, actually loading weights
    let builder = Build::<v5::ModelJobBuilder<f16>>::build(builder).await?;
    let state = builder.state();
    let runtime = JobRuntime::new(builder).await;

    Ok((context, runtime, Box::new(state)))
}
