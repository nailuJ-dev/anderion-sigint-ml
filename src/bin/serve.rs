#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;

use anderion_sigint_ml::service::{ServiceState, build_router, load_reference_bundle};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = env::var("SIGINT_MODEL_MANIFEST")?;
    let payload = env::var("SIGINT_MODEL_PAYLOAD")?;
    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let bundle = load_reference_bundle(manifest, payload)?;
    let state = ServiceState::new(bundle.into_pipeline()?);
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
