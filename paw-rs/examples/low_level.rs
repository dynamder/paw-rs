//! Low-level API example using the default backend.
//!
//! Usage:
//!   PAW_API_KEY=paw_sk_... cargo run --example low_level
//!   PAW_API_KEY=paw_sk_... cargo run --example low_level --features candle

use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> paw_core::Result<()> {
    let config = PawConfig::from_env();

    let mut f = PawFnBuilder::builder()
        .config(config)
        .spec("Classify sentiment: return POSITIVE or NEGATIVE")
        .ephemeral(true)
        .compile()
        .await?;

    let result = f.run("I love this product!")?;
    println!("output: {result}");
    Ok(())
}
