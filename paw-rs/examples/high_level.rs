//! High-level API: dynamic dispatch via `PawFnBuilder::builder()`.
//!
//! The builder handles slug resolution, bundle download, asset caching,
//! and model loading internally, returning `Box<dyn PawFnTrait>`.
//!
//! Usage:
//!   PAW_API_KEY=paw_sk_... cargo run --example high_level

use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> paw_core::Result<()> {
    let config = PawConfig::from_env();

    // Dynamic dispatch: interpreter detected at runtime from the slug
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
