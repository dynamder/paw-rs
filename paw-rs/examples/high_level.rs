//! High-level API example: compile a program and run inference.
//!
//! Uses the type-state builder (`PawFnBuilder`) which handles slug resolution,
//! bundle download, asset caching, and model loading internally.
//!
//! Usage:
//!   PAW_API_KEY=paw_sk_... cargo run --example high_level

use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> paw_core::Result<()> {
    // Config from environment (PAW_API_KEY, PAW_API_URL, etc.)
    let config = PawConfig::from_env();

    // Build a new program via the PAW API
    let mut f = PawFnBuilder::builder()
        .config(config)
        .spec("Classify sentiment: return POSITIVE or NEGATIVE")
        .ephemeral(true)
        .compile()
        .await?;

    // Run inference
    let input = "I love this product!";
    let result = f.run(input)?;
    println!("input:  {input}");
    println!("output: {result}");

    Ok(())
}
