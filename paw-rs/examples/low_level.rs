//! Low-level API example: full manual pipeline without the type-state builder.
//!
//! Demonstrates each step explicitly:
//!   1. List compilers → pick GPT-2
//!   2. Compile a spec
//!   3. Download the .paw bundle
//!   4. Ensure base model + tokenizer are cached
//!   5. Load model via PawFnLoader
//!   6. Run inference
//!
//! Usage:
//!   PAW_API_KEY=paw_sk_... cargo run --example low_level

use paw_core::prelude::*;
use paw_core::CompileRequest;

#[tokio::main]
async fn main() -> Result<()> {
    // ── 0. Config from environment ────────────────────────────────────
    let config = PawConfig::from_env();
    config
        .effective_api_key()
        .ok_or_else(|| Error::Other("PAW_API_KEY required".into()))?;
    let client = PawClient::new(&config);

    // ── 1. Pick a compiler ────────────────────────────────────────────
    println!("[1/6] Finding GPT-2 compiler...");
    let compilers = client.list_compilers().await?;
    let compiler = compilers
        .iter()
        .find(|c| c.name.to_lowercase().contains("gpt2"))
        .map(|c| c.name.clone())
        .unwrap_or_else(|| {
            eprintln!("Available compilers:");
            for c in &compilers {
                eprintln!("  {}: {}", c.name, c.display_name);
            }
            panic!("No GPT-2 compiler found");
        });
    println!("  using: {compiler}");

    // ── 2. Compile ────────────────────────────────────────────────────
    println!("\n[2/6] Compiling new program...");
    let program = client
        .compile(
            CompileRequest::builder()
                .spec("Classify sentiment: return POSITIVE or NEGATIVE")
                .compiler(&compiler)
                .ephemeral(true)
                .build()?,
        )
        .await?;
    println!("  program_id: {}, status: {}", program.id, program.status);

    // ── 3. Download .paw bundle ───────────────────────────────────────
    println!("\n[3/6] Downloading .paw bundle...");
    let dir = client.download_paw(&program.id).await?;
    let bundle = PawBundle::load_from_dir(&dir)?;
    println!("  interpreter: {}", bundle.interpreter_model());

    // ── 4. Cache base model + tokenizer ───────────────────────────────
    println!("\n[4/6] Caching base model & tokenizer...");
    paw_candle::ensure_assets(&config, &dir, bundle.interpreter_model()).await?;

    // ── 5. Load model ─────────────────────────────────────────────────
    println!("\n[5/6] Loading model...");
    let candle_config = paw_candle::PawCandleConfig::builder()
        .core(config)
        .build();
    let mut func = paw_candle::PawFnLoader::new(dir)
        .config(candle_config)
        .load()?;

    // ── 6. Run inference ──────────────────────────────────────────────
    println!("\n[6/6] Running inference...");
    let input = "Urgent: your account has been compromised";
    let opts = paw_candle::PawRuntimeOptions {
        max_tokens: Some(20),
        ..Default::default()
    };
    let output = func.run(input, &opts)?;
    println!("  input:  {input}");
    println!("  output: {output}");

    Ok(())
}
