//! End-to-end: compile a GPT-2 program, download, load, and run inference.
//!
//! Requires PAW_API_KEY (needed for compilation).
//! Usage:
//!   PAW_API_KEY=sk-... cargo run --example gpt2_inference

use paw_candle::{PawCandleConfig, PawFnLoader, PawRuntimeOptions};
use paw_core::prelude::*;
use paw_core::CompileRequest;

#[tokio::main]
async fn main() -> Result<()> {
    let config = PawConfig::from_env();
    config
        .effective_api_key()
        .ok_or_else(|| Error::Other("PAW_API_KEY required for compilation".into()))?;
    let client = PawClient::new(&config);

    // ── 1. Find the GPT-2 compiler ──────────────────────────────────────
    println!("[1/5] Finding GPT-2 compiler...");
    let compilers = client.list_compilers().await?;
    let gpt2_compiler = compilers
        .iter()
        .find(|c| c.name.to_lowercase().contains("gpt2"))
        .or_else(|| compilers.iter().find(|c| c.name == "gpt2-q8_0"))
        .map(|c| c.name.clone())
        .unwrap_or_else(|| {
            eprintln!("Available compilers:");
            for c in &compilers {
                eprintln!("  {}: {}", c.name, c.display_name);
            }
            panic!("No GPT-2 compiler found");
        });
    println!("  using compiler: {gpt2_compiler}");

    // ── 2. Compile ──────────────────────────────────────────────────────
    println!("\n[2/5] Compiling new program...");
    let program = client
        .compile(
            CompileRequest::builder()
                .spec("Classify sentiment: return POSITIVE or NEGATIVE")
                .compiler(&gpt2_compiler)
                .ephemeral(true)
                .build()?,
        )
        .await?;
    println!("  program_id: {}, status: {}", program.id, program.status);

    // ── 3. Download .paw bundle ────────────────────────────────────────
    println!("\n[3/5] Downloading .paw bundle...");
    let dir = client.download_paw(&program.id).await?;
    let bundle = PawBundle::load_from_dir(&dir)?;
    println!("  interpreter: {}", bundle.interpreter_model());
    println!("  adapter: {} KB",
        std::fs::metadata(&bundle.adapter_path).map(|m| m.len() / 1024).unwrap_or(0));

    // ── 4. Ensure base model + tokenizer are cached ────────────────────
    println!("\n[4/5] Ensuring base model + tokenizer are cached...");
    paw_candle::ensure_assets(&config, &dir, bundle.interpreter_model()).await?;

    // ── 5. Load and run inference ──────────────────────────────────────
    println!("\n[5/5] Loading model, running inference...");
    let candle_config = PawCandleConfig::builder()
        .core(config)
        .build();
    let mut func = PawFnLoader::new(dir)
        .config(candle_config)
        .load()?;
    println!("  model loaded, starting generation...");

    let input = "Urgent: your account has been compromised";
    let opts = PawRuntimeOptions { max_tokens: Some(30), ..Default::default() };
    let output = func.run(input, &opts)?;
    println!("  input:  {input}");
    println!("  output: {output}");

    if output.trim().is_empty() {
        eprintln!("  (empty output — generation produced no text beyond input)");
    }

    Ok(())
}
