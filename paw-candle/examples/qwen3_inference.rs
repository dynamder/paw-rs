//! End-to-end: run the email-triage program with Qwen3-0.6B.
//!
//! Downloads the .paw bundle, Qwen3 base model, and tokenizer,
//! then runs inference with the full Qwen3 + LoRA pipeline.
//!
//! Usage:
//!   cargo run --release --example qwen3_inference

use hf_hub::HFClient;
use paw_candle::prelude::*;
use paw_core::prelude::*;

const QWEN3_REPO: &str = "programasweights/Qwen3-0.6B-GGUF-Q6_K";
const QWEN3_FILE: &str = "qwen3-0.6b-q6_k.gguf";
const TOKENIZER_REPO: &str = "Qwen";
const TOKENIZER_MODEL: &str = "Qwen3-0.6B";
const TOKENIZER_FILE: &str = "tokenizer.json";

async fn ensure_cached<T: AsRef<std::path::Path>>(
    hf: &HFClient, repo: &str, model: &str, file: &str, dst: T,
) -> Result<()> {
    let dst = dst.as_ref();
    if dst.exists() { return Ok(()); }
    println!("  downloading {repo}/{model}/{file}...");
    let cached = hf.model(repo, model)
        .download_file().filename(file).send().await
        .map_err(|e| Error::Other(format!("hf-hub: {e}")))?;
    if let Some(p) = dst.parent() { std::fs::create_dir_all(p)?; }
    std::fs::copy(&cached, dst)?;
    println!("  cached to {}", dst.display());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // ── 1. Download email-triage bundle ────────────────────────────────
    println!("[1/4] Downloading email-triage bundle...");
    let program_id = client.resolve_slug("email-triage").await?;
    let dir = client.download_paw(&program_id).await?;
    let bundle = PawBundle::load_from_dir(&dir)?;
    println!("  interpreter: {}", bundle.interpreter_model());

    // ── 2. Download Qwen3 base model ────────────────────────────────────
    println!("[2/4] Ensuring Qwen3 base model...");
    let hf = HFClient::new().map_err(|e| Error::Other(format!("hf-hub init: {e}")))?;
    let gguf_path = config.base_models_dir().join(QWEN3_FILE);
    ensure_cached(&hf, "programasweights", "Qwen3-0.6B-GGUF-Q6_K", QWEN3_FILE, &gguf_path).await?;

    // ── 3. Download tokenizer ──────────────────────────────────────────
    println!("[3/4] Ensuring tokenizer...");
    let tok_path = dir.join(TOKENIZER_FILE);
    ensure_cached(&hf, TOKENIZER_REPO, TOKENIZER_MODEL, TOKENIZER_FILE, &tok_path).await?;

    // ── 4. Load model and run inference ────────────────────────────────
    println!("[4/4] Loading model, running inference...");
    let candle_config = PawCandleConfig::builder().core(config).build();
    let mut func = PawFnLoader::new(dir)
        .config(candle_config)
        .load()?;
    println!("  model loaded, generating...");

    let input = "Urgent: your account has been compromised";
    let opts = PawRuntimeOptions { max_tokens: Some(30), ..Default::default() };
    let output = func.run(input, &opts)?;
    println!("  input:  {input}");
    println!("  output: {output}");

    Ok(())
}
