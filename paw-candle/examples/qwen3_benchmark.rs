//! Benchmark: end-to-end inference with Qwen3-0.6B.
//!
//! Measures prefill and decode timing across multiple runs.
//!
//! Usage:
//!   cargo run --release --example qwen3_benchmark
//!   cargo run --release --example qwen3_benchmark -- --runs 50 --max-tokens 30

use std::time::{Duration, Instant};

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
    let n_runs: usize = std::env::args()
        .find_map(|a| a.strip_prefix("--runs=")?.parse().ok())
        .unwrap_or(20);
    let max_tokens: usize = std::env::args()
        .find_map(|a| a.strip_prefix("--max-tokens=")?.parse().ok())
        .unwrap_or(30);

    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // ── 1. Download email-triage bundle ────────────────────────────────
    println!("[1/4] Downloading email-triage bundle...");
    let program_id = client.resolve_slug("email-triage").await?;
    let dir = client.download_paw(&program_id).await?;

    // Remove any old prefix cache
    let prefix_cache_path = dir.join("prefix_kv_cache.bin");
    if prefix_cache_path.exists() {
        std::fs::remove_file(&prefix_cache_path)?;
        println!("  removed old prefix KV cache");
    }

    // ── 2. Download Qwen3 base model ────────────────────────────────────
    println!("[2/4] Ensuring Qwen3 base model...");
    let hf = HFClient::new().map_err(|e| Error::Other(format!("hf-hub init: {e}")))?;
    let gguf_path = config.base_models_dir().join(QWEN3_FILE);
    ensure_cached(&hf, "programasweights", "Qwen3-0.6B-GGUF-Q6_K", QWEN3_FILE, &gguf_path).await?;

    // ── 3. Download tokenizer ──────────────────────────────────────────
    println!("[3/4] Ensuring tokenizer...");
    let tok_path = dir.join(TOKENIZER_FILE);
    ensure_cached(&hf, TOKENIZER_REPO, TOKENIZER_MODEL, TOKENIZER_FILE, &tok_path).await?;

    // ── 4. Load model ──────────────────────────────────────────────────
    println!("[4/4] Loading model...");
    let load_start = Instant::now();
    let candle_config = PawCandleConfig::builder().core(config).build();
    let mut func = PawFnLoader::new(dir)
        .config(candle_config)
        .load()?;
    let load_dur = load_start.elapsed();
    println!("  model loaded in {:.2}s", load_dur.as_secs_f64());

    // ── 5. Benchmark ───────────────────────────────────────────────────
    let input = "Urgent: your account has been compromised";
    let opts = PawRuntimeOptions { max_tokens: Some(max_tokens), ..Default::default() };

    let mut timings: Vec<Duration> = Vec::with_capacity(n_runs);

    for i in 0..n_runs {
        let start = Instant::now();
        let output = func.run(input, &opts)?;
        let elapsed = start.elapsed();
        timings.push(elapsed);

        if i < 3 || i == n_runs - 1 {
            println!("  run {:>2}: {:.1}ms  output: {output:?}", i + 1, elapsed.as_secs_f64() * 1000.0);
        }
    }



    // Statistics
    timings.sort();
    let avg: Duration = timings.iter().sum::<Duration>() / n_runs as u32;
    let first = timings[0];
    let steady: Vec<&Duration> = timings.iter().skip(1).collect();
    let steady_avg: Duration = steady.iter().copied().sum::<Duration>() / steady.len() as u32;
    let steady_min = **steady.iter().min().unwrap();
    let steady_max = **steady.iter().max().unwrap();

    println!();
    println!("═══ Benchmark Results ═══");
    println!("  Input:      {input:?}");
    println!("  Max tokens: {max_tokens}");
    println!("  Runs:       {n_runs}");
    println!("  First call: {:.1}ms", first.as_secs_f64() * 1000.0);
    println!("  Steady avg: {:.1}ms", steady_avg.as_secs_f64() * 1000.0);
    println!("  Steady min: {:.1}ms", steady_min.as_secs_f64() * 1000.0);
    println!("  Steady max: {:.1}ms", steady_max.as_secs_f64() * 1000.0);
    println!("  Overall avg: {:.1}ms", avg.as_secs_f64() * 1000.0);

    Ok(())
}
