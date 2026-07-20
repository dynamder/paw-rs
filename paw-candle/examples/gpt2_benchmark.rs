use std::time::{Duration, Instant};
use paw_candle::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let n_runs: usize = std::env::args().find_map(|a| a.strip_prefix("--runs=")?.parse().ok()).unwrap_or(5);
    let max_tokens: usize = std::env::args().find_map(|a| a.strip_prefix("--max-tokens=")?.parse().ok()).unwrap_or(10);

    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    println!("[1/3] Loading GPT-2 program...");
    let program_id = "ef960d9e5c2c6bc3365a";
    let dir = client.download_paw(&program_id).await?;
    let _ = std::fs::remove_file(dir.join("prefix_kv_cache.bin"));

    println!("[2/3] Ensuring GPT-2 base model...");
    let bundle = PawBundle::load_from_dir(&dir)?;
    paw_candle::ensure_assets(&config, &dir, bundle.interpreter_model()).await?;

    println!("[3/3] Running inference...");
    let candle_config = PawCandleConfig::builder().core(config).build();
    let mut func = PawFnLoader::new(dir).config(candle_config).load()?;

    let input = "I love this product!";
    let opts = PawRuntimeOptions { max_tokens: Some(max_tokens), ..Default::default() };

    let mut timings = Vec::new();
    for i in 0..n_runs {
        let start = Instant::now();
        let output = func.run(input, &opts)?;
        let elapsed = start.elapsed();
        timings.push(elapsed);
        if i < 3 || i == n_runs - 1 {
            eprintln!("  run {:>2}: {:.1}ms  output: {output:?}", i + 1, elapsed.as_secs_f64() * 1000.0);
        }
    }

    let first = timings[0];
    let steady_avg = timings.iter().skip(1).sum::<Duration>() / (n_runs - 1) as u32;
    eprintln!("\n═══ GPT-2 Benchmark ═══");
    eprintln!("  First call: {:.1}ms", first.as_secs_f64() * 1000.0);
    eprintln!("  Steady avg: {:.1}ms", steady_avg.as_secs_f64() * 1000.0);
    Ok(())
}
