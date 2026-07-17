/// Quick tool to inspect and clear the local PAW cache.
///
/// Usage:
///   cargo run --example clear_cache                    # show size
///   cargo run --example clear_cache -- --clear-all     # wipe everything
///   cargo run --example clear_cache -- --clear-programs
///   cargo run --example clear_cache -- --clear-models
use paw_core::prelude::*;

fn main() -> Result<()> {
    let config = PawConfig::from_env();
    let cache = CacheManager::new(&config);

    let size = cache.total_size().unwrap_or(0);
    let size_mb = size as f64 / 1_048_576.0;
    println!("Cache:   {}", config.cache_dir().display());
    println!("Size:    {size_mb:.1} MB");

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Ok(());
    }

    match args[0].as_str() {
        "--clear-all" => {
            cache.clear()?;
            println!("Cleared all");
        }
        "--clear-programs" => {
            cache.clear_programs()?;
            println!("Cleared programs");
        }
        "--clear-models" => {
            cache.clear_base_models()?;
            println!("Cleared base models");
        }
        other => eprintln!(
            "Unknown flag: {other} (use --clear-all, --clear-programs, or --clear-models)"
        ),
    }

    Ok(())
}
