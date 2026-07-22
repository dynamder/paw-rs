use std::sync::Arc;

use paw_candle::{PawCandleConfig, PawFnLoader};
use paw_core::PawConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prog_id = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("0939d42cce70e08ae54c");

    let paw_config = PawConfig::from_env();
    let program_dir = paw_config.programs_dir().join(prog_id);
    assert!(program_dir.exists());

    let config = PawCandleConfig::builder()
        .core(paw_config)
        .max_model_copies(4)
        .build();

    println!("Loading 3 PawFunctions with max_model_copies=4...");
    let f1 = PawFnLoader::new(&program_dir)
        .config(config.clone())
        .load()?;
    let f2 = PawFnLoader::new(&program_dir)
        .config(config.clone())
        .load()?;
    let f3 = PawFnLoader::new(&program_dir).config(config).load()?;

    // Verify all share the same pool
    let p1 = Arc::as_ptr(&f1.pool) as *const ();
    let p2 = Arc::as_ptr(&f2.pool) as *const ();
    let p3 = Arc::as_ptr(&f3.pool) as *const ();
    let s1 = Arc::as_ptr(&f1.model) as *const ();
    let s2 = Arc::as_ptr(&f2.model) as *const ();
    let s3 = Arc::as_ptr(&f3.model) as *const ();

    println!();
    println!("Pool pointers:");
    println!("  f1: {p1:?}");
    println!(
        "  f2: {p2:?} {}",
        if p1 == p2 { "= same pool" } else { "DIFFERENT" }
    );
    println!(
        "  f3: {p3:?} {}",
        if p1 == p3 { "= same pool" } else { "DIFFERENT" }
    );
    println!();
    println!("Model pointers:");
    println!("  f1: {s1:?}");
    println!(
        "  f2: {s2:?} {}",
        if s1 == s2 {
            "= same model"
        } else {
            "DIFFERENT"
        }
    );
    println!(
        "  f3: {s3:?} {}",
        if s1 == s3 {
            "= same model"
        } else {
            "DIFFERENT"
        }
    );

    // Count pool models
    let locks = f1.pool.state.lock().unwrap();
    println!();
    println!(
        "Pool state: {} model(s) loaded (max: {})",
        locks.models.len(),
        locks.max
    );
    drop(locks);

    if p1 == p2 && p1 == p3 {
        println!("\nPASS: All 3 functions share the same pool");
        if s1 == s2 && s1 == s3 {
            println!("PASS: All functions reference the same model copy (only 1 loaded)");
        } else {
            println!("INFO: Different model copies within the same pool (multiple loaded)");
        }
    } else {
        println!("\nFAIL: Pools are different!");
    }

    Ok(())
}
