//! Static typing with base model sharing via `PawFn<T, B>`.
//!
//! Uses `PawFn::<Qwen3_0_6B, Candle>` so the compiler knows the interpreter
//! and backend at compile time.  Multiple `PawFn<T, B>` instances of the
//! same `(T, B)` share a single base model in memory (keyed by `TypeId`).
//!
//! Usage:
//!   PAW_API_KEY=paw_sk_... cargo run --example typed_api

use paw_rs::paw_candle::Qwen3_0_6B;
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> paw_core::Result<()> {
    // ── 1. Load an existing program ────────────────────────────────────
    println!("[1/2] Loading email-triage (Qwen3 model)...");
    let mut a = PawFn::<Qwen3_0_6B, Candle>::load_slug("email-triage").await?;

    // ── 2. Compile a new program on the same model ────────────────────
    println!("[2/2] Compiling sentiment program (shares Qwen3 model)...");
    let mut b = PawFn::<Qwen3_0_6B, Candle>::compile_spec(
        "Classify sentiment: return POSITIVE or NEGATIVE",
        "paw-4b-qwen3-0.6b",
    )
    .await?;

    // Both `a` and `b` share the same Qwen3 base model in memory.
    // The LoRA adapter is swapped per `run()` call — no extra VRAM needed.

    println!("  a: {}", a.run("Is this urgent?")?);
    println!("  b: {}", b.run("I love this product!")?);
    println!("  a: {}", a.run("Server is down!")?);
    println!("  b: {}", b.run("This is terrible")?);

    Ok(())
}
