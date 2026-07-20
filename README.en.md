# PAW RS — ProgramAsWeights Rust SDK

Unofficial Rust SDK for embedding ProgramAsWeights inference in Rust projects.

> [中文版本](./README.md)

---

## Installation

```toml
[dependencies]
paw-rs = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

The default backend is **llama.cpp** (CPU, ~240ms for Qwen3-0.6B, 30 tokens). 
For GPU acceleration or the typed `PawFn<T>` API, enable the `candle` backend:

```toml
paw-rs = { version = "0.1", default-features = false, features = ["candle", "cuda"] }
```

---

## SDK Quick Start

### Dynamic dispatch (builder — any backend)

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFnBuilder::builder()
        .slug("email-triage")
        .load()
        .await?;
    let result = f.run("Urgent: server is down!")?;
    println!("{result}"); // "immediate"
    Ok(())
}
```

### Compile and run

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFnBuilder::builder()
        .spec("Classify sentiment: return POSITIVE or NEGATIVE")
        .compile()
        .await?;
    let result = f.run("I love this product!")?;
    println!("{result}"); // "POSITIVE"
    Ok(())
}
```

### Static typing with model sharing (candle backend)

Requires `--features candle`. Two `PawFn<Qwen3_0_6B, Candle>` instances share one base model in memory:

```rust
use paw_rs::prelude::*;
use paw_rs::paw_core::{Candle, Qwen3_0_6B};

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut a = PawFn::<Qwen3_0_6B, Candle>::load_slug("email-triage").await?;
    let mut b = PawFn::<Qwen3_0_6B, Candle>::compile_spec(
        "Classify sentiment", "paw-4b-qwen3-0.6b",
    ).await?;
    println!("{}", a.run("Server is down!")?);
    println!("{}", b.run("I love this product!")?);
    Ok(())
}
```

### Custom runtime options

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFnBuilder::builder()
        .slug("email-triage")
        .load()
        .await?;

    let opts = paw_core::PawRuntimeOptions {
        max_tokens: Some(100),
        temperature: 0.7,
        ..Default::default()
    };
    let result = f.run_with("What should I do?", &opts)?;
    println!("{result}");
    Ok(())
}
```

---

## CLI

Same as before. See `paw-rs --help`.

---

## Feature Flags

| flag | Description |
|------|-------------|
| `candle` | Candle backend (requires `default-features = false`) |
| `llamacpp` | llama.cpp backend (default) |
| `cuda` | NVIDIA GPU (forwarded to active backend) |
| `metal` | Apple Silicon GPU |
| `mkl` | Intel MKL CPU acceleration (candle only) |

```bash
# llama.cpp CPU (default)
cargo run -- run --program email-triage --input "test"

# candle + CUDA GPU
cargo run --no-default-features --features candle,cuda -- run --program email-triage --input "test"

# candle + MKL (CPU)
cargo run --no-default-features --features candle,mkl -- run --program email-triage --input "test"
```

---

## Performance

| Backend | Qwen3 (10 tokens) | Model memory | GPU support |
|---------|------------------|--------------|-------------|
| llama.cpp (CPU) | ~240ms | 588 MB | CUDA / Metal / Vulkan |
| candle (CPU, MKL) | ~2000ms | 588 MB | CUDA / Metal |
| candle (CPU, native) | ~680ms | 588 MB | CUDA / Metal |
| candle (CUDA) | ~200ms | 588 MB + VRAM | CUDA |

---

## Architecture

| crate | Description |
|-------|-------------|
| `paw-core` | `InterpreterModel` / `Backend` traits, `PawFnTrait`, `PawRuntimeOptions`, HTTP client, cache |
| `paw-candle` | `CandleBackend`, `Qwen3Model`, `Gpt2Model`, `PawFnLoader` |
| `paw-llamacpp` | `LlamaCppBackend` (experimental), ~2.8x CPU speedup over candle |
| `paw-rs` | `PawFn<T, B>`, `PawFnBuilder`, CLI |

---

## Examples

| Example | Crate | Description | API Key? |
|---------|-------|-------------|----------|
| `high_level` | `paw-rs` | Builder: compile → infer | Yes |
| `low_level` | `paw-rs` | Builder: load → infer | Yes |
| `typed_api` | `paw-rs` | Static typing with model sharing | Yes |
| `qwen3_inference` | `paw-candle` | Load existing program | No |
| `llamacpp_benchmark` | `paw-llamacpp` | llama.cpp latency test | No |
| `verify_bundle` | `paw-candle` | LoRA forward verification | No |
| `download_and_save` | `paw-core` | Bundle format roundtrip | No |

```bash
# Builder (default llamacpp backend)
PAW_API_KEY=sk_... cargo run --example high_level -p paw-rs

# Static typed (candle + model sharing)
PAW_API_KEY=sk_... cargo run --example typed_api -p paw-rs --features candle

# llama.cpp benchmark (no API key needed)
cargo run --release --example llamacpp_benchmark -p paw-llamacpp
```
