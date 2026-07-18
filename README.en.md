# PAW RS — Unofficial ProgramAsWeights Rust SDK

Unofficial Rust SDK for embedding ProgramAsWeights inference in Rust projects.

**⚠️ Note**: This SDK is not officially maintained by ProgramAsWeights.
CPU inference is approximately 4-5x slower than the official Python SDK (llama.cpp backend).
GPU acceleration via CUDA/Metal is supported and can reduce the gap.

> [中文版本](./README.md)

## Quick Start

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .slug("email-triage")
        .load()
        .await?;
    let result = f.run("Urgent: server is down!")?;
    println!("{result}"); // "immediate"
    Ok(())
}
```

## Installation

```toml
[dependencies]
paw-rs = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Use Cases

- Embed PAW neural inference in pure Rust projects
- No Python runtime required
- Suitable for HTTP services, CLI tools, and other Rust-native environments

## Performance

| | Python SDK (llama.cpp) | Rust SDK (candle CPU) |
|---|---|---|
| Qwen3 (30 tokens) | ~200ms | ~900ms |
| GPT-2 (10 tokens) | — | ~140ms |
| Model memory | ~600MB | ~600MB |

GPU acceleration via `--features cuda` or `--features metal` can significantly reduce the gap.

## CLI

```bash
# Run an existing program
paw-rs run --program email-triage --input "Is this urgent?"

# Compile a new program
paw-rs compile --spec "Classify sentiment as positive or negative"

# Query program info
paw-rs info email-triage

# Global options
paw-rs --api-url https://api.programasweights.com --api-key paw_sk_xxx run --program ...
```

## Supported Models

| Model | ID | GGUF Size |
|---|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 594 MB |
| GPT-2 (124M) | `gpt2` | 134 MB |

## Feature Flags

| flag | Description |
|---|---|
| `cuda` | NVIDIA GPU acceleration (`--features cuda`) |
| `metal` | Apple Silicon GPU acceleration |

## Architecture

| crate | Description |
|---|---|
| `paw-core` | HTTP client, cache, bundle parsing |
| `paw-candle` | Candle inference engine, model loading, LoRA |
| `paw-rs` | High-level API (`PawFn` / `PawFnBuilder`) + CLI |

Low-level crates are accessible via `paw_rs::paw_core` and `paw_rs::paw_candle`.

## Links

- [ProgramAsWeights Website](https://programasweights.com)
- [Python SDK Documentation](https://programasweights.readthedocs.io)
- [Candle Framework](https://github.com/huggingface/candle)
