# PAW RS — ProgramAsWeights Rust SDK

Unofficial Rust SDK for embedding ProgramAsWeights inference in Rust projects.

> [中文版本](./README.md)

**⚠️ Note**: This SDK is not officially maintained by ProgramAsWeights.
CPU inference is approximately 4–5x slower than the official Python SDK (llama.cpp backend).
GPU acceleration via `--features cuda` can reduce the gap significantly.

---

## Installation

```toml
[dependencies]
paw-rs = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

---

## SDK Quick Start

### Run an existing program

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .slug("email-triage")
        .load()
        .await?;
    let result = f.run("Urgent: server is down!")?;
    println!("{result}"); // "immediate"
    Ok(())
}
```

### Compile and run a new program

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .spec("Classify sentiment: return POSITIVE or NEGATIVE")
        .compile()
        .await?;
    let result = f.run("I love this product!")?;
    println!("{result}"); // "POSITIVE"
    Ok(())
}
```

### Custom runtime options

```rust
use paw_rs::prelude::*;
use paw_rs::paw_candle::PawRuntimeOptions;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .slug("email-triage")
        .load()
        .await?;

    let opts = PawRuntimeOptions {
        max_tokens: Some(100),
        temperature: 0.7,
        ..Default::default()
    };
    let result = f.run_with("What should I do about this?", &opts)?;
    println!("{result}");
    Ok(())
}
```

### Custom configuration

```rust
use paw_rs::prelude::*;
use paw_rs::paw_core::PawConfig;
use paw_rs::paw_candle::DevicePreference;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let config = PawConfig::builder()
        .api_url("https://custom.example.com")
        .api_key("paw_sk_xxx")
        .n_ctx(4096)
        .verbose(true)
        .build()?;

    let mut f = PawFn::builder()
        .config(config)
        .device(DevicePreference::Cpu)
        .slug("email-triage")
        .load()
        .await?;

    let result = f.run("Is this urgent?")?;
    println!("{result}");
    Ok(())
}
```

### Low-level API: manual loading

```rust
use paw_rs::paw_core::{PawClient, PawConfig};
use paw_rs::paw_candle::{PawFnLoader, PawCandleConfig, PawRuntimeOptions};

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // Download the .paw program bundle
    let dir = client.download_paw("some-program-id").await?;

    // Load the model locally
    let mut func = PawFnLoader::new(dir)
        .config(PawCandleConfig::default())
        .load()?;

    let result = func.run("hello", &PawRuntimeOptions::default())?;
    println!("{result}");
    Ok(())
}
```

---

## CLI

The `paw-rs` binary provides a Python-SDK-compatible CLI:

### Authentication

```bash
# Interactive login (opens browser + paste key)
paw-rs login

# Provide API key directly
paw-rs login paw_sk_your_api_key

# Global --api-key works for all commands
paw-rs --api-key paw_sk_xxx run --program email-triage --input "test"
```

### Compile

```bash
# Minimal compile
paw-rs compile --spec "Classify message urgency as low, medium, or high"

# With compiler, slug, and private flag
paw-rs compile \
  --spec "Extract key points from text" \
  --compiler paw-4b-qwen3-0.6b \
  --slug my-extractor \
  --private

# JSON output (for scripts / agent integration)
paw-rs --json compile --spec "Classify sentiment"
```

### Run

```bash
# Run by slug
paw-rs run --program email-triage --input "The server is on fire!"

# Run by program ID
paw-rs run --program a1b2c3d4e5f6a1b2 --input "hello"

# With custom parameters
paw-rs run \
  --program email-triage \
  --input "What's the deadline?" \
  --max-tokens 256 \
  --temperature 0.5 \
  --verbose

# JSON output
paw-rs --json run --program email-triage --input "test"
# → {"program":"email-triage","input":"test","output":"immediate"}
```

### Rename

```bash
# Set or change a slug (positional args)
paw-rs rename a1b2c3d4e5f6a1b2 my-slug

# Remove slug (pass empty string)
paw-rs rename a1b2c3d4e5f6a1b2 ""

# JSON output
paw-rs rename a1b2c3d4e5f6a1b2 my-slug --json
```

### Info

```bash
# Query program metadata (positional arg)
paw-rs info email-triage

# By program ID
paw-rs info a1b2c3d4e5f6a1b2

# JSON output
paw-rs info email-triage --json
```

### Global flags

```bash
# Custom server URL
paw-rs --api-url https://api.custom.com compile --spec "..."

# Global API key (for authenticated endpoints)
paw-rs --api-key paw_sk_xxx info my-program

# JSON mode (works with all subcommands)
paw-rs --json compile --spec "Classify urgency"
paw-rs --json run --program email-triage --input "test"
paw-rs --json info my-program
```

### Agent / scripting workflow

```bash
# Compile → get ID → run
PROGRAM_ID=$(paw-rs --json compile --spec "Classify urgency" | jq -r '.program_id')
paw-rs run --program "$PROGRAM_ID" --input "Please review by EOD" --json | jq -r '.output'
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PAW_API_URL` | PAW server URL | `https://programasweights.com` |
| `PAW_API_KEY` | API key | (none) |
| `PAW_CACHE_DIR` | Cache directory | `~/.cache/programasweights/` |
| `PAW_CONFIG_DIR` | Config directory | `~/.config/programasweights/` |
| `PAW_N_CTX` | Context window size | `2048` |
| `PAW_GPU_LAYERS` | GPU layers (`-1`=all, `0`=CPU) | `-1` |
| `PAW_VERBOSE` | Verbose logging (`1`/`true`) | `false` |
| `PAW_OFFLINE` | Offline mode | `false` |

---

## Supported Models

| Model | ID | GGUF Size |
|---|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 594 MB |
| GPT-2 (124M) | `gpt2` | 134 MB |

---

## Feature Flags

| flag | Description |
|---|---|
| `cuda` | NVIDIA GPU acceleration (`--features cuda`) |
| `metal` | Apple Silicon GPU acceleration |

```bash
cargo run --features cuda -- run --program email-triage --input "test"
```

---

## Performance

| | Python SDK (llama.cpp) | Rust SDK (candle CPU) |
|---|---|---|
| Qwen3 (30 tokens) | ~200ms | ~900ms |
| GPT-2 (10 tokens) | — | ~140ms |
| Model memory | ~600MB | ~600MB |

---

## Examples

Example files are located in each crate's `examples/` directory:

| Example | Crate | Description | API Key? |
|---------|-------|-------------|----------|
| [`high_level`](paw-rs/examples/high_level.rs) | `paw-rs` | High-level API: compile → infer (one-shot) | Yes |
| [`low_level`](paw-rs/examples/low_level.rs) | `paw-rs` | Low-level API: 6-step manual pipeline | Yes |
| [`qwen3_inference`](paw-candle/examples/qwen3_inference.rs) | `paw-candle` | Load existing program and infer | No |
| [`gpt2_inference`](paw-candle/examples/gpt2_inference.rs) | `paw-candle` | Compile → download → infer (GPT-2) | Yes |
| [`download_and_save`](paw-core/examples/download_and_save.rs) | `paw-core` | Download bundle + binary format roundtrip | No |
| [`verify_bundle`](paw-candle/examples/verify_bundle.rs) | `paw-candle` | Load LoRA → forward pass verification | No |

```bash
# High-level API example (requires API key)
PAW_API_KEY=paw_sk_... cargo run --example high_level -p paw-rs

# Low-level API example
PAW_API_KEY=paw_sk_... cargo run --example low_level -p paw-rs

# Load existing program (no API key needed)
cargo run --release --example qwen3_inference -p paw-candle
```

---

## Architecture

| crate | Description |
|---|---|
| `paw-core` | HTTP client, cache management, bundle parsing, type definitions |
| `paw-candle` | Candle inference engine, quantized model loading, LoRA adapters |
| `paw-rs` | High-level API (`PawFn` / `PawFnBuilder`) + CLI binary + examples |

Low-level crates are accessible via `paw_rs::paw_core` and `paw_rs::paw_candle`. Full examples in `paw-rs/examples/`, `paw-candle/examples/`, `paw-core/examples/`.

---

## Links

- [ProgramAsWeights Website](https://programasweights.com)
- [Python SDK Documentation](https://programasweights.readthedocs.io)
- [Candle Framework](https://github.com/huggingface/candle)
