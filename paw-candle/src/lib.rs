//! # PAW Candle — Candle-based inference runtime for ProgramAsWeights
//!
//! This crate provides the local inference engine for PAW programs using
//! `candle` for tensor computation. It loads quantized GGUF base models,
//! applies LoRA adapters, and runs autoregressive generation.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use paw_candle::{PawCandleConfig, PawFnLoader, PawRuntimeOptions};
//! use paw_core::PawClient;
//!
//! # async fn example() -> Result<(), paw_candle::Error> {
//! let client = PawClient::from_env();
//! let dir = client.download_paw("email-triage").await?;
//!
//! let config = PawCandleConfig::default();
//! let mut func = PawFnLoader::new(dir)
//!     .config(config)
//!     .load()?;
//! let result = func.run("Urgent: server is down!", &PawRuntimeOptions::default())?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! ## Examples
//!
//! Runnable examples in [`paw-candle/examples/`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples):
//!
//! | File | Description |
//! |------|-------------|
//! | [`qwen3_inference.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/qwen3_inference.rs) | Load existing program and run inference |
//! | [`gpt2_inference.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/gpt2_inference.rs) | Compile, download, infer (requires API key) |
//! | [`verify_bundle.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/verify_bundle.rs) | Load LoRA, forward pass verification |
//! | [`qwen3_benchmark.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/qwen3_benchmark.rs) | End-to-end latency benchmark |
//! | [`gpt2_benchmark.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/gpt2_benchmark.rs) | GPT-2 latency benchmark |

pub mod interpreter;
mod kv_cache;
pub mod lora;
pub mod models;
mod tokenizer;

pub mod config;
pub mod runtime;

pub use config::PawCandleConfigBuilder;
pub use config::{DevicePreference, PawCandleConfig};
pub use interpreter::{CandleBackend, get_or_load_model};
pub use paw_core;
pub use paw_core::{Dynamic, Gpt2, InterpreterModel, Qwen3_0_6B};
pub use paw_core::{PawFnTrait, PawRuntimeOptions};
pub use runtime::{PawFnLoader, PawFunction, ensure_assets};

pub use lora::{GgufLoraAdapter, LoraLayer};

pub type Error = paw_core::Error;

pub mod prelude {
    pub use super::config::{DevicePreference, PawCandleConfig, PawCandleConfigBuilder};
    pub use super::lora::{GgufLoraAdapter, LoraLayer};
    pub use super::runtime::{PawFnLoader, PawFunction};
    pub use paw_core::prelude::*;
    pub use paw_core::{Gpt2, InterpreterModel, Qwen3_0_6B};
    pub use paw_core::{PawFnTrait, PawRuntimeOptions};
}
