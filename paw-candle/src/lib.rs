//! # PAW Candle — Candle-based inference runtime for ProgramAsWeights
//!
//! This crate provides the local inference engine for PAW programs using
//! `candle` for tensor computation. It loads quantized GGUF base models,
//! applies LoRA adapters, and runs autoregressive generation.
//!
//! ## Model Sharing
//!
//! All PawFunctions using the same base model share a global model pool,
//! reducing memory when running multiple programs. Configure via
//! [`PawCandleConfig::max_model_copies`]:
//!
//! | `max_model_copies` | Behavior |
//! |---|---|
//! | 1 (default) | Single model copy, serial execution, minimal memory |
//! | N (N > 1) | Up to N copies, lazy-loaded on demand, N-way parallel |
//!
//! Model copies are **lazy-loaded**: starting with 1, more are created only
//! when concurrent inference calls would otherwise block.
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
//! With model copies:
//!
//! ```rust,no_run
//! use paw_candle::{PawCandleConfig, PawFnLoader, PawRuntimeOptions};
//! use paw_core::PawConfig;
//!
//! # fn example() -> Result<(), paw_candle::Error> {
//! let config = PawCandleConfig::builder()
//!     .core(PawConfig::from_env())
//!     .max_model_copies(4)  // up to 4 parallel, lazy-loaded
//!     .build();
//!
//! let func_a = PawFnLoader::new("program_a").config(config.clone()).load()?;
//! let func_b = PawFnLoader::new("program_b").config(config).load()?;
//! // Both share the same pool; extra copies loaded only if both run concurrently
//! # Ok(())
//! # }
//! ```
//!
//! ## Examples
//!
//! | Example | Description |
//! |---------|-------------|
//! | [`qwen3_inference`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/qwen3_inference.rs) | Load existing program and run inference |
//! | [`gpt2_inference`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/gpt2_inference.rs) | Compile, download, infer (requires API key) |
//! | [`verify_bundle`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/verify_bundle.rs) | Load LoRA, forward pass verification |
//! | [`qwen3_benchmark`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/qwen3_benchmark.rs) | End-to-end latency benchmark |
//! | [`gpt2_benchmark`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/gpt2_benchmark.rs) | GPT-2 latency benchmark |
//! | [`compare_ref`](https://github.com/dynamder/paw-rs/tree/main/paw-candle/examples/compare_ref.rs) | Reference output for cross-backend comparison |

pub mod interpreter;
mod kv_cache;
pub mod lora;
pub mod models;
mod pool;
mod tokenizer;

pub mod config;
pub mod runtime;

pub use config::PawCandleConfigBuilder;
pub use config::{DevicePreference, PawCandleConfig};
pub use interpreter::CandleBackend;
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
