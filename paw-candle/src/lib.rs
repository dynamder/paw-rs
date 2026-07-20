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

mod kv_cache;
pub mod lora;
pub mod models;
mod tokenizer;

pub mod config;
pub mod runtime;

pub use config::PawCandleConfigBuilder;
pub use config::{DevicePreference, PawCandleConfig};
pub use paw_core;
pub use runtime::{ensure_assets, PawFnLoader, PawFunction, PawRuntimeOptions};

pub use lora::{GgufLoraAdapter, LoraLayer};

pub type Error = paw_core::Error;

pub mod prelude {
    pub use super::config::{DevicePreference, PawCandleConfig, PawCandleConfigBuilder};
    pub use super::lora::{GgufLoraAdapter, LoraLayer};
    pub use super::runtime::{PawFnLoader, PawFunction, PawRuntimeOptions};
    pub use paw_core::prelude::*;
}
