//! # PAW LlamaCpp — llama.cpp inference runtime for ProgramAsWeights
//!
//! This crate provides the llama.cpp-based inference engine for PAW programs.
//! It loads GGUF base models, applies LoRA adapters, and runs autoregressive
//! generation via llama.cpp.
//!
//! ## Model Sharing
//!
//! All PawFunctions using the same base model (e.g., Qwen3-0.6B) share a global
//! model pool, dramatically reducing memory when running multiple programs.
//!
//! Configure via [`PawLlamaCppConfig::max_model_copies`]:
//!
//! | `max_model_copies` | Behavior |
//! |---|---|
//! | 1 (default) | Single model copy, all PawFunctions share, serial execution |
//! | N (N > 1) | Up to N model copies, lazy-loaded on demand, N-way parallel |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use paw_llamacpp::{PawFnLoader, PawLlamaCppConfig, PawRuntimeOptions};
//! use paw_core::PawConfig;
//!
//! # fn example() -> Result<(), paw_llamacpp::Error> {
//! let config = PawLlamaCppConfig::builder()
//!     .core(PawConfig::from_env())
//!     .max_model_copies(1)  // 1 copy = shared + serial (default)
//!     .build();
//!
//! let func = PawFnLoader::new("/path/to/program_dir")
//!     .config(config)
//!     .load()?;
//!
//! let result = func.run("Hello, world!", &PawRuntimeOptions::default())?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! Multiple programs sharing the same model:
//!
//! ```rust,no_run
//! # use paw_llamacpp::{PawFnLoader, PawLlamaCppConfig, PawRuntimeOptions};
//! # use paw_core::PawConfig;
//! # fn example() -> Result<(), paw_llamacpp::Error> {
//! let config = PawLlamaCppConfig::builder()
//!     .core(PawConfig::from_env())
//!     .max_model_copies(1)  // share 1 model, serial execution
//!     .build();
//!
//! let func_a = PawFnLoader::new("program_a").config(config.clone()).load()?;
//! let func_b = PawFnLoader::new("program_b").config(config).load()?;
//! // func_a and func_b share the same Qwen3 model (~588 MB each,
//! // now only ~588 MB total instead of ~1.2 GB)
//! # Ok(())
//! # }
//! ```
//!
//! ## Examples
//!
//! | Example | Description |
//! |---------|-------------|
//! | [`llamacpp_benchmark`](https://github.com/dynamder/paw-rs/tree/main/paw-llamacpp/examples/llamacpp_benchmark.rs) | Latency benchmark |
//! | [`parallel_benchmark`](https://github.com/dynamder/paw-rs/tree/main/paw-llamacpp/examples/parallel_benchmark.rs) | Parallel throughput test |
//! | [`verify_backend`](https://github.com/dynamder/paw-rs/tree/main/paw-llamacpp/examples/verify_backend.rs) | Correctness verification |
//! | [`compare_test`](https://github.com/dynamder/paw-rs/tree/main/paw-llamacpp/examples/compare_test.rs) | Cross-backend comparison |

pub mod backend;
pub mod config;
pub mod pool;
pub mod runtime;

pub use backend::LlamaCppBackend;
pub use config::{DevicePreference, PawLlamaCppConfig, PawLlamaCppConfigBuilder};
pub use paw_core;
pub use paw_core::{PawFnTrait, PawRuntimeOptions};
pub use runtime::{PawFnLoader, PawFunction};

pub type Error = paw_core::Error;

pub mod prelude {
    pub use super::config::{DevicePreference, PawLlamaCppConfig, PawLlamaCppConfigBuilder};
    pub use super::runtime::{PawFnLoader, PawFunction};
    pub use paw_core::prelude::*;
}
