//! # PAW RS — ProgramAsWeights Rust SDK
//!
//! High-level ergonomic API for loading, compiling, and running PAW programs.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//!
//! # async fn example() -> Result<(), paw_core::Error> {
//! // Load an existing program by slug
//! let mut fn = PawFn::builder()
//!     .slug("email-triage")
//!     .load()
//!     .await?;
//! let result = fn.run("Urgent: server is down!")?;
//! println!("{result}");
//!
//! // Or compile a new program
//! let mut fn = PawFn::builder()
//!     .spec("Classify sentiment: return POSITIVE or NEGATIVE")
//!     .compile()
//!     .await?;
//! let result = fn.run("I love this!")?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! ## Low-level access
//!
//! Both `paw_core` and `paw_candle` are re-exported as sub-modules for full control:
//!
//! ```rust,no_run
//! use paw_rs::paw_core::{PawClient, PawConfig};
//! use paw_rs::paw_candle::{PawFnLoader, PawCandleConfig, PawRuntimeOptions};
//! # async fn example() -> Result<(), paw_candle::Error> {
//! let config = PawConfig::from_env();
//! let client = PawClient::new(&config);
//! let dir = client.download_paw("some-id").await?;
//! let mut func = PawFnLoader::new(dir).config(PawCandleConfig::default()).load()?;
//! let result = func.run("hello", &PawRuntimeOptions::default())?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```

pub use paw_candle;
pub use paw_core;

pub mod function;
pub mod prelude;

pub use function::{PawFn, PawFnBuilder};
