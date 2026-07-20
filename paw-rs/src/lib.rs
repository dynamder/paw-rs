//! # PAW RS — ProgramAsWeights Rust SDK
//!
//! High-level ergonomic API for loading, compiling, and running PAW programs.
//!
//! ## Quick Start
//!
//! ### Run an existing program
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//!
//! # async fn example() -> std::result::Result<(), paw_core::Error> {
//! let mut f = PawFn::builder()
//!     .slug("email-triage")
//!     .load()
//!     .await?;
//! let result = f.run("Urgent: server is down!")?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! ### Compile a new program
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//!
//! # async fn example() -> std::result::Result<(), paw_core::Error> {
//! let mut f = PawFn::builder()
//!     .spec("Classify sentiment: return POSITIVE or NEGATIVE")
//!     .compile()
//!     .await?;
//! let result = f.run("I love this!")?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom runtime options
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//! use paw_rs::paw_candle::PawRuntimeOptions;
//!
//! # async fn example() -> std::result::Result<(), paw_core::Error> {
//! let mut f = PawFn::builder().slug("email-triage").load().await?;
//! let result = f.run_with("What should I do?", &PawRuntimeOptions {
//!     max_tokens: Some(100),
//!     temperature: 0.7,
//!     ..Default::default()
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Configuration
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//! use paw_rs::paw_core::PawConfig;
//! use paw_rs::paw_candle::DevicePreference;
//!
//! # async fn example() -> std::result::Result<(), paw_core::Error> {
//! let config = PawConfig::builder()
//!     .api_url("https://custom.example.com")
//!     .api_key("paw_sk_xxx")
//!     .n_ctx(4096)
//!     .verbose(true)
//!     .build()?;
//!
//! let mut f = PawFn::builder()
//!     .config(config)
//!     .device(DevicePreference::Cpu)
//!     .slug("email-triage")
//!     .load()
//!     .await?;
//!
//! let result = f.run("Is this urgent?")?;
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
//!
//! ## CLI
//!
//! The crate also ships the `paw-rs` binary — see [`README.md`] or run `paw-rs --help` for usage.

pub use paw_candle;
pub use paw_core;

pub mod function;
pub mod prelude;

pub use function::{PawFn, PawFnBuilder};

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use paw_candle::{DevicePreference, PawRuntimeOptions};

    use super::*;

    // ── PawFn builder type-state machine ─────────────────────────────

    /// Helper: assert that a value has a given type at compile time.
    fn _same_type<T>(_: &T, _: &T) {}

    #[test]
    fn test_builder_returns_unset() {
        let b = PawFn::builder();
        let _: PawFnBuilder<function::Unset> = b;
    }

    #[test]
    fn test_builder_unset_to_forload() {
        let b = PawFn::builder().slug("test-program");
        let _: PawFnBuilder<function::ForLoad> = b;
    }

    #[test]
    fn test_builder_unset_to_forcompile() {
        let b = PawFn::builder().spec("Classify sentiment");
        let _: PawFnBuilder<function::ForCompile> = b;
    }

    #[test]
    fn test_builder_forload_forwards_compiler() {
        // Compiler set before .slug() should be carried into ForLoad
        let _b = PawFn::builder()
            .compiler("test-compiler")
            .slug("test");
    }

    #[test]
    fn test_builder_forcompile_has_compiler() {
        // Compiler can be set before or after .spec()
        let _b = PawFn::builder()
            .spec("test")
            .compiler("test-compiler");
        let _b = PawFn::builder()
            .compiler("test-compiler")
            .spec("test");
    }

    #[test]
    fn test_builder_config_and_device_passed_through() {
        let config = paw_core::PawConfig::builder()
            .api_url("https://test.example.com")
            .build()
            .unwrap();
        // Config should propagate through state transitions
        let b = PawFn::builder()
            .config(config)
            .device(DevicePreference::Cpu)
            .slug("test");
        let _: PawFnBuilder<function::ForLoad> = b;
    }

    #[test]
    fn test_builder_ephemeral_propagates() {
        let _b = PawFn::builder()
            .ephemeral(true)
            .spec("test");
    }

    // ── PawFn type ───────────────────────────────────────────────────

    #[test]
    fn test_paw_fn_from_inner_type() {
        // Verify the function signature compiles
        fn _take_from_inner(f: fn(paw_candle::PawFunction) -> PawFn) {
            let _ = f;
        }
        _take_from_inner(PawFn::from_inner);
    }

    #[test]
    fn test_paw_fn_send() {
        // PawFn should be Send if PawFunction is Send
        fn _assert_send<T: Send>() {}
        _assert_send::<PawFn>();
    }

    // ── PawRuntimeOptions ────────────────────────────────────────────

    #[test]
    fn test_runtime_options_default() {
        let opts = PawRuntimeOptions::default();
        assert_eq!(opts.max_tokens, None);
        assert_eq!(opts.temperature, 0.0);
        assert_eq!(opts.top_p, 1.0);
    }

    #[test]
    fn test_runtime_options_custom() {
        let opts = PawRuntimeOptions {
            max_tokens: Some(100),
            temperature: 0.7,
            top_p: 0.9,
        };
        assert_eq!(opts.max_tokens, Some(100));
        assert_eq!(opts.temperature, 0.7);
        assert_eq!(opts.top_p, 0.9);
    }

    // ── Prelude ──────────────────────────────────────────────────────

    #[test]
    fn test_prelude_exports() {
        use crate::prelude::*;
        let _: PawFnBuilder<function::Unset> = PawFn::builder();
        let _: PawFnBuilder<function::ForLoad> = PawFn::builder().slug("x");
        let _: PawFnBuilder<function::ForCompile> = PawFn::builder().spec("x");
    }

    // ── Re-exports ───────────────────────────────────────────────────

    #[test]
    fn test_paw_core_reexported() {
        let _config = paw_core::PawConfig::from_env();
        let _ = paw_candle::PawCandleConfig::default();
    }
}
