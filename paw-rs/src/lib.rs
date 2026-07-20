//! # PAW RS — ProgramAsWeights Rust SDK
//!
//! High-level ergonomic API for loading, compiling, and running PAW programs.
//!
//! ## Quick Start
//!
//! ### Run an existing program (dynamic dispatch)
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//!
//! # async fn example() -> std::result::Result<(), paw_core::Error> {
//! let mut f = PawFnBuilder::builder()
//!     .slug("email-triage")
//!     .load()
//!     .await?;
//! let result = f.run("Urgent: server is down!")?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```
//!
//! ### Static typing with model sharing
//!
//! ```rust,no_run
//! use paw_rs::prelude::*;
//! use paw_rs::paw_candle::Qwen3_0_6B;
//!
//! # async fn example() -> std::result::Result<(), paw_core::Error> {
//! let mut a = PawFn::<Qwen3_0_6B>::load_slug("email-triage").await?;
//! let mut b = PawFn::<Qwen3_0_6B>::compile_spec(
//!     "Classify sentiment", "paw-4b-qwen3-0.6b",
//! ).await?;
//! let r1 = a.run("Urgent: server is down!")?;
//! let r2 = b.run("I love this product!")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Examples
//!
//! See `paw-rs/examples/` for runnable examples:
//!
//! - `high_level.rs` — Type-state builder: compile → infer in one call
//! - `low_level.rs` — Manual 6-step pipeline with PawClient + PawFnLoader
//!
//! ```bash
//! PAW_API_KEY=paw_sk_... cargo run --example high_level -p paw-rs
//! PAW_API_KEY=paw_sk_... cargo run --example low_level -p paw-rs
//! ```
//!
//! ## CLI
//!
//! The crate also ships the `paw-rs` binary — see `README.md` or run `paw-rs --help` for usage.

pub use paw_candle;
pub use paw_core;

pub mod function;
pub mod prelude;

pub use function::{PawFn, PawFnBuilder};

#[cfg(test)]
mod tests {
    use paw_candle::{DevicePreference, PawRuntimeOptions, Qwen3_0_6B};

    use super::*;

    #[test]
    fn test_builder_returns_unset() {
        let b = PawFnBuilder::builder();
        let _: PawFnBuilder<function::Unset> = b;
    }

    #[test]
    fn test_builder_unset_to_forload() {
        let b = PawFnBuilder::builder().slug("test-program");
        let _: PawFnBuilder<function::ForLoad> = b;
    }

    #[test]
    fn test_builder_unset_to_forcompile() {
        let b = PawFnBuilder::builder().spec("Classify sentiment");
        let _: PawFnBuilder<function::ForCompile> = b;
    }

    #[test]
    fn test_builder_forload_forwards_compiler() {
        let _b = PawFnBuilder::builder()
            .compiler("test-compiler")
            .slug("test");
    }

    #[test]
    fn test_builder_forcompile_has_compiler() {
        let _b = PawFnBuilder::builder()
            .spec("test")
            .compiler("test-compiler");
        let _b = PawFnBuilder::builder()
            .compiler("test-compiler")
            .spec("test");
    }

    #[test]
    fn test_builder_config_and_device_passed_through() {
        let config = paw_core::PawConfig::builder()
            .api_url("https://test.example.com")
            .build()
            .unwrap();
        let b = PawFnBuilder::builder()
            .config(config)
            .device(DevicePreference::Cpu)
            .slug("test");
        let _: PawFnBuilder<function::ForLoad> = b;
    }

    #[test]
    fn test_builder_ephemeral_propagates() {
        let _b = PawFnBuilder::builder()
            .ephemeral(true)
            .spec("test");
    }

    #[test]
    fn test_paw_fn_from_inner_type() {
        fn _take_from_inner(f: fn(paw_candle::PawFunction) -> PawFn<Qwen3_0_6B>) { let _ = f; }
        _take_from_inner(PawFn::from_inner);
    }

    #[test]
    fn test_paw_fn_send() {
        fn _assert_send<T: Send>() {}
        _assert_send::<PawFn<Qwen3_0_6B>>();
    }

    #[test]
    fn test_paw_fn_send_dynamic() {
        fn _assert_send<T: Send>() {}
        _assert_send::<PawFn<paw_candle::Dynamic>>();
    }

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

    #[test]
    fn test_prelude_exports() {
        use crate::prelude::*;
        let _: PawFnBuilder<function::Unset> = PawFnBuilder::builder();
        let _: PawFnBuilder<function::ForLoad> = PawFnBuilder::builder().slug("x");
        let _: PawFnBuilder<function::ForCompile> = PawFnBuilder::builder().spec("x");
    }

    #[test]
    fn test_paw_core_reexported() {
        let _config = paw_core::PawConfig::from_env();
        let _ = paw_candle::PawCandleConfig::default();
    }
}
