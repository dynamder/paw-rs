//! # PAW RS — ProgramAsWeights Rust SDK
//!
//! High-level ergonomic API for loading, compiling, and running PAW programs.
//!
//! ## Quick Start
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
//! ## CLI
//!
//! The crate also ships the `paw-rs` binary — run `paw-rs --help` for usage.

#[cfg(feature = "candle")]
pub use paw_candle;
#[cfg(feature = "llamacpp")]
pub use paw_llamacpp;
pub use paw_core;

pub mod function;
pub mod prelude;

pub use function::PawFnBuilder;
#[cfg(feature = "candle")]
pub use function::PawFn;

#[cfg(test)]
mod tests {
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
    fn test_builder_config_passed_through() {
        let config = paw_core::PawConfig::builder()
            .api_url("https://test.example.com")
            .build()
            .unwrap();
        let b = PawFnBuilder::builder()
            .config(config)
            .slug("test");
        let _: PawFnBuilder<function::ForLoad> = b;
    }

    #[test]
    fn test_builder_ephemeral_propagates() {
        let _b = PawFnBuilder::builder()
            .ephemeral(true)
            .spec("test");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_paw_fn_from_inner_type() {
        fn _take_from_inner(f: fn(paw_candle::PawFunction) -> PawFn<paw_candle::Qwen3_0_6B>) {
            let _ = f;
        }
        _take_from_inner(PawFn::from_inner);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_paw_fn_send() {
        fn _assert_send<T: Send>() {}
        _assert_send::<PawFn<paw_candle::Qwen3_0_6B>>();
    }

    #[test]
    fn test_runtime_options_default() {
        let opts = paw_core::PawRuntimeOptions::default();
        assert_eq!(opts.max_tokens, None);
        assert_eq!(opts.temperature, 0.0);
        assert_eq!(opts.top_p, 1.0);
    }

    #[test]
    fn test_runtime_options_custom() {
        let opts = paw_core::PawRuntimeOptions {
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
    }
}
