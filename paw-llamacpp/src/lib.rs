mod tokenizer;

pub mod config;
pub mod runtime;

pub use config::{DevicePreference, PawLlamaCppConfig, PawLlamaCppConfigBuilder};
pub use paw_core;
pub use runtime::{PawFnLoader, PawFunction, PawRuntimeOptions};

pub type Error = paw_core::Error;

pub mod prelude {
    pub use super::config::{DevicePreference, PawLlamaCppConfig, PawLlamaCppConfigBuilder};
    pub use super::runtime::{PawFnLoader, PawFunction, PawRuntimeOptions};
    pub use paw_core::prelude::*;
}
