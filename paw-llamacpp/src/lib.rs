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
