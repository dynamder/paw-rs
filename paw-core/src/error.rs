use std::path::PathBuf;

/// Error types for paw-core operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {status} — {message}")]
    Api { status: u16, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("Bundle missing required file: {0}")]
    MissingFile(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Timeout: program assets not ready after {0}s")]
    Timeout(u64),

    #[error("Path contains unsafe characters: {0}")]
    UnsafePath(PathBuf),

    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),

    #[error("Format error: {0}")]
    Format(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
