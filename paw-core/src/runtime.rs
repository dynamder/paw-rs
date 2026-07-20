use crate::error::Error;

#[derive(Debug, Clone)]
pub struct PawRuntimeOptions {
    pub max_tokens: Option<usize>,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for PawRuntimeOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.0,
            top_p: 1.0,
        }
    }
}

pub trait PawFnTrait: Send {
    fn run(&mut self, input: &str) -> Result<String, Error>;
    fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error>;
    fn interpreter(&self) -> &str;
}
