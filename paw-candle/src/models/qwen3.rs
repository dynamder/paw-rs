#![allow(dead_code)]

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use super::{gguf_get, load_gguf_tensors, QuantizedModel};

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl Qwen3Config {
    pub fn from_gguf(content: &candle_core::quantized::gguf_file::Content) -> Self {
        let hidden_size = gguf_get(content, "llama.embedding_length", 1536);
        Self {
            hidden_size,
            num_hidden_layers: gguf_get(content, "llama.block_count", 28),
            num_attention_heads: gguf_get(content, "llama.attention.head_count", 12),
            num_key_value_heads: gguf_get(content, "llama.attention.head_count_kv", 12),
            head_dim: hidden_size / gguf_get(content, "llama.attention.head_count", 12),
            vocab_size: gguf_get(content, "llama.vocab_size", 151936),
        }
    }
}

pub struct Qwen3Model {
    config: Qwen3Config,
    device: Device,
    #[allow(dead_code)]
    tensors: HashMap<String, candle_core::quantized::QTensor>,
}

impl Qwen3Model {
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self, candle_core::Error> {
        let (content, tensors) = load_gguf_tensors(path, device)?;
        let config = Qwen3Config::from_gguf(&content);
        Ok(Self {
            config,
            device: device.clone(),
            tensors,
        })
    }
}

impl QuantizedModel for Qwen3Model {
    fn device(&self) -> &Device {
        &self.device
    }
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
    fn forward(
        &mut self,
        _input_ids: &Tensor,
        _position: usize,
    ) -> Result<Tensor, candle_core::Error> {
        Err(candle_core::Error::msg("not implemented"))
    }
    fn embed_tokens(&self) -> &Tensor {
        panic!("not implemented")
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    fn head_dim(&self) -> usize {
        self.config.head_dim
    }
    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }
    fn eos_token_id(&self) -> u32 {
        151643
    }
    fn model_name(&self) -> &str {
        "Qwen/Qwen3-0.6B"
    }
}
