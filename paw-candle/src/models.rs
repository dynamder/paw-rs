//! Model implementations for each supported architecture.
//!
//! Each module provides a quantized GGUF model that implements the
//! [`QuantizedModel`] trait used by the inference runtime.

/// Qwen3-0.6B model.
pub mod qwen3;

/// GPT-2 model (simple decoder-only transformer).
pub mod gpt2;

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::lora::{GgufLoraAdapter, LoraLayer};

/// Read a GGUF integer metadata field, supporting all numeric GGUF value types.
pub(crate) fn gguf_get(content: &gguf_file::Content, key: &str, default: usize) -> usize {
    use gguf_file::Value;
    content
        .metadata
        .get(key)
        .map(|v| match v {
            Value::U8(n) => *n as usize,
            Value::I8(n) => *n as usize,
            Value::U16(n) => *n as usize,
            Value::I16(n) => *n as usize,
            Value::U32(n) => *n as usize,
            Value::I32(n) => *n as usize,
            Value::U64(n) => *n as usize,
            Value::I64(n) => *n as usize,
            Value::String(s) => s.parse().unwrap_or(default),
            Value::F32(f) => *f as usize,
            Value::F64(f) => *f as usize,
            _ => default,
        })
        .unwrap_or(default)
}

/// Load all tensors from a GGUF file.
///
/// Uses mmap for zero-copy file access and rayon for parallel tensor
/// dequantization, plus ahash for faster HashMap lookups.
pub(crate) fn load_gguf_tensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<
    (
        gguf_file::Content,
        ahash::HashMap<String, candle_core::quantized::QTensor>,
    ),
    candle_core::Error,
> {
    // mmap the GGUF file for zero-copy read access
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    // Parse GGUF structure from the mmap'd data
    let content = gguf_file::Content::read(&mut cursor)?;

    // Collect tensor names for parallel loading
    let names: Vec<&String> = content.tensor_infos.keys().collect();
    let n = names.len();

    // Load all tensors in parallel using rayon.
    use std::sync::Mutex;
    let result: Mutex<
        std::collections::HashMap<String, candle_core::quantized::QTensor, ahash::RandomState>,
    > = Mutex::new(std::collections::HashMap::with_capacity_and_hasher(
        n,
        ahash::RandomState::new(),
    ));
    let load_err = Mutex::new(None::<candle_core::Error>);

    // Each thread creates its own Cursor from the shared mmap and reads
    // a different tensor.  The `content` is a parsed GGUF header shared
    // read-only — no mutation.
    let n_threads = rayon::current_num_threads().max(1);
    rayon::scope(|s| {
        for chunk in names.chunks((n + n_threads - 1) / n_threads) {
            let chunk: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            s.spawn(|_| {
                let mut local = std::io::Cursor::new(&mmap[..]);
                for name in chunk {
                    let qtensor = match content.tensor(&mut local, name, device) {
                        Ok(t) => t,
                        Err(e) => {
                            *load_err.lock().unwrap() = Some(e);
                            return;
                        }
                    };
                    result.lock().unwrap().insert(name.to_string(), qtensor);
                }
            });
        }
    });

    if let Some(e) = load_err.into_inner().unwrap() {
        return Err(e);
    }
    Ok((content, result.into_inner().unwrap()))
}

/// Extract the underlying weight tensor from a QMatMul as f32.
pub(crate) fn qmatmul_to_f32(qm: &QMatMul, device: &Device) -> Result<Tensor, candle_core::Error> {
    match qm {
        QMatMul::QTensor(qt) => qt.dequantize(device),
        QMatMul::Tensor(t) => Ok(t.clone()),
        QMatMul::TensorF16(t) => t.to_dtype(DType::F32),
    }
}

/// Fuse a LoRA layer into a QMatMul weight matrix in-place.
/// Replaces the weight with `W + B @ A * scale`, changing QMatMul to f32 Tensor.
pub(crate) fn fuse_lora_weight(
    qm: &mut QMatMul,
    lora: &LoraLayer,
    device: &Device,
) -> Result<(), candle_core::Error> {
    let w = qmatmul_to_f32(qm, device)?;
    let delta = (lora.b.matmul(&lora.a)? * (lora.scale as f64))?;
    let fused = (w + delta)?;
    *qm = QMatMul::Tensor(fused);
    Ok(())
}

/// A quantized model that can perform autoregressive generation.
pub trait QuantizedModel: Send {
    /// Returns the model's device.
    fn device(&self) -> &Device;

    /// Returns the number of layers in the model.
    fn num_layers(&self) -> usize;

    /// Forward pass: process `input_ids` at the given position index.
    ///
    /// - `input_ids`: shape `[1, seq_len]`, the token IDs to process.
    /// - `position`: the starting position in the sequence (for KV cache indexing).
    ///
    /// Returns logits of shape `[1, seq_len, vocab_size]`.
    fn forward(
        &mut self,
        input_ids: &Tensor,
        position: usize,
    ) -> std::result::Result<Tensor, candle_core::Error>;

    /// Get the embedding weight tensor (for tied embeddings / lm_head).
    fn embed_tokens(&self) -> &Tensor;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Hidden size.
    fn hidden_size(&self) -> usize;

    /// Head dimension.
    fn head_dim(&self) -> usize;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> usize;

    /// Number of KV heads (for GQA / MQA).
    fn num_kv_heads(&self) -> usize;

    /// Get the EOS token ID.
    fn eos_token_id(&self) -> u32 {
        0
    }

    /// Get the model name/identifier.
    fn model_name(&self) -> &str;

    /// Attach LoRA adapter to the model (default: no-op).
    fn set_lora(&mut self, _adapter: &GgufLoraAdapter) -> usize {
        0
    }

    /// Remove all LoRA adapters from the model, restoring it to base state.
    fn clear_lora(&mut self) {}

    /// Fuse LoRA adapters directly into weight tensors (default: no-op).
    /// Called after set_lora(). Eliminates per-step LoRA side-path computation.
    fn fuse_lora(&mut self) -> std::result::Result<(), candle_core::Error> {
        Ok(())
    }

    /// Load prefix KV cache into the model (default: no-op).
    fn set_prefix_cache(&mut self, _prefix: &[(Tensor, Tensor)]) {}

    /// Extract the first `prefix_len` tokens from each layer's KV cache.
    fn extract_prefix_cache(&self, _prefix_len: usize) -> Option<Vec<(Tensor, Tensor)>> {
        None
    }
}
