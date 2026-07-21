//! GGUF LoRA adapter parser and side-path merger.
//!
//! Parses the Q4_0 quantized LoRA adapter from a `.paw` bundle's
//! `adapter.gguf` file, dequantizes the tensors, and provides
//! side-path application for the GPT-2 forward pass.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{quantized::gguf_file, Device, Tensor};
use paw_core::Error;

/// A single LoRA layer pair (A, B) for one weight matrix.
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// A matrix: [rank, in_dim]
    pub a: Tensor,
    /// B matrix: [out_dim, rank]
    pub b: Tensor,
    /// Scaling factor = alpha / rank
    pub scale: f32,
}

impl LoraLayer {
    /// Apply the LoRA side-path: returns `B(A(x)) * scale`
    pub fn apply(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let a_t = self.a.transpose(0, 1)?.unsqueeze(0)?;
        let b_t = self.b.transpose(0, 1)?.unsqueeze(0)?;
        let intermediate = x.matmul(&a_t)?;
        let result = intermediate.matmul(&b_t)?;
        Ok((result * (self.scale as f64))?)
    }
}

/// A parsed GGUF LoRA adapter.
pub struct GgufLoraAdapter {
    /// Per-weight-matrix LoRA pairs, keyed by weight name (e.g. `"blk.0.attn_qkv"`).
    pub layers: ahash::HashMap<String, LoraLayer>,
}

impl GgufLoraAdapter {
    /// Parse a GGUF LoRA adapter file.
    ///
    /// Expects tensor names like:
    /// - `blk.{i}.attn_qkv.weight.lora_a` — [rank, hidden]
    /// - `blk.{i}.attn_qkv.weight.lora_b` — [out_dim, rank]
    pub fn from_gguf_file<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self, Error> {
        // mmap the GGUF file for zero-copy read access
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| Error::Other(format!("open adapter: {e}")))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Other(format!("mmap adapter: {e}")))?;
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| Error::Other(format!("read gguf: {e}")))?;

        // Extract alpha and rank from metadata
        let alpha = content
            .metadata
            .get("lora.alpha")
            .and_then(|v| match v {
                gguf_file::Value::F32(f) => Some(*f),
                gguf_file::Value::I32(i) => Some(*i as f32),
                gguf_file::Value::U32(u) => Some(*u as f32),
                _ => None,
            })
            .unwrap_or(16.0);

        // Load all tensors in parallel using rayon (same pattern as load_gguf_tensors)
        let names: Vec<&String> = content.tensor_infos.keys().collect();
        let n = names.len();
        use std::sync::Mutex;
        let tensors: Mutex<std::collections::HashMap<String, Tensor, ahash::RandomState>> =
            Mutex::new(HashMap::with_capacity_and_hasher(
                n,
                ahash::RandomState::new(),
            ));
        let load_err = Mutex::new(None::<Error>);

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
                                *load_err.lock().unwrap() =
                                    Some(Error::Other(format!("load tensor {name}: {e}")));
                                return;
                            }
                        };
                        let t = match qtensor.dequantize(device) {
                            Ok(t) => t,
                            Err(e) => {
                                *load_err.lock().unwrap() =
                                    Some(Error::Other(format!("dequantize {name}: {e}")));
                                return;
                            }
                        };
                        tensors.lock().unwrap().insert(name.to_string(), t);
                    }
                });
            }
        });

        if let Some(e) = load_err.into_inner().unwrap() {
            return Err(e);
        }
        let mut tensors = tensors.into_inner().unwrap();

        // Pair into LoraLayers: strip `.lora_a` / `.lora_b` suffix
        let mut layers: ahash::HashMap<String, LoraLayer> =
            HashMap::with_capacity_and_hasher(content.tensor_infos.len() / 2, Default::default());
        let suffix_a = ".lora_a";
        let suffix_b = ".lora_b";
        let keys: Vec<String> = tensors.keys().cloned().collect();

        for key in &keys {
            if let Some(name) = key.strip_suffix(suffix_a) {
                let b_key = format!("{name}{suffix_b}");
                if let Some(a) = tensors.remove(key) {
                    if let Some(b) = tensors.remove(&b_key) {
                        let rank = a
                            .dim(0)
                            .map_err(|e| Error::Other(format!("lora dim: {e}")))?;
                        let base = name.strip_suffix(".weight").unwrap_or(name);
                        layers.insert(
                            base.to_string(),
                            LoraLayer {
                                a,
                                b,
                                scale: alpha / rank as f32,
                            },
                        );
                    }
                }
            }
        }

        Ok(Self { layers })
    }

    /// Number of parsed LoRA layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }
}
