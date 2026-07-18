use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use std::path::Path;

use super::{gguf_get, load_gguf_tensors, QuantizedModel};
use crate::lora::{GgufLoraAdapter, LoraLayer};

// ── Config (inferred from tensor shapes) ──────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f64,
    pub rope_theta: f64,
}

impl Qwen3Config {
    pub fn from_tensors(
        content: &candle_core::quantized::gguf_file::Content,
        tensors: &ahash::HashMap<String, candle_core::quantized::QTensor>,
    ) -> Self {
        let h = |name: &str, dim: usize| -> usize {
            tensors
                .get(name)
                .and_then(|t| t.shape().dims().get(dim).copied())
                .unwrap_or(0)
        };
        let hidden_size = h("token_embd.weight", 1);
        let q_dim = h("blk.0.attn_q.weight", 0);
        let kv_dim = h("blk.0.attn_k.weight", 0);
        let head_cnt = gguf_get(content, "llama.attention.head_count", 16);
        let head_dim = q_dim / head_cnt.max(1);
        Self {
            hidden_size,
            num_hidden_layers: gguf_get(content, "llama.block_count", 28),
            num_attention_heads: head_cnt,
            num_key_value_heads: if kv_dim > 0 {
                kv_dim / head_dim
            } else {
                head_cnt
            },
            head_dim,
            intermediate_size: h("blk.0.ffn_gate.weight", 0),
            vocab_size: h("token_embd.weight", 0),
            max_position_embeddings: gguf_get(content, "llama.context_length", 32768),
            layer_norm_epsilon: content
                .metadata
                .get("llama.attention.layer_norm_rms_epsilon")
                .and_then(|v| match v {
                    candle_core::quantized::gguf_file::Value::F32(f) => Some(*f as f64),
                    candle_core::quantized::gguf_file::Value::F64(f) => Some(*f),
                    _ => None,
                })
                .unwrap_or(1e-6),
            rope_theta: content
                .metadata
                .get("llama.rope.freq_base")
                .and_then(|v| match v {
                    candle_core::quantized::gguf_file::Value::F32(f) => Some(*f as f64),
                    candle_core::quantized::gguf_file::Value::F64(f) => Some(*f),
                    _ => None,
                })
                .unwrap_or(1_000_000.0),
        }
    }
}

// ── Block (weights stored as QMatMul for memory efficiency) ───────────

pub struct Qwen3Block {
    attn_norm: Tensor,
    attn_q: QMatMul,
    attn_k: QMatMul,
    attn_v: QMatMul,
    attn_q_norm: Tensor,
    attn_k_norm: Tensor,
    attn_out: QMatMul,
    ffn_norm: Tensor,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    // LoRA
    lora_q: Option<LoraLayer>,
    lora_k: Option<LoraLayer>,
    lora_v: Option<LoraLayer>,
    lora_output: Option<LoraLayer>,
    lora_gate: Option<LoraLayer>,
    lora_up: Option<LoraLayer>,
    lora_down: Option<LoraLayer>,
}

// ── Model ─────────────────────────────────────────────────────────────

pub struct Qwen3Model {
    config: Qwen3Config,
    device: Device,
    wte_weight: Tensor,       // dequantized for embedding lookup
    output_weight: QMatMul,    // quantized lm_head
    blocks: Vec<Qwen3Block>,
    output_norm: Tensor,
    kv_cache: Vec<Option<(Tensor, Tensor)>>, // per-layer (k, v) after QK-Norm & RoPE
    rope_cos: Tensor, // precomputed cos: [n_ctx, head_dim/2]
    rope_sin: Tensor, // precomputed sin: [n_ctx, head_dim/2]
}

impl Qwen3Model {
    const EPS: f64 = 1e-6;

    /// Detect SIMD support for fast quantized matmul on the current CPU.
    fn is_simd_quantized_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        { std::is_x86_feature_detected!("avx2") }
        #[cfg(target_arch = "aarch64")]
        { std::is_aarch64_feature_detected!("neon") }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { false }
    }

    /// Wrap a QTensor as QMatMul using quantized matmul when `quantized` is true,
    /// otherwise dequantize to f32 and wrap in QMatMul::Tensor (fallback).
    fn make_qmatmul(qt: QTensor, quantized: bool, device: &Device) -> Result<QMatMul, candle_core::Error> {
        if quantized {
            Ok(QMatMul::QTensor(std::sync::Arc::new(qt)))
        } else {
            let t = qt.dequantize(device)?;
            Ok(QMatMul::Tensor(t))
        }
    }

    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self, candle_core::Error> {
        let (content, mut tensors) = load_gguf_tensors(path, device)?;
        let config = Qwen3Config::from_tensors(&content, &tensors);

        let use_quantized = Self::is_simd_quantized_available();

        let mut take = |name: &str| -> QTensor {
            tensors
                .remove(name)
                .unwrap_or_else(|| panic!("Missing: {name}"))
        };

        let wte_q = take("token_embd.weight");
        let wte_weight = wte_q.dequantize(device)?;
        let output_weight = Self::make_qmatmul(take("output.weight"), use_quantized, device)?;
        let output_norm = take("output_norm.weight").dequantize(device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("blk.{i}.");
            blocks.push(Qwen3Block {
                attn_norm: take(&format!("{p}attn_norm.weight")).dequantize(device)?,
                attn_q: Self::make_qmatmul(take(&format!("{p}attn_q.weight")), use_quantized, device)?,
                attn_k: Self::make_qmatmul(take(&format!("{p}attn_k.weight")), use_quantized, device)?,
                attn_v: Self::make_qmatmul(take(&format!("{p}attn_v.weight")), use_quantized, device)?,
                attn_q_norm: take(&format!("{p}attn_q_norm.weight")).dequantize(device)?,
                attn_k_norm: take(&format!("{p}attn_k_norm.weight")).dequantize(device)?,
                attn_out: Self::make_qmatmul(take(&format!("{p}attn_output.weight")), use_quantized, device)?,
                ffn_norm: take(&format!("{p}ffn_norm.weight")).dequantize(device)?,
                ffn_gate: Self::make_qmatmul(take(&format!("{p}ffn_gate.weight")), use_quantized, device)?,
                ffn_up: Self::make_qmatmul(take(&format!("{p}ffn_up.weight")), use_quantized, device)?,
                ffn_down: Self::make_qmatmul(take(&format!("{p}ffn_down.weight")), use_quantized, device)?,
                lora_q: None,
                lora_k: None,
                lora_v: None,
                lora_output: None,
                lora_gate: None,
                lora_up: None,
                lora_down: None,
            });
        }

        eprintln!(
            "Qwen3 loaded: {} layers, hidden={}, heads={}, kv={}, dim={}, intermediate={}, vocab={}, theta={}, eps={}",
            config.num_hidden_layers, config.hidden_size,
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.intermediate_size, config.vocab_size,
            config.rope_theta, config.layer_norm_epsilon,
        );

        let kv_cache = vec![None; config.num_hidden_layers];

        // Precompute RoPE cos/sin for all positions up to max_position_embeddings
        let half = config.head_dim / 2;
        let n_ctx = config.max_position_embeddings.min(32768);
        let theta_ln = (config.rope_theta).ln() as f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| ((-2.0 * i as f32) * theta_ln / config.head_dim as f32).exp())
            .collect();
        let inv_freq_t = Tensor::from_slice(&inv_freq, (1, half), device)?;
        let positions = Tensor::arange(0f32, n_ctx as f32, device)?
            .reshape((n_ctx, 1))?;
        let angles = positions.matmul(&inv_freq_t)?;
        let rope_cos = angles.cos()?;
        let rope_sin = angles.sin()?;

        Ok(Self {
            config,
            device: device.clone(),
            wte_weight,
            output_weight,
            blocks,
            output_norm,
            kv_cache,
            rope_cos,
            rope_sin,
        })
    }

    fn apply_lora(&mut self, adapter: &GgufLoraAdapter) -> usize {
        let mut count = 0;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            for suffix in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                let key = format!("blk.{i}.{suffix}");
                if let Some(layer) = adapter.layers.get(&key) {
                    match suffix {
                        "attn_q" => blk.lora_q = Some(layer.clone()),
                        "attn_k" => blk.lora_k = Some(layer.clone()),
                        "attn_v" => blk.lora_v = Some(layer.clone()),
                        "attn_output" => blk.lora_output = Some(layer.clone()),
                        "ffn_gate" => blk.lora_gate = Some(layer.clone()),
                        "ffn_up" => blk.lora_up = Some(layer.clone()),
                        "ffn_down" => blk.lora_down = Some(layer.clone()),
                        _ => unreachable!(),
                    }
                    count += 1;
                }
            }
        }
        count
    }

    /// Fuse LoRA adapters directly into weight tensors.
    /// This eliminates the per-step LoRA side-path computation.
    fn rms_norm(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, candle_core::Error> {
        candle_nn::ops::rms_norm(x, w, Self::EPS as f32)
    }

    /// Causal mask for partial prefill when prefix KV cache exists.
    /// new_len = number of new input tokens, total_len = cached_len + new_len.
    /// New tokens attend to all cached tokens + causal among themselves.
    fn partial_causal_mask(new_len: usize, total_len: usize, device: &Device) -> Result<Tensor, candle_core::Error> {
        let cached_len = total_len - new_len;
        let idx_f = Tensor::arange(0f32, total_len as f32, device)?;
        let row_f = Tensor::arange(cached_len as f32, total_len as f32, device)?.unsqueeze(1)?;
        let col_f = idx_f.unsqueeze(0)?;
        let ge = row_f.broadcast_ge(&col_f)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (new_len, total_len), device)?;
        let zeros = Tensor::zeros((new_len, total_len), DType::F32, device)?;
        let mask = ge.where_cond(&zeros, &neg_inf)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }

    fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor, candle_core::Error> {
        let idx_f = Tensor::arange(0f32, seq_len as f32, device)?;
        let row = idx_f.unsqueeze(1)?;
        let col = idx_f.unsqueeze(0)?;
        let ge = row.broadcast_ge(&col)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
        let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
        let mask = ge.where_cond(&zeros, &neg_inf)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }

    fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor), candle_core::Error> {
        let half = self.config.head_dim / 2;
        // Use precomputed cos/sin from cache
        let cos = self.rope_cos.narrow(0, start_pos, seq_len)?;
        let sin = self.rope_sin.narrow(0, start_pos, seq_len)?;
        let cos_2d = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, half]
        let sin_2d = sin.unsqueeze(0)?.unsqueeze(0)?;
        let rope_emb = |t: &Tensor| -> Result<Tensor, candle_core::Error> {
            let t = t.to_dtype(DType::F32)?;
            let t_c = t.contiguous()?;
            let half_dim = t_c.dim(3)?;
            let even = t_c.narrow(3, 0, half)?;
            let odd = t_c.narrow(3, half, half_dim - half)?;
            let rotated_even = (even.broadcast_mul(&cos_2d)? - odd.broadcast_mul(&sin_2d)?)?;
            let rotated_odd = (even.broadcast_mul(&sin_2d)? + odd.broadcast_mul(&cos_2d)?)?;
            Tensor::cat(&[&rotated_even, &rotated_odd], 3)
        };
        Ok((rope_emb(q)?, rope_emb(k)?))
    }

    fn gqa_repeat_static(kv: &Tensor, n_q: usize, n_kv: usize) -> Result<Tensor, candle_core::Error> {
        if n_q == n_kv {
            return Ok(kv.clone());
        }
        let rep = n_q / n_kv;
        let b = kv.dim(0)?;
        let n_kv_actual = kv.dim(1)?;
        let seq_len = kv.dim(2)?;
        let head_dim = kv.dim(3)?;
        let reshaped = kv.reshape((b, n_kv_actual, 1, seq_len, head_dim))?;
        let repeated = reshaped.repeat(&[1, 1, rep, 1, 1])?;
        repeated.reshape((b, n_q, seq_len, head_dim))
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
        input_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor, candle_core::Error> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let device = &self.device;
        let is_prefill = self.kv_cache[0].is_none();
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Embedding lookup
        let flat = input_ids.flatten_all()?;
        let mut h = self.wte_weight.index_select(&flat, 0)?;
        h = h.reshape((_b_sz, seq_len, self.config.hidden_size))?;

        let mask = if is_prefill {
            // Full prefill: causal mask over all seq_len tokens
            Some(Self::causal_mask(seq_len, device)?)
        } else {
            None
        };

        for (i, blk) in self.blocks.iter().enumerate() {
            let h_ln = self.rms_norm(&h, &blk.attn_norm)?;

            let q = blk.attn_q.forward(&h_ln)?;
            let k = blk.attn_k.forward(&h_ln)?;
            let v = blk.attn_v.forward(&h_ln)?;

            let q = match blk.lora_q { Some(ref l) => (q + l.apply(&h_ln)?)?, None => q };
            let k = match blk.lora_k { Some(ref l) => (k + l.apply(&h_ln)?)?, None => k };
            let v = match blk.lora_v { Some(ref l) => (v + l.apply(&h_ln)?)?, None => v };

            // Multi-head reshape
            let q = q.reshape((1, seq_len, n_heads, head_dim))?.transpose(1, 2)?.contiguous()?;
            let k = k.reshape((1, seq_len, n_kv_heads, head_dim))?.transpose(1, 2)?.contiguous()?;
            let v = v.reshape((1, seq_len, n_kv_heads, head_dim))?.transpose(1, 2)?.contiguous()?;

            // QK-Norm
            let q = self.rms_norm(&q, &blk.attn_q_norm)?;
            let k = self.rms_norm(&k, &blk.attn_k_norm)?;

            // RoPE with absolute positions
            let (q, k) = self.apply_rotary_emb(&q, &k, position, seq_len)?;

            // KV cache
            // Determine mask for this layer
            let use_mask = if is_prefill {
                mask.as_ref().map(|m| m.clone())
            } else if seq_len > 1 {
                // Partial prefill: prefix cache exists + batch of new tokens
                let cached_len = self.kv_cache[i].as_ref().map(|(k, _)| k.dim(2)).unwrap_or(Ok(0)).unwrap_or(0);
                let total_seq = cached_len + seq_len;
                Some(Self::partial_causal_mask(seq_len, total_seq, device)?)
            } else {
                None
            };

            let (k_cat, v_cat) = match self.kv_cache[i].take() {
                Some((k_cached, v_cached)) => {
                    let k_new = Tensor::cat(&[&k_cached, &k], 2)?.contiguous()?;
                    let v_new = Tensor::cat(&[&v_cached, &v], 2)?.contiguous()?;
                    self.kv_cache[i] = Some((k_new.clone(), v_new.clone()));
                    (k_new, v_new)
                }
                None => {
                    self.kv_cache[i] = Some((k.clone(), v.clone()));
                    (k, v)
                }
            };

            // GQA repeat KV
            let k = Self::gqa_repeat_static(&k_cat, n_heads, n_kv_heads)?;
            let v = Self::gqa_repeat_static(&v_cat, n_heads, n_kv_heads)?;

            // Scaled dot-product attention
            let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let scores = match use_mask.as_ref() {
                Some(m) => scores.broadcast_add(m)?,
                None => scores,
            };
            let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
            let h_attn = weights.matmul(&v)?;
            let h_attn = h_attn.transpose(1, 2)?.reshape((1, seq_len, n_heads * head_dim))?;

            // Output projection + LoRA
            let attn_delta = match blk.lora_output {
                Some(ref l) => Some(l.apply(&h_attn)?),
                None => None,
            };
            let h_attn = blk.attn_out.forward(&h_attn)?;
            let h_attn = match attn_delta {
                Some(ref d) => (h_attn + d)?,
                None => h_attn,
            };
            h = (h + h_attn)?;

            // Pre-FFN RMSNorm
            let h_ln = self.rms_norm(&h, &blk.ffn_norm)?;

            // SwiGLU MLP
            let gate = blk.ffn_gate.forward(&h_ln)?;
            let gate = match blk.lora_gate {
                Some(ref l) => (gate + l.apply(&h_ln)?)?,
                None => gate,
            };
            let up = blk.ffn_up.forward(&h_ln)?;
            let up = match blk.lora_up {
                Some(ref l) => (up + l.apply(&h_ln)?)?,
                None => up,
            };
            let activated = (gate.silu()? * up)?;
            let down_delta = match blk.lora_down {
                Some(ref l) => Some(l.apply(&activated)?),
                None => None,
            };
            let h_mlp = blk.ffn_down.forward(&activated)?;
            let h_mlp = match down_delta {
                Some(ref d) => (h_mlp + d)?,
                None => h_mlp,
            };
            h = (h + h_mlp)?;
        }

        // Final norm + lm_head
        h = self.rms_norm(&h, &self.output_norm)?;
        h = self.output_weight.forward(&h)?;
        Ok(h)
    }

    fn embed_tokens(&self) -> &Tensor {
        &self.wte_weight
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
    fn set_lora(&mut self, adapter: &GgufLoraAdapter) -> usize {
        self.apply_lora(adapter)
    }

    fn set_prefix_cache(&mut self, prefix: &[(Tensor, Tensor)]) {
        for (i, pair) in prefix.iter().enumerate().take(self.config.num_hidden_layers) {
            self.kv_cache[i] = Some(pair.clone());
        }
    }

    fn extract_prefix_cache(&self, prefix_len: usize) -> Option<Vec<(Tensor, Tensor)>> {
        if self.kv_cache[0].is_none() {
            return None;
        }
        let mut result = Vec::with_capacity(self.config.num_hidden_layers);
        for entry in &self.kv_cache {
            match entry {
                Some((k, v)) => {
                    let k_prefix = k
                        .narrow(2, 0, prefix_len)
                        .ok()?
                        .contiguous()
                        .ok()?;
                    let v_prefix = v
                        .narrow(2, 0, prefix_len)
                        .ok()?
                        .contiguous()
                        .ok()?;
                    result.push((k_prefix, v_prefix));
                }
                None => return None,
            }
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::HFClient;

    #[tokio::test]
    #[ignore = "downloads ~594MB model on first run"]
    async fn forward_output_shape() {
        let client = HFClient::new().expect("hf-hub client");
        let path = client
            .model("programasweights", "Qwen3-0.6B-GGUF-Q6_K")
            .download_file()
            .filename("qwen3-0.6b-q6_k.gguf")
            .send()
            .await
            .expect("download Qwen3 GGUF");
        let device = Device::Cpu;
        let mut model = Qwen3Model::from_gguf(&path, &device).expect("load model");

        let input = Tensor::new(&[100u32, 200, 300, 400, 500], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let logits = model.forward(&input, 0).expect("forward pass");
        assert_eq!(logits.dims(), &[1, 5, 151936]);

        let last = logits.squeeze(0).unwrap().get(4).unwrap().to_vec1::<f32>().unwrap();
        assert!(
            last.iter().filter(|v| v.is_finite()).count() > last.len() / 2,
            "most logits should be finite"
        );
    }
}
