use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
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
        tensors: &std::collections::HashMap<String, candle_core::quantized::QTensor>,
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

// ── Block (all weights dequantized to f32) ────────────────────────────

pub struct Qwen3Block {
    attn_norm: Tensor,
    attn_q: Tensor,   // [q_dim, hidden]
    attn_k: Tensor,   // [kv_dim, hidden]
    attn_v: Tensor,   // [kv_dim, hidden]
    attn_q_norm: Tensor,
    attn_k_norm: Tensor,
    attn_out: Tensor, // [hidden, q_dim]
    ffn_norm: Tensor,
    ffn_gate: Tensor, // [intermediate, hidden]
    ffn_up: Tensor,   // [intermediate, hidden]
    ffn_down: Tensor, // [hidden, intermediate]
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
    wte_weight: Tensor,     // dequantized for embedding lookup
    output_weight: Tensor,  // dequantized lm_head
    blocks: Vec<Qwen3Block>,
    output_norm: Tensor,
    kv_cache: Vec<Option<(Tensor, Tensor)>>, // per-layer (k, v) after QK-Norm & RoPE
}

impl Qwen3Model {
    const EPS: f64 = 1e-6;

    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self, candle_core::Error> {
        let (content, mut tensors) = load_gguf_tensors(path, device)?;
        let config = Qwen3Config::from_tensors(&content, &tensors);

        let mut take = |name: &str| -> QTensor {
            tensors
                .remove(name)
                .unwrap_or_else(|| panic!("Missing: {name}"))
        };

        let wte_q = take("token_embd.weight");
        let wte_weight = wte_q.dequantize(device)?;
        let output_weight = take("output.weight").dequantize(device)?;
        let output_norm = take("output_norm.weight").dequantize(device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("blk.{i}.");
            blocks.push(Qwen3Block {
                attn_norm: take(&format!("{p}attn_norm.weight")).dequantize(device)?,
                attn_q: take(&format!("{p}attn_q.weight")).dequantize(device)?,
                attn_k: take(&format!("{p}attn_k.weight")).dequantize(device)?,
                attn_v: take(&format!("{p}attn_v.weight")).dequantize(device)?,
                attn_q_norm: take(&format!("{p}attn_q_norm.weight")).dequantize(device)?,
                attn_k_norm: take(&format!("{p}attn_k_norm.weight")).dequantize(device)?,
                attn_out: take(&format!("{p}attn_output.weight")).dequantize(device)?,
                ffn_norm: take(&format!("{p}ffn_norm.weight")).dequantize(device)?,
                ffn_gate: take(&format!("{p}ffn_gate.weight")).dequantize(device)?,
                ffn_up: take(&format!("{p}ffn_up.weight")).dequantize(device)?,
                ffn_down: take(&format!("{p}ffn_down.weight")).dequantize(device)?,
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
        Ok(Self {
            config,
            device: device.clone(),
            wte_weight,
            output_weight,
            blocks,
            output_norm,
            kv_cache,
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
        let dim = self.config.head_dim;
        let device = &self.device;
        let half = dim / 2;
        let theta_ln = (self.config.rope_theta).ln() as f32;
        let freqs = Tensor::arange(0f32, half as f32, device)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&Tensor::full(-2.0 * theta_ln / dim as f32, half, device)?)?;
        let freqs = freqs.exp()?;
        let positions = Tensor::arange(start_pos as f32, (start_pos + seq_len) as f32, device)?.unsqueeze(1)?;
        let angles = positions.broadcast_mul(&freqs)?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        let rope_emb = |t: &Tensor| -> Result<Tensor, candle_core::Error> {
            let t = t.to_dtype(DType::F32)?;
            // Manual RoPE: q = [1, heads, seq, dim]
            // Split into halves: even and odd positions
            let cos_2d = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, half]
            let sin_2d = sin.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, half]
            let t_contig = t.contiguous()?;
            let even = t_contig.narrow(3, 0, half)?.to_dtype(DType::F32)?;
            let odd = t_contig.narrow(3, half, half)?.to_dtype(DType::F32)?;
            let rotated_even = (even.broadcast_mul(&cos_2d)? - odd.broadcast_mul(&sin_2d)?)?;
            let rotated_odd = (even.broadcast_mul(&sin_2d)? + odd.broadcast_mul(&cos_2d)?)?;
            Tensor::cat(&[&rotated_even, &rotated_odd], 3)
        };
        Ok((rope_emb(q)?, rope_emb(k)?))
    }

    /// Matmul for [batch, seq, in_dim] @ [in_dim, out_dim].
    fn flatten_bmm(x: &Tensor, w: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, s, d) = x.dims3()?;
        let x_2d = x.reshape((b * s, d))?;
        let r = x_2d.matmul(w)?;
        r.reshape((b, s, r.dim(1)?))
    }

    fn gqa_repeat_static(kv: &Tensor, n_q: usize, n_kv: usize) -> Result<Tensor, candle_core::Error> {
        if n_q == n_kv {
            return Ok(kv.clone());
        }
        let rep = n_q / n_kv;
        // kv: [1, n_kv, seq, head_dim]
        // Need each head repeated rep times consecutively:
        //   [h0, h0, ..., h1, h1, ..., h_{n_kv-1}, ..., h_{n_kv-1}]
        // NOT block-repeat: [h0, h1, ..., h_{n_kv-1}, h0, ..., h_{n_kv-1}]
        let n_kv_actual = kv.dim(1)?;
        let mut parts = Vec::with_capacity(n_kv_actual * rep);
        for h in 0..n_kv_actual {
            let head = kv.narrow(1, h, 1)?;
            for _ in 0..rep {
                parts.push(head.clone());
            }
        }
        Tensor::cat(&parts, 1)
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

            let q = Self::flatten_bmm(&h_ln, &blk.attn_q.t()?)?;
            let k = Self::flatten_bmm(&h_ln, &blk.attn_k.t()?)?;
            let v = Self::flatten_bmm(&h_ln, &blk.attn_v.t()?)?;

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
            let h_attn = Self::flatten_bmm(&h_attn, &blk.attn_out.t()?)?;
            let h_attn = match attn_delta {
                Some(ref d) => (h_attn + d)?,
                None => h_attn,
            };
            h = (h + h_attn)?;

            // Pre-FFN RMSNorm
            let h_ln = self.rms_norm(&h, &blk.ffn_norm)?;

            // SwiGLU MLP
            let gate = Self::flatten_bmm(&h_ln, &blk.ffn_gate.t()?)?;
            let gate = match blk.lora_gate {
                Some(ref l) => (gate + l.apply(&h_ln)?)?,
                None => gate,
            };
            let up = Self::flatten_bmm(&h_ln, &blk.ffn_up.t()?)?;
            let up = match blk.lora_up {
                Some(ref l) => (up + l.apply(&h_ln)?)?,
                None => up,
            };
            let activated = (gate.silu()? * up)?;
            let down_delta = match blk.lora_down {
                Some(ref l) => Some(l.apply(&activated)?),
                None => None,
            };
            let h_mlp = Self::flatten_bmm(&activated, &blk.ffn_down.t()?)?;
            let h_mlp = match down_delta {
                Some(ref d) => (h_mlp + d)?,
                None => h_mlp,
            };
            h = (h + h_mlp)?;
        }

        // Final norm + lm_head
        h = self.rms_norm(&h, &self.output_norm)?;
        h = Self::flatten_bmm(&h, &self.output_weight.t()?)?;
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
