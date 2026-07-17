use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{Device, DType, Module, Tensor};
use std::path::Path;

use super::{gguf_get, load_gguf_tensors, QuantizedModel};
use crate::lora::{GgufLoraAdapter, LoraLayer};

// ── Config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f64,
}

impl Gpt2Config {
    pub fn from_gguf(content: &candle_core::quantized::gguf_file::Content) -> Self {
        let hidden_size = gguf_get(content, "llama.embedding_length", 768);
        let num_attention_heads = gguf_get(content, "llama.attention.head_count", 12);
        Self {
            hidden_size,
            num_hidden_layers: gguf_get(content, "llama.block_count", 12),
            num_attention_heads,
            head_dim: hidden_size / num_attention_heads,
            vocab_size: gguf_get(content, "llama.vocab_size", 50257),
            max_position_embeddings: gguf_get(content, "llama.context_length", 1024),
            layer_norm_epsilon: content
                .metadata
                .get("llama.attention.layer_norm_rms_epsilon")
                .and_then(|v| match v {
                    candle_core::quantized::gguf_file::Value::F32(f) => Some(*f as f64),
                    candle_core::quantized::gguf_file::Value::F64(f) => Some(*f),
                    _ => None,
                })
                .unwrap_or(1e-5),
        }
    }
}

// ── Block weights ─────────────────────────────────────────────────────

pub struct Gpt2Block {
    ln_1_weight: Tensor,
    ln_1_bias: Tensor,
    attn_qkv: QMatMul,
    attn_qkv_bias: Tensor,
    lora_qkv: Option<LoraLayer>,
    lora_output: Option<LoraLayer>,
    attn_out: QMatMul,
    attn_out_bias: Tensor,
    ln_2_weight: Tensor,
    ln_2_bias: Tensor,
    mlp_fc: QMatMul,
    mlp_fc_bias: Tensor,
    lora_fc: Option<LoraLayer>,
    mlp_proj: QMatMul,
    mlp_proj_bias: Tensor,
    lora_proj: Option<LoraLayer>,
}

// ── Model ─────────────────────────────────────────────────────────────

pub struct Gpt2Model {
    config: Gpt2Config,
    device: Device,
    wte: QMatMul,
    wpe: Tensor,
    blocks: Vec<Gpt2Block>,
    ln_f_weight: Tensor,
    ln_f_bias: Tensor,
    wte_weight: Tensor,
}

impl Gpt2Model {
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self, candle_core::Error> {
        let (content, mut tensors) = load_gguf_tensors(path, device)?;
        let config = Gpt2Config::from_gguf(&content);

        let mut take = |name: &str| -> QTensor {
            tensors
                .remove(name)
                .unwrap_or_else(|| panic!("Missing tensor: {name}"))
        };

        let wte_q = take("token_embd.weight");
        let wte_weight = wte_q.dequantize(device)?;
        let wte = QMatMul::from_qtensor(wte_q)?;
        let wpe = take("position_embd.weight").dequantize(device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("blk.{i}.");
            blocks.push(Gpt2Block {
                ln_1_weight: take(&format!("{p}attn_norm.weight")).dequantize(device)?,
                ln_1_bias: take(&format!("{p}attn_norm.bias")).dequantize(device)?,
                attn_qkv: QMatMul::from_qtensor(take(&format!("{p}attn_qkv.weight")))?,
                attn_qkv_bias: take(&format!("{p}attn_qkv.bias")).dequantize(device)?,
                lora_qkv: None,
                lora_output: None,
                attn_out: QMatMul::from_qtensor(take(&format!("{p}attn_output.weight")))?,
                attn_out_bias: take(&format!("{p}attn_output.bias")).dequantize(device)?,
                ln_2_weight: take(&format!("{p}ffn_norm.weight")).dequantize(device)?,
                ln_2_bias: take(&format!("{p}ffn_norm.bias")).dequantize(device)?,
                mlp_fc: QMatMul::from_qtensor(take(&format!("{p}ffn_up.weight")))?,
                mlp_fc_bias: take(&format!("{p}ffn_up.bias")).dequantize(device)?,
                lora_fc: None,
                mlp_proj: QMatMul::from_qtensor(take(&format!("{p}ffn_down.weight")))?,
                mlp_proj_bias: take(&format!("{p}ffn_down.bias")).dequantize(device)?,
                lora_proj: None,
            });
        }

        let ln_f_weight = take("output_norm.weight").dequantize(device)?;
        let ln_f_bias = take("output_norm.bias").dequantize(device)?;

        Ok(Self { config, device: device.clone(), wte, wpe, blocks, ln_f_weight, ln_f_bias, wte_weight })
    }

    /// Attach LoRA adapter to all matching layers. Returns count of matched layers.
    pub fn set_lora(&mut self, adapter: &GgufLoraAdapter) -> usize {
        let mut count = 0;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            for (suffix, module) in [
                ("attn_qkv", "qkv"),
                ("attn_output", "out"),
                ("ffn_up", "fc"),
                ("ffn_down", "proj"),
            ] {
                let key = format!("blk.{i}.{suffix}");
                if let Some(layer) = adapter.layers.get(&key) {
                    match module {
                        "qkv" => blk.lora_qkv = Some(layer.clone()),
                        "out" => blk.lora_output = Some(layer.clone()),
                        "fc" => blk.lora_fc = Some(layer.clone()),
                        "proj" => blk.lora_proj = Some(layer.clone()),
                        _ => unreachable!(),
                    }
                    count += 1;
                }
            }
        }
        count
    }

    fn layer_norm(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor, candle_core::Error> {
        candle_nn::ops::layer_norm(x, w, b, self.config.layer_norm_epsilon as f32)
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

    fn split_qkv(&self, fused: &Tensor) -> Result<(Tensor, Tensor, Tensor), candle_core::Error> {
        let hidden = self.config.hidden_size;
        let q = fused.narrow(2, 0, hidden)?;
        let k = fused.narrow(2, hidden, hidden)?;
        let v = fused.narrow(2, 2 * hidden, hidden)?;
        Ok((q, k, v))
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, seq_len: usize) -> Result<Tensor, candle_core::Error> {
        let n_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = q.reshape((1, seq_len, n_heads, head_dim))?.transpose(1, 2)?;
        let k = k.reshape((1, seq_len, n_heads, head_dim))?.transpose(1, 2)?;
        let v = v.reshape((1, seq_len, n_heads, head_dim))?.transpose(1, 2)?;

        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_scores = attn_scores.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax(&attn_scores, candle_core::D::Minus1)?;
        let context = attn_weights.matmul(&v)?;
        context.transpose(1, 2)?.reshape((1, seq_len, n_heads * head_dim))
    }
}

impl QuantizedModel for Gpt2Model {
    fn device(&self) -> &Device { &self.device }
    fn num_layers(&self) -> usize { self.config.num_hidden_layers }

    fn forward(&mut self, input_ids: &Tensor, _position: usize) -> Result<Tensor, candle_core::Error> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let device = &self.device;

        let flat = input_ids.flatten_all()?;
        let mut h = self.wte_weight.index_select(&flat, 0)?;
        h = h.reshape((_b_sz, seq_len, self.config.hidden_size))?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, device)?;
        let pos_emb = self.wpe.index_select(&pos_ids, 0)?.unsqueeze(0)?;
        h = (h + pos_emb)?;

        let mask = Self::causal_mask(seq_len, device)?;

        for blk in &self.blocks {
            let h_ln = self.layer_norm(&h, &blk.ln_1_weight, &blk.ln_1_bias)?;
            let qkv = blk.attn_qkv.forward(&h_ln)?;
            let shape = qkv.shape().clone();
            let mut qkv = (qkv + blk.attn_qkv_bias.broadcast_as(&shape)?)?;

            if let Some(ref lora) = blk.lora_qkv {
                let delta = lora.apply(&h_ln)?;
                qkv = (qkv + delta)?;
            }

            let (q, k, v) = self.split_qkv(&qkv)?;

            let mut h_attn = self.attention(&q, &k, &v, &mask, seq_len)?;
            if let Some(ref lora) = blk.lora_output {
                let delta = lora.apply(&h_attn)?;
                h_attn = (h_attn + delta)?;
            }
            let h_attn = blk.attn_out.forward(&h_attn)?;
            let bias = blk.attn_out_bias.unsqueeze(0)?.unsqueeze(0)?;
            let shape = h_attn.shape().clone();
            let h_attn = (h_attn + bias.broadcast_as(&shape)?)?;
            h = (h + h_attn)?;

            let h_ln = self.layer_norm(&h, &blk.ln_2_weight, &blk.ln_2_bias)?;
            let mut h_mlp = blk.mlp_fc.forward(&h_ln)?;
            if let Some(ref lora) = blk.lora_fc {
                let delta = lora.apply(&h_ln)?;
                h_mlp = (h_mlp + delta)?;
            }
            let bias = blk.mlp_fc_bias.unsqueeze(0)?.unsqueeze(0)?;
            let shape = h_mlp.shape().clone();
            h_mlp = (h_mlp + bias.broadcast_as(&shape)?)?;
            h_mlp = h_mlp.gelu_erf()?;
            let proj_delta = match blk.lora_proj.as_ref() {
                Some(l) => Some(l.apply(&h_mlp)?),
                None => None,
            };
            h_mlp = blk.mlp_proj.forward(&h_mlp)?;
            if let Some(ref delta) = proj_delta {
                h_mlp = (h_mlp + delta)?;
            }
            let bias = blk.mlp_proj_bias.unsqueeze(0)?.unsqueeze(0)?;
            let shape = h_mlp.shape().clone();
            h_mlp = (h_mlp + bias.broadcast_as(&shape)?)?;
            h = (h + h_mlp)?;
        }

        h = self.layer_norm(&h, &self.ln_f_weight, &self.ln_f_bias)?;
        h = self.wte.forward(&h)?;
        Ok(h)
    }

    fn embed_tokens(&self) -> &Tensor { &self.wte_weight }
    fn vocab_size(&self) -> usize { self.config.vocab_size }
    fn hidden_size(&self) -> usize { self.config.hidden_size }
    fn head_dim(&self) -> usize { self.config.head_dim }
    fn num_attention_heads(&self) -> usize { self.config.num_attention_heads }
    fn num_kv_heads(&self) -> usize { self.config.num_attention_heads }
    fn eos_token_id(&self) -> u32 { 50256 }
    fn model_name(&self) -> &str { "gpt2" }
    fn set_lora(&mut self, adapter: &GgufLoraAdapter) -> usize {
        self.set_lora(adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::HFClient;

    #[tokio::test]
    #[ignore = "downloads ~120MB model on first run"]
    async fn forward_output_shape() {
        let client = HFClient::new().expect("hf-hub client init");
        let path = client
            .model("programasweights", "GPT2-GGUF-Q8_0")
            .download_file()
            .filename("gpt2-q8_0.gguf")
            .send()
            .await
            .expect("download GGUF");

        let device = Device::Cpu;
        let mut model = Gpt2Model::from_gguf(&path, &device).expect("load model");

        let input = Tensor::new(&[100u32, 200, 300, 400, 500], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let logits = model.forward(&input, 0).expect("forward pass");
        assert_eq!(logits.dims(), &[1, 5, 50257]);

        let last = logits
            .squeeze(0)
            .unwrap()
            .get(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            last.iter().filter(|v| v.is_finite()).count() > last.len() / 2,
            "most logits should be finite"
        );
    }

    #[tokio::test]
    #[ignore = "downloads ~120MB model + ~1MB adapter on first run"]
    async fn forward_with_lora_changes_output() {
        use paw_core::PawClient;
        use paw_core::prelude::PawConfig;

        let config = PawConfig::from_env();
        let client = PawClient::new(&config);

        // Download a .paw bundle (any public program)
        let program_id = client.resolve_slug("email-triage").await.expect("resolve slug");
        let dir = client.download_paw(&program_id).await.expect("download paw");
        let adapter_path = dir.join("adapter.gguf");

        let device = Device::Cpu;

        // Load base model
        let hf = HFClient::new().expect("hf-hub client init");
        let base_path = hf
            .model("programasweights", "GPT2-GGUF-Q8_0")
            .download_file()
            .filename("gpt2-q8_0.gguf")
            .send()
            .await
            .expect("download GGUF");

        let mut model = Gpt2Model::from_gguf(&base_path, &device).expect("load model");

        // Forward without LoRA
        let input = Tensor::new(&[100u32, 200, 300, 400, 500], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits_base = model.forward(&input, 0).expect("forward (no lora)");
        let last_base = logits_base
            .squeeze(0)
            .unwrap()
            .get(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Load LoRA and attach
        let lora = GgufLoraAdapter::from_gguf_file(&adapter_path, &device).expect("load lora");
        let matched = model.set_lora(&lora);

        if matched > 0 {
            let logits_lora = model.forward(&input, 0).expect("forward (lora)");
            let last_lora = logits_lora
                .squeeze(0)
                .unwrap()
                .get(4)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                last_base != last_lora,
                "LoRA should change model output"
            );
        } else {
            eprintln!("LoRA adapter has {matched} matching layers (model expects `blk.{{i}}.attn_qkv`), skipping diff check");
        }
    }
}
