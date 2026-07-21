//! Example: download a .paw bundle, load GGUF base model + LoRA adapter,
//! and run forward pass to verify the full pipeline works.
//!
//! Automatically picks the correct base model based on the bundle's interpreter.
//! Usage:  cargo run --release --example verify_bundle

use candle_core::{Device, Tensor};
use hf_hub::HFClient;
use paw_candle::lora::GgufLoraAdapter;
use paw_candle::models::QuantizedModel;
use paw_candle::models::gpt2::Gpt2Model;
use paw_candle::models::qwen3::Qwen3Model;
use paw_core::cache::known_models;
use paw_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // ── 1. Download a .paw bundle ──────────────────────────────────────
    println!("[1/5] Resolving slug and downloading .paw bundle...");
    let program_id = client.resolve_slug("email-triage").await?;
    let dir = client.download_paw(&program_id).await?;
    let bundle = PawBundle::load_from_dir(&dir)?;

    println!("  program_id: {program_id}");
    println!(
        "  spec: {}",
        &bundle.meta.spec[..bundle.meta.spec.len().min(80)]
    );
    println!("  interpreter: {}", bundle.interpreter_model());
    println!(
        "  adapter: {} KB",
        std::fs::metadata(&bundle.adapter_path)
            .map(|m| m.len() / 1024)
            .unwrap_or(0)
    );
    println!(
        "  template: {} chars prefix, {} chars suffix",
        bundle.split_template().0.len(),
        bundle.split_template().1.len()
    );

    // ── 2. Load LoRA adapter ────────────────────────────────────────────
    println!("\n[2/5] Loading LoRA adapter...");
    let device = Device::Cpu;
    let lora = GgufLoraAdapter::from_gguf_file(&bundle.adapter_path, &device)?;
    println!("  parsed {} LoRA pairs", lora.len());

    let qkv_count = lora.layers.keys().filter(|k| k.contains("attn_q")).count();
    let ffn_count = lora.layers.keys().filter(|k| k.contains("ffn_up")).count();
    println!("  attention targets: {qkv_count} layers");
    println!("  MLP targets:       {ffn_count} layers");

    // ── 3. Ensure base model GGUF is cached (uses paw cache, then hf-hub) ─
    let interpreter = bundle.interpreter_model();
    let (repo, file) = match known_models::interpreter_to_gguf(interpreter) {
        Some((r, f)) => (r, f),
        None => {
            return Err(Error::UnsupportedModel(interpreter.to_string()));
        }
    };
    let gguf_path = config.base_models_dir().join(file);
    if !gguf_path.exists() {
        println!("\n[3/5] Downloading GGUF ({repo}/{file})...");
        let hf = HFClient::new().expect("hf-hub client");
        let cached = hf
            .model(repo, "")
            .download_file()
            .filename(file)
            .send()
            .await
            .expect("download base model");
        if let Some(p) = gguf_path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        std::fs::copy(&cached, &gguf_path).expect("copy to paw cache");
        println!("  cached to: {}", gguf_path.display());
    } else {
        println!(
            "\n[3/5] GGUF base model already cached: {}",
            gguf_path.display()
        );
    }

    // ── 4. Load base model and attach LoRA ──────────────────────────────
    println!("\n[4/5] Loading base model and attaching LoRA...");
    let lower = interpreter.to_lowercase();
    let mut model: Box<dyn QuantizedModel> = if lower.contains("qwen") {
        Box::new(Qwen3Model::from_gguf(&gguf_path, &device).expect("load Qwen3 model"))
    } else if lower.contains("gpt2") {
        Box::new(Gpt2Model::from_gguf(&gguf_path, &device).expect("load GPT-2 model"))
    } else {
        return Err(Error::UnsupportedModel(interpreter.to_string()));
    };
    let matched = model.set_lora(&lora);
    println!("  matched {matched}/{} layers", model.num_layers());

    // ── 5. Run forward pass ────────────────────────────────────────────
    println!("\n[5/5] Running forward pass...");
    let input = Tensor::new(&[100u32, 200, 300, 400, 500], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let logits = model.forward(&input, 0).expect("forward pass");
    let dims = logits.dims().to_vec();
    println!("  output shape: {:?}", dims);
    assert_eq!(dims.len(), 3, "expected [1, seq_len, vocab_size]");
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 5);

    let last = logits
        .squeeze(0)
        .unwrap()
        .get(4)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let finite = last.iter().filter(|v| v.is_finite()).count();
    println!("  last-token logits: {finite}/{} finite", last.len());
    assert!(finite > last.len() / 2, "most logits should be finite");

    let mut indexed: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!(
        "  top-3 predictions: id={}, id={}, id={}",
        indexed[0].0, indexed[1].0, indexed[2].0
    );

    println!("\n✓ Full pipeline verified: download → load → LoRA → forward");
    Ok(())
}
