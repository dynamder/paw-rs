/// Example: download a .paw bundle, inspect, then demonstrate binary `.paw` v2 format roundtrip.
///
/// The `.paw` program bundle is a ZIP archive (adapter.gguf + prompt_template.txt + meta.json).
/// The binary `.paw` v2 format (PAW\x02) is used for prefix KV cache persistence.
/// This example demonstrates both.
///
/// Works on public programs without an API key.
/// Usage:
///   cargo run --example download_and_save
use std::collections::HashMap;

use paw_core::prelude::*;
use safetensors::tensor::Dtype;

fn sample_tensor(name: &str, rows: usize, cols: usize) -> (String, TensorData) {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    (
        name.to_string(),
        TensorData {
            dtype: Dtype::F32,
            shape: vec![rows, cols],
            data: raw,
        },
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // ── Resolve slug and download (no API key needed for public progs) ─
    let program_id = client.resolve_slug("email-triage").await?;
    let dir = client.download_paw(&program_id).await?;

    // ── Read meta.json and show spec ───────────────────────────────────
    let bundle = PawBundle::load_from_dir(&dir)?;
    println!("spec: {}", bundle.meta.spec);
    println!("interpreter: {}", bundle.interpreter_model());
    println!("template: {} chars", bundle.prompt_template.len());
    println!(
        "adapter: {} KB",
        std::fs::metadata(&bundle.adapter_path)
            .map(|m| m.len() / 1024)
            .unwrap_or(0)
    );

    // ── Binary .paw v2 format roundtrip ────────────────────────────────
    // Create sample tensor data (simulating KV cache layers)
    let mut tensors = HashMap::new();
    for (k, v) in [
        sample_tensor("layer_0_key", 2, 4),
        sample_tensor("layer_0_value", 2, 4),
        sample_tensor("lora_blk.0.attn_q.lora_a", 4, 2),
    ] {
        tensors.insert(k, v);
    }

    let mut meta = PawFileMeta::default();
    meta.num_layers = 1;
    meta.has_lora = true;
    meta.interpreter_model = bundle.interpreter_model().to_string();
    meta.spec = bundle.meta.spec.clone();
    meta.lora_config = Some(LoRAConfig {
        rank: 4,
        alpha: 16.0,
        target_modules: vec!["attn_q".into()],
    });

    let tmp = std::env::temp_dir().join("repacked.paw");
    println!(
        "\nwriting binary .paw v2 -> {} ({} tensors)",
        tmp.display(),
        tensors.len()
    );
    PawFormatWriter::save(&tmp, tensors, &meta)?;

    let (reloaded, meta2) = PawFormatReader::load(&tmp)?;
    println!(
        "   verified: {} tensors, version={}, interpreter={}",
        reloaded.len(),
        meta2.format_version,
        meta2.interpreter_model,
    );

    // Verify specific tensor
    assert!(reloaded.contains_key("layer_0_key"));
    assert!(reloaded.contains_key("lora_blk.0.attn_q.lora_a"));
    println!("   specific tensors present ✓");

    std::fs::remove_file(&tmp).ok();
    println!("\n✓ Full pipeline verified: download → bundle → binary format roundtrip");
    Ok(())
}
