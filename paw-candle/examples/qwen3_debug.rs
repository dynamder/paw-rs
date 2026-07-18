//! Debug Qwen3: compare model output with vs without LoRA.
use std::path::PathBuf;
use candle_core::{Device, Tensor};
use hf_hub::HFClient;
use paw_candle::models::qwen3::Qwen3Model;
use paw_candle::lora::GgufLoraAdapter;
use paw_candle::models::QuantizedModel;
use paw_core::prelude::*;

const QWEN3_FILE: &str = "qwen3-0.6b-q6_k.gguf";
const TOKENIZER_FILE: &str = "tokenizer.json";

fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/programasweights")
}

async fn ensure_cached<T: AsRef<std::path::Path>>(
    hf: &HFClient, repo: &str, model: &str, file: &str, dst: T,
) -> Result<()> {
    let dst = dst.as_ref();
    if dst.exists() { return Ok(()); }
    let cached = hf.model(repo, model)
        .download_file().filename(file).send().await
        .map_err(|e| Error::Other(format!("hf-hub: {e}")))?;
    if let Some(p) = dst.parent() { std::fs::create_dir_all(p)?; }
    std::fs::copy(&cached, dst)?;
    Ok(())
}

fn top_k(t: &Tensor, k: usize) -> Vec<(u32, f32)> {
    let vals = t.to_vec1::<f32>().unwrap();
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap());
    idx[..k.min(idx.len())].iter().map(|&i| (i as u32, vals[i])).collect()
}

fn compare_logits(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    println!("[1] Downloading email-triage bundle...");
    let program_id = client.resolve_slug("email-triage").await?;
    let dir = client.download_paw(&program_id).await?;

    let hf = HFClient::new().map_err(|e| Error::Other(format!("hf-hub init: {e}")))?;
    let gguf_path = config.base_models_dir().join(QWEN3_FILE);
    ensure_cached(&hf, "programasweights", "Qwen3-0.6B-GGUF-Q6_K", QWEN3_FILE, &gguf_path).await?;
    let tok_path = dir.join(TOKENIZER_FILE);
    ensure_cached(&hf, "Qwen", "Qwen3-0.6B", TOKENIZER_FILE, &tok_path).await?;

    let device = Device::Cpu;

    // Load tokenizer
    let raw_tok = tokenizers::Tokenizer::from_file(tok_path).expect("tokenizer");

    // Tokenize test input using the PAW Tokenizer wrapper (handles special tokens)
    let tok = paw_candle::runtime::PawFunction::dummy_tokenizer_from(raw_tok);

    // Read the actual prompt template for email-triage
    let template = std::fs::read_to_string(dir.join("prompt_template.txt")).expect("read template");
    let placeholder = "{INPUT_PLACEHOLDER}";
    let prefix_text: String;
    let suffix_text: String;
    if let Some(pos) = template.find(placeholder) {
        prefix_text = template[..pos].to_string();
        suffix_text = template[pos + placeholder.len()..].to_string();
    } else {
        prefix_text = template.clone();
        suffix_text = String::new();
    }

    let input_text = "Urgent: server down!";
    let full_input = format!("{}{}", input_text, suffix_text);
    let prefix_ids = tok.encode(&prefix_text).unwrap();
    let input_ids = tok.encode(&full_input).unwrap();
    let mut ids = prefix_ids.clone();
    ids.extend(&input_ids);
    eprintln!("Test input via PAW Tokenizer: {} tokens", ids.len());
    eprintln!("First 10 token IDs: {:?}", &ids[..10.min(ids.len())]);

    // Load model WITHOUT lora, run forward, capture logits
    println!("[2] Loading model (no LoRA)...");
    let mut model = Qwen3Model::from_gguf(&gguf_path, &device).expect("load model");
    let input_t = Tensor::new(&ids[..], &device).unwrap().unsqueeze(0).unwrap();
    let logits_no_lora = model.forward(&input_t, 0).expect("forward no lora");
    let last_no_lora = logits_no_lora.squeeze(0).unwrap()
        .get(logits_no_lora.dim(1).unwrap() - 1).unwrap();
    let top5_no = top_k(&last_no_lora, 5);
    eprintln!("\nWITHOUT LoRA - Top 5 next tokens:");
    for (id, score) in &top5_no {
        let t = tok.id_to_token(*id).unwrap_or_else(|| "<unk>".to_string());
        eprintln!("  [{id:>6}] {t:30} score={score:.4}");
    }

    // Now load a FRESH model WITH LoRA
    println!("\n[3] Loading model (WITH LoRA)...");
    let mut model2 = Qwen3Model::from_gguf(&gguf_path, &device).expect("load model 2");
    let adapter_path = dir.join("adapter.gguf");
    let lora = GgufLoraAdapter::from_gguf_file(&adapter_path, &device).expect("load lora");
    let matched = model2.set_lora(&lora);
    eprintln!("LoRA matched {matched} layers");

    let logits_with_lora = model2.forward(&input_t, 0).expect("forward with lora");
    let last_with_lora = logits_with_lora.squeeze(0).unwrap()
        .get(logits_with_lora.dim(1).unwrap() - 1).unwrap();
    let top5_yes = top_k(&last_with_lora, 5);
    eprintln!("\nWITH LoRA - Top 5 next tokens:");
    for (id, score) in &top5_yes {
        let t = tok.id_to_token(*id).unwrap_or_else(|| "<unk>".to_string());
        eprintln!("  [{id:>6}] {t:30} score={score:.4}");
    }

    // Compare the logits
    let v_no = last_no_lora.to_vec1::<f32>().unwrap();
    let v_yes = last_with_lora.to_vec1::<f32>().unwrap();
    let diff = compare_logits(&v_no, &v_yes);
    eprintln!("\nMean absolute logit difference: {diff:.6}");
    eprintln!("  (should be close to 0 if LoRA has no effect, >> 0 if LoRA changes output)");

    // Find which tokens changed the most
    let mut deltas: Vec<(u32, f32)> = v_no.iter().zip(v_yes.iter())
        .enumerate().map(|(i, (a, b))| (i as u32, (a - b).abs()))
        .collect();
    deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("\nTop 10 tokens by LoRA impact:");
    for (id, delta) in &deltas[..10] {
        let t = tok.id_to_token(*id).unwrap_or_else(|| "<unk>".to_string());
        eprintln!("  [{id:>6}] {t:30} delta={delta:.4} (without={:.4}, with={:.4})", v_no[*id as usize], v_yes[*id as usize]);
    }

    // Generate with LoRA
    eprintln!("\n[4] Generating text WITH LoRA...");
    let mut gen_ids = ids.clone();
    for step in 0..15 {
        let inp = Tensor::new(&gen_ids[..], &device).unwrap().unsqueeze(0).unwrap();
        let log = model2.forward(&inp, 0).expect("gen forward");
        let last = log.squeeze(0).unwrap().get(log.dim(1).unwrap() - 1).unwrap();
        let next = last.argmax(0).unwrap().to_scalar::<u32>().unwrap();
        let text = tok.decode(&[next], true).unwrap_or_default();
        eprintln!("  step {step:>2}: token {next:>6} -> {text:?}");
        if next == 151645 || next == 151643 {
            break;
        }
        gen_ids.push(next);
    }

    Ok(())
}
