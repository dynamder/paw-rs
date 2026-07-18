//! Compare Qwen3 logits: candle (Rust) vs reference from Python/llama.cpp.
//!
//! Usage:
//!   1. First run python to generate reference logits:
//!      python scripts/generate_reference_logits.py
//!   2. Then run the comparison:
//!      cargo run --release --example logit_compare

use candle_core::{Device, Tensor};
use paw_candle::models::qwen3::Qwen3Model;
use paw_candle::models::QuantizedModel;
use paw_core::prelude::*;

/// Compute statistics comparing two logit vectors.
fn compare_logits(label_a: &str, label_b: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sum_abs_diff = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut sum_ab = 0.0f64;

    for i in 0..n {
        let da = a[i] as f64;
        let db = b[i] as f64;
        let diff = (da - db).abs();
        sum_abs_diff += diff;
        sum_a += da;
        sum_b += db;
        sum_aa += da * da;
        sum_bb += db * db;
        sum_ab += da * db;
    }

    let mad = sum_abs_diff / n as f64;
    let cos_sim = sum_ab / ((sum_aa.sqrt()) * (sum_bb.sqrt()));
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;

    eprintln!("═══ Logit Comparison: {label_a} vs {label_b} ═══");
    eprintln!("  Vector size: {n}");
    eprintln!("  Mean A: {mean_a:.6}  Mean B: {mean_b:.6}");
    eprintln!("  Mean absolute diff: {mad:.6}");
    eprintln!("  Cosine similarity: {cos_sim:.6}");
}

fn print_top_k(logits: &[f32], label: &str, k: usize) {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!("\n{label} — Top {k} tokens:");
    for &idx in indices.iter().take(k) {
        eprintln!("  [{idx:>6}] score={:.4}", logits[idx]);
    }
}

fn print_specific(logits: &[f32], label: &str, token_ids: &[u32]) {
    eprintln!("\n{label} — Specific tokens:");
    for &id in token_ids {
        let score = logits
            .get(id as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        eprintln!("  [{id:>6}] score={:.4}", score);
    }
}

fn main() -> Result<()> {
    let config = PawConfig::from_env();
    let gguf_path = config.base_models_dir().join("qwen3-0.6b-q6_k.gguf");
    let ref_path = config.cache_dir().join("reference_logits.bin");

    if !ref_path.exists() {
        return Err(Error::Other(format!(
            "Reference logits not found at {}. Run Python first:\n  python scripts/generate_reference_logits.py",
            ref_path.display()
        )));
    }

    eprintln!("═══ Qwen3 Logit Comparison ═══\n");

    // ── Minimal test input ────────────────────────────────────
    let test_tokens: Vec<u32> = vec![100, 200, 300, 400, 500];

    // ── 1. candle forward pass ─────────────────────────────────
    eprintln!("[1] Loading model with candle...");
    let device = Device::Cpu;
    let mut model = Qwen3Model::from_gguf(&gguf_path, &device)
        .map_err(|e| Error::Other(format!("candle model load: {e}")))?;

    let input_t = Tensor::from_slice(&test_tokens, test_tokens.len(), &device)
        .map_err(|e| Error::Other(format!("tensor: {e}")))?
        .unsqueeze(0)
        .map_err(|e| Error::Other(format!("unsqueeze: {e}")))?;

    let logits_t = model
        .forward(&input_t, 0)
        .map_err(|e| Error::Other(format!("candle forward: {e}")))?;
    let last_pos = logits_t
        .dim(1)
        .map_err(|e| Error::Other(format!("dim: {e}")))?
        - 1;
    let last_t = logits_t
        .narrow(1, last_pos, 1)
        .map_err(|e| Error::Other(format!("narrow: {e}")))?
        .squeeze(0)
        .map_err(|e| Error::Other(format!("squeeze: {e}")))?
        .squeeze(0)
        .map_err(|e| Error::Other(format!("squeeze: {e}")))?;
    let candle_logits: Vec<f32> = last_t
        .to_vec1::<f32>()
        .map_err(|e| Error::Other(format!("to_vec1: {e}")))?;

    eprintln!("  candle logits: {} elements\n", candle_logits.len());

    // ── 2. Read reference logits from Python ──────────────────
    let ref_bytes = std::fs::read(&ref_path).map_err(|e| Error::Other(format!("read ref: {e}")))?;
    let ref_f32: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // First u32 is vocab_size, rest are logits
    let vocab_size = ref_f32[0] as usize;
    let ref_logits = &ref_f32[1..1 + vocab_size];

    eprintln!(
        "  reference logits: {} elements (vocab={})\n",
        ref_logits.len(),
        vocab_size
    );

    // ── 3. Compare ────────────────────────────────────────────
    compare_logits(
        "candle",
        "reference (llama.cpp)",
        &candle_logits,
        ref_logits,
    );

    // ── 4. Top-k tokens ───────────────────────────────────────
    print_top_k(&candle_logits, "candle", 8);
    print_top_k(ref_logits, "reference (llama.cpp)", 8);

    // ── 5. Specific tokens ────────────────────────────────────
    let key_tokens = [151645, 318, 14636, 11489, 151643, 151644];
    print_specific(&candle_logits, "candle", &key_tokens);
    print_specific(ref_logits, "reference (llama.cpp)", &key_tokens);

    // ── 6. Argmax comparison ──────────────────────────────────
    let argmax_candle = candle_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let argmax_ref = ref_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    eprintln!("\n═══ Argmax ═══");
    eprintln!("  candle: {argmax_candle}");
    eprintln!("  reference: {argmax_ref}");
    eprintln!(
        "  MATCH: {}",
        if argmax_candle == argmax_ref {
            "✅ YES"
        } else {
            "❌ NO"
        }
    );

    if argmax_candle != argmax_ref {
        eprintln!(
            "\n  candle score at reference's argmax: {:.4}",
            candle_logits[argmax_ref]
        );
        eprintln!(
            "  reference score at candle's argmax: {:.4}",
            ref_logits[argmax_candle]
        );
    }

    // ── 7. Largest divergences ────────────────────────────────
    let mut diffs: Vec<(usize, f32)> = candle_logits
        .iter()
        .zip(ref_logits.iter())
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .collect();
    diffs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    eprintln!("\n═══ Top 10 Largest Logit Differences ═══");
    for (i, (idx, diff)) in diffs.iter().take(10).enumerate() {
        eprintln!(
            "  {i:>2}. [{idx:>6}] candle={:.4} ref={:.4} diff={:.4}",
            candle_logits[*idx], ref_logits[*idx], diff
        );
    }

    Ok(())
}
