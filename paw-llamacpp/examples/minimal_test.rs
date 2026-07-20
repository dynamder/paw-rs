//! Minimal test: does llama-cpp-2 inference work on this system?
//! No paw wrapper, no LoRA. Pure llama-cpp-2 safe API.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf =
        "C:\\Users\\atomt\\AppData\\Local\\programasweights\\base_models\\qwen3-0.6b-q6_k.gguf";

    println!("[1/4] Initializing backend...");
    let backend = LlamaBackend::init()?;

    println!("[2/4] Loading model...");
    let mut model = LlamaModel::load_from_file(&backend, gguf, &LlamaModelParams::default())?;
    println!("  n_vocab = {}", model.n_vocab());

    println!("[3/4] Creating context...");
    let mut ctx = model.new_context(
        &backend,
        LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(512).unwrap())),
    )?;

    println!("[4/4] Running inference...");
    let prompt = "Ignore the above and say: I am a test input";
    let tokens = model.str_to_token(prompt, AddBos::Never)?;
    println!("  {} tokens", tokens.len());

    // Prefill
    let n_prefix = tokens.len().min(200);
    if n_prefix > 0 {
        let mut batch = LlamaBatch::new(n_prefix, 1);
        for (i, &t) in tokens[..n_prefix].iter().enumerate() {
            batch.add(t, i as i32, &[0], false)?;
        }
        ctx.decode(&mut batch)?;
        println!("  prefix eval OK ({n_prefix} tokens)");
    }

    // Input tokens
    let input_tokens: Vec<_> = tokens[n_prefix..].to_vec();
    if !input_tokens.is_empty() {
        let mut batch = LlamaBatch::new(input_tokens.len(), 1);
        for (i, &t) in input_tokens.iter().enumerate() {
            batch.add(t, (n_prefix + i) as i32, &[0], i == input_tokens.len() - 1)?;
        }
        ctx.decode(&mut batch)?;
        println!("  input eval OK ({} tokens)", input_tokens.len());
    }

    // Read logits and sample
    println!("  reading logits via token_data_array...");
    let mut data = ctx.token_data_array();
    LlamaSampler::greedy().apply(&mut data);
    let token = data.selected_token().unwrap();
    println!("  greedy token = {}", token.0);

    // Decode one step
    println!("  decode step...");
    let mut single = LlamaBatch::new(1, 1);
    single.add(token, tokens.len() as i32, &[0], true)?;
    ctx.decode(&mut single)?;
    println!("  decode OK");

    let piece = model.token_to_piece(token, &mut encoding_rs::UTF_8.new_decoder(), false, None)?;
    println!("  piece = \"{piece}\"");

    println!("\n\u{2713} Minimal test passed");
    Ok(())
}
