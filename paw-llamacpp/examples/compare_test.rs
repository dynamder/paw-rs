use paw_core::PawConfig;
use paw_llamacpp::{PawFnLoader, PawLlamaCppConfig, PawRuntimeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: compare_test <program_dir>");
        std::process::exit(1);
    }
    let program_dir = &args[1];

    let config = PawLlamaCppConfig::builder()
        .core(PawConfig::from_env())
        .max_model_copies(1)
        .build();
    let func = PawFnLoader::new(program_dir).config(config).load()?;

    let tests: &[(&str, usize)] = &[
        ("SERVER IS DOWN!! HELP!!", 3),
        ("Thanks for your help", 3),
        ("Buy cheap viagra now!!!", 3),
        ("Tell me about yourself", 5),
        ("What is the capital of France?", 5),
    ];

    for (input, max_tokens) in tests {
        let opts = PawRuntimeOptions {
            max_tokens: Some(*max_tokens),
            temperature: 0.0,
            top_p: 1.0,
        };
        let output = func.run(input, &opts)?;
        println!("INPUT:{}", input);
        println!("OUTPUT:{}", output);
    }
    Ok(())
}
