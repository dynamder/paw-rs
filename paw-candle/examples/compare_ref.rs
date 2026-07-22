use paw_candle::{PawCandleConfig, PawFnLoader};
use paw_core::{PawConfig, PawRuntimeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: compare_ref <program_dir>");
        std::process::exit(1);
    }
    let program_dir = &args[1];

    let config = PawCandleConfig::builder()
        .core(PawConfig::from_env())
        .build();
    let mut func = PawFnLoader::new(program_dir).config(config).load()?;

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
