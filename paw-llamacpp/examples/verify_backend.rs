use paw_core::PawConfig;
use paw_llamacpp::{PawFnLoader, PawLlamaCppConfig, PawRuntimeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prog_id = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("fccdea9da515e3f20dd6");

    let paw_config = PawConfig::from_env();
    let program_dir = paw_config.programs_dir().join(prog_id);
    assert!(
        program_dir.exists(),
        "Program dir not found: {}",
        program_dir.display()
    );

    println!("=== Llamacpp Backend Correctness Verification ===");
    println!("Program: {}\n", program_dir.display());

    let test_inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ];
    let opts = PawRuntimeOptions {
        max_tokens: Some(15),
        temperature: 0.0,
        top_p: 1.0,
    };

    // ── Test 1: Serial inference (max_model_copies=1) ──
    println!("--- Test 1: Single instance, serial (max_model_copies=1) ---");
    let config1 = PawLlamaCppConfig::builder()
        .core(paw_config.clone())
        .max_model_copies(1)
        .build();
    let f1 = PawFnLoader::new(&program_dir).config(config1).load()?;

    for (i, input) in test_inputs.iter().enumerate() {
        let out = f1.run(input, &opts)?;
        println!("  Input{}: \"{}\"", i + 1, input);
        println!("  Output: \"{}\"", out);
        assert!(!out.is_empty(), "Output should not be empty");
        assert!(
            out.chars().all(|c| !c.is_control() || c == '\n'),
            "Output should be valid text"
        );
    }
    println!("  PASS\n");

    // ── Test 2: Same-function repeatability ──
    println!("--- Test 2: Repeatability (same function, same input, 3 runs) ---");
    let repeated: Vec<_> = (0..3)
        .map(|_| f1.run(test_inputs[0], &opts).unwrap())
        .collect();
    for (i, r) in repeated.iter().enumerate() {
        println!("  Run {i}: \"{}\"", r);
    }
    assert!(
        repeated.iter().all(|r| r == &repeated[0]),
        "Greedy decoding should be deterministic"
    );
    println!("  PASS (deterministic)\n");

    // ── Test 3: Concurrent with shared pool (max_model_copies=4) ──
    println!("--- Test 3: 4 concurrent instances, max_model_copies=4 ---");
    let config4 = PawLlamaCppConfig::builder()
        .core(paw_config.clone())
        .max_model_copies(4)
        .build();

    let mut funcs: Vec<_> = (0..4)
        .map(|_| {
            PawFnLoader::new(&program_dir)
                .config(config4.clone())
                .load()
                .unwrap()
        })
        .collect();

    let results: Vec<String> = funcs
        .iter_mut()
        .map(|f| f.run(test_inputs[0], &opts).unwrap())
        .collect();

    for (i, r) in results.iter().enumerate() {
        println!("  Instance {i}: \"{}\"", r);
        assert!(!r.is_empty(), "Instance {i} output should not be empty");
    }
    assert_eq!(
        &results[0], &results[1],
        "All instances with same adapter should produce same output"
    );
    assert_eq!(
        &results[0], &results[2],
        "Greedy sampling is deterministic across contexts"
    );
    assert_eq!(&results[0], &results[3], "");
    println!("  PASS (all instances consistent)\n");

    // ── Test 4: Serial vs concurrent consistency ──
    println!("--- Test 4: Serial output matches concurrent output ---");
    let serial_out = f1.run(test_inputs[1], &opts)?;
    let conc_out = funcs[0].run(test_inputs[1], &opts)?;
    println!("  Serial:     \"{}\"", serial_out);
    println!("  Concurrent: \"{}\"", conc_out);
    assert_eq!(
        serial_out, conc_out,
        "Serial and concurrent should produce identical output"
    );
    println!("  PASS\n");

    // ── Test 5: Threaded parallel execution ──
    println!("--- Test 5: Threaded parallel (4 threads, max_model_copies=4) ---");
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let ready = Arc::new(AtomicBool::new(false));
    let test_input = test_inputs[2].to_string();
    let t_opts = opts.clone();

    let handles: Vec<_> = funcs
        .drain(..)
        .map(|f| {
            let r = Arc::clone(&ready);
            let inp = test_input.clone();
            let topts = t_opts.clone();
            std::thread::spawn(move || {
                while !r.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                f.run(&inp, &topts).unwrap()
            })
        })
        .collect();

    std::thread::sleep(std::time::Duration::from_millis(50));
    ready.store(true, Ordering::Release);

    let thread_results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for (i, r) in thread_results.iter().enumerate() {
        println!("  Thread {i}: \"{}\"", r);
        assert!(!r.is_empty());
    }
    let all_same = thread_results.windows(2).all(|w| w[0] == w[1]);
    assert!(all_same, "All threads should produce same greedy output");
    println!("  PASS (all threads consistent)\n");

    println!("=== All 5 tests PASSED ===");
    Ok(())
}
