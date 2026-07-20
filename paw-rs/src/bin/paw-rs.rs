//! PAW CLI — command-line interface for ProgramAsWeights.
//!
//! Usage:
//!     paw-rs compile --spec "..." [--compiler] [--slug] [--private] [--json]
//!     paw-rs run --program <id> --input "..." [--max-tokens] [--temperature] [--verbose] [--json]
//!     paw-rs login [key]
//!     paw-rs rename <program> <new_slug> [--json]
//!     paw-rs info <program> [--json]
//!
//! All commands support --api-url, --api-key, and --json for structured output.

use std::process;

use clap::{Parser, Subcommand};
use paw_core::{CompileRequest, PawClient, PawConfig};

#[derive(Debug, Parser)]
#[command(name = "paw-rs", about = "ProgramAsWeights: compile and run neural programs")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(long, global = true, help = "PAW server URL")]
    api_url: Option<String>,

    #[arg(long, global = true, help = "API key")]
    api_key: Option<String>,

    #[arg(long, global = true, help = "Output structured JSON (agent-friendly)")]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compile a spec on the server
    Compile {
        #[arg(long, help = "Natural language specification")]
        spec: String,

        #[arg(long, help = "Compiler model (omit to use server default)")]
        compiler: Option<String>,

        #[arg(long, help = "URL-safe handle (e.g. message-classifier)")]
        slug: Option<String>,

        #[arg(long, help = "Make program private")]
        private: bool,
    },

    /// Run a program locally
    Run {
        #[arg(long, help = "Program ID or slug")]
        program: String,

        #[arg(long, help = "Input text")]
        input: String,

        #[arg(long, help = "Max tokens to generate")]
        max_tokens: Option<usize>,

        #[arg(long, help = "Temperature (0.0 = greedy)")]
        temperature: Option<f64>,

        #[arg(long, help = "Verbose output")]
        verbose: bool,
    },

    /// Save API key for authentication
    Login {
        #[arg(help = "API key (paw_sk_...). Omit to open browser")]
        key: Option<String>,
    },

    /// Set or change a program's slug
    Rename {
        #[arg(help = "Program ID or current slug")]
        program: String,

        #[arg(help = "New slug (e.g. message-classifier) or empty string to remove")]
        new_slug: String,
    },

    /// Show program info
    Info {
        #[arg(help = "Program ID or slug")]
        program: String,
    },
}

fn apply_auth_overrides(api_url: Option<&str>, api_key: Option<&str>, verbose: bool) {
    // SAFETY: Setting env vars before any other code runs is safe in single-threaded startup.
    if let Some(url) = api_url {
        unsafe { std::env::set_var("PAW_API_URL", url); }
    }
    if let Some(key) = api_key {
        unsafe { std::env::set_var("PAW_API_KEY", key); }
    }
    if verbose {
        unsafe { std::env::set_var("PAW_VERBOSE", "1"); }
    }
}

fn init_tracing(verbose: bool) {
    use tracing_subscriber::prelude::*;
    let filter = if verbose {
        "debug"
    } else {
        "info"
    };
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(
            tracing_subscriber::EnvFilter::try_new(filter)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let verbose = match &cli.command {
        Command::Run { verbose, .. } => *verbose,
        _ => false,
    };

    init_tracing(verbose);
    apply_auth_overrides(cli.api_url.as_deref(), cli.api_key.as_deref(), verbose);

    let json = cli.json;

    let result = match &cli.command {
        Command::Compile { spec, compiler, slug, private } => {
            cmd_compile(spec, compiler.as_deref(), slug.as_deref(), *private, json).await
        }
        Command::Run { program, input, max_tokens, temperature, .. } => {
            cmd_run(program, input, *max_tokens, *temperature, json).await
        }
        Command::Login { key } => {
            cmd_login(key.as_deref())
        }
        Command::Rename { program, new_slug } => {
            cmd_rename(program, new_slug, json).await
        }
        Command::Info { program } => {
            cmd_info(program, json).await
        }
    };

    match result {
        Ok(code) => process::exit(code),
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}

// ── Commands ────────────────────────────────────────────────────────

async fn cmd_compile(spec: &str, compiler: Option<&str>, slug: Option<&str>, private: bool, json: bool) -> Result<i32, paw_core::Error> {
    if !json {
        let preview: String = spec.chars().take(80).collect();
        println!("Compiling: {preview}...");
    }

    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    let mut req = CompileRequest::builder()
        .spec(spec.to_string())
        .public(!private);
    if let Some(c) = compiler {
        req = req.compiler(c);
    }
    if let Some(s) = slug {
        req = req.slug(s);
    }
    let program = client.compile(req.build()?).await?;

    if json {
        let output = serde_json::json!({
            "program_id": program.id,
            "slug": program.slug,
            "status": program.status,
            "error": program.error,
            "timings": program.timings,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return Ok(if program.error.is_some() { 1 } else { 0 });
    }

    if let Some(ref e) = program.error {
        println!("Error: {e}");
        return Ok(1);
    }

    println!("Program ID: {}", program.id);
    if let Some(ref s) = program.slug {
        println!("Slug: {s}");
    }
    println!("Status: {}", program.status);
    if let Some(ref timings) = program.timings {
        let total = timings.get("total_ms").copied().unwrap_or(0.0);
        println!("Total time: {total:.0}ms");
    }
    let ref_ = program.slug.as_deref().unwrap_or(&program.id);
    println!("\nTo run locally:");
    println!("  paw-rs run --program \"{ref_}\" --input \"your input here\"");
    Ok(0)
}

async fn cmd_run(program_ref: &str, input: &str, max_tokens: Option<usize>, temperature: Option<f64>, json: bool) -> Result<i32, paw_core::Error> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // 1. Resolve slug → program ID
    let program_id = match client.resolve_slug(program_ref).await {
        Ok(id) => id,
        Err(_) => program_ref.to_string(), // not a slug, treat as raw program ID
    };

    // 2. Download/refresh program bundle
    let dir = client.download_paw(&program_id).await?;

    // 3. Ensure base model GGUF + tokenizer are cached locally
    let bundle = paw_core::PawBundle::load_from_dir(&dir)?;
    let interpreter = bundle.interpreter_model();
    paw_candle::ensure_assets(&config, &dir, interpreter).await?;

    // 4. Load model via PawFnLoader
    let candle_config = paw_candle::PawCandleConfig::builder()
        .core(config)
        .build();
    let inner = paw_candle::PawFnLoader::new(dir)
        .config(candle_config)
        .load()?;
    let mut func = paw_rs::PawFn::<paw_candle::Dynamic>::from_inner(inner);

    let opts = paw_candle::PawRuntimeOptions {
        max_tokens,
        temperature: temperature.unwrap_or(0.0),
        ..Default::default()
    };

    let result = func.run_with(input, &opts)?;

    if json {
        let output = serde_json::json!({
            "program": program_ref,
            "input": input,
            "output": result,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("{result}");
    }
    Ok(0)
}

fn cmd_login(key: Option<&str>) -> Result<i32, paw_core::Error> {
    let key = match key {
        Some(k) => k.to_string(),
        None => {
            let settings_url = paw_core::config::get_api_url().trim_end_matches('/').to_string() + "/settings";
            println!("Generate an API key at {settings_url}");
            if webbrowser::open(&settings_url).is_ok() {
                println!("Opened browser to {settings_url}");
            }

            let key = rpassword::prompt_password("Paste your API key: ")
                .map_err(|e| paw_core::Error::Io(e))?;
            let key = key.trim().to_string();

            if key.is_empty() {
                println!("No key provided. Aborted.");
                return Ok(0);
            }

            if !key.starts_with("paw_sk_") {
                println!("Warning: key doesn't start with 'paw_sk_'. Saving anyway.");
            }

            key
        }
    };

    paw_core::login(&key).map_err(|e| paw_core::Error::Io(e))?;
    println!("API key saved.");
    Ok(0)
}

async fn cmd_rename(program_ref: &str, new_slug: &str, json: bool) -> Result<i32, paw_core::Error> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    let data = client.rename_slug(program_ref, new_slug).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&data).unwrap());
    } else {
        if let Some(slug) = data.get("slug").and_then(|v| v.as_str()) {
            if slug.is_empty() {
                println!("Slug removed.");
            } else {
                println!("Renamed to: {slug}");
            }
        } else {
            println!("Slug removed.");
        }
    }
    Ok(0)
}

fn extra_str(meta: &paw_core::BundleMeta, key: &str, default: &str) -> String {
    meta.extra
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

async fn cmd_info(program_ref: &str, json: bool) -> Result<i32, paw_core::Error> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    let meta = match client.get_program_meta(program_ref).await {
        Ok(m) => m,
        Err(_) => {
            if json {
                let output = serde_json::json!({"error": "not_found", "program": program_ref});
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            } else {
                println!("Program {program_ref} not found.");
            }
            return Ok(1);
        }
    };

    if json {
        let value = serde_json::to_value(&meta).unwrap();
        println!("{}", serde_json::to_string_pretty(&value).unwrap());
    } else {
        let program_id = extra_str(&meta, "id", "N/A");
        let spec = &meta.spec;
        let spec_preview: String = spec.chars().take(100).collect();
        let interpreter = extra_str(&meta, "interpreter", "N/A");
        let compiler = extra_str(&meta, "compiler_snapshot", "N/A");
        let aliases: Vec<String> = meta
            .extra
            .get("aliases")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let hf_url = extra_str(&meta, "hf_url", "");

        println!("Program: {program_id}");
        println!("  Spec: {spec_preview}");
        println!("  Interpreter: {interpreter}");
        println!("  Compiler: {compiler}");
        if !aliases.is_empty() {
            println!("  Aliases: {}", aliases.join(", "));
        }
        if !hf_url.is_empty() {
            println!("  HF URL: {hf_url}");
        }
    }
    Ok(0)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn preview(spec: &str) -> String {
        spec.chars().take(80).collect()
    }

    // ── Compile ──────────────────────────────────────────────────────

    #[test]
    fn test_compile_minimal() {
        let cli = Cli::try_parse_from(&["paw-rs", "compile", "--spec", "hello"]).unwrap();
        match &cli.command {
            Command::Compile { spec, compiler, slug, private } => {
                assert_eq!(spec, "hello");
                assert!(compiler.is_none());
                assert!(slug.is_none());
                assert!(!private);
            }
            _ => panic!("expected Compile"),
        }
    }

    #[test]
    fn test_compile_all_flags() {
        let cli = Cli::try_parse_from(&[
            "paw-rs", "compile",
            "--spec", "classify sentiment",
            "--compiler", "paw-4b",
            "--slug", "my-classifier",
            "--private",
        ]).unwrap();
        match &cli.command {
            Command::Compile { spec, compiler, slug, private } => {
                assert_eq!(spec, "classify sentiment");
                assert_eq!(compiler.as_deref(), Some("paw-4b"));
                assert_eq!(slug.as_deref(), Some("my-classifier"));
                assert!(private);
            }
            _ => panic!("expected Compile"),
        }
    }

    #[test]
    fn test_compile_missing_spec_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "compile"]).unwrap_err();
        assert!(err.to_string().contains("spec"), "error: {err}");
    }

    // ── Run ──────────────────────────────────────────────────────────

    #[test]
    fn test_run_minimal() {
        let cli = Cli::try_parse_from(&[
            "paw-rs", "run",
            "--program", "abc123",
            "--input", "test input",
        ]).unwrap();
        match &cli.command {
            Command::Run { program, input, max_tokens, temperature, verbose } => {
                assert_eq!(program, "abc123");
                assert_eq!(input, "test input");
                assert!(max_tokens.is_none());
                assert!(temperature.is_none());
                assert!(!verbose);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn test_run_all_flags() {
        let cli = Cli::try_parse_from(&[
            "paw-rs", "run",
            "--program", "abc123",
            "--input", "hello",
            "--max-tokens", "256",
            "--temperature", "0.5",
            "--verbose",
        ]).unwrap();
        match &cli.command {
            Command::Run { program, input, max_tokens, temperature, verbose } => {
                assert_eq!(program, "abc123");
                assert_eq!(input, "hello");
                assert_eq!(*max_tokens, Some(256));
                assert!((temperature.clone().unwrap() - 0.5).abs() < 1e-9);
                assert!(verbose);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn test_run_missing_program_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "run", "--input", "x"]).unwrap_err();
        assert!(err.to_string().contains("program"), "error: {err}");
    }

    #[test]
    fn test_run_missing_input_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "run", "--program", "x"]).unwrap_err();
        assert!(err.to_string().contains("input"), "error: {err}");
    }

    // ── Login ────────────────────────────────────────────────────────

    #[test]
    fn test_login_with_key() {
        let cli = Cli::try_parse_from(&["paw-rs", "login", "paw_sk_test123"]).unwrap();
        match &cli.command {
            Command::Login { key } => {
                assert_eq!(key.as_deref(), Some("paw_sk_test123"));
            }
            _ => panic!("expected Login"),
        }
    }

    #[test]
    fn test_login_without_key() {
        let cli = Cli::try_parse_from(&["paw-rs", "login"]).unwrap();
        match &cli.command {
            Command::Login { key } => {
                assert!(key.is_none());
            }
            _ => panic!("expected Login"),
        }
    }

    // ── Rename (positional args) ─────────────────────────────────────

    #[test]
    fn test_rename() {
        let cli = Cli::try_parse_from(&["paw-rs", "rename", "abc123", "new-slug"]).unwrap();
        match &cli.command {
            Command::Rename { program, new_slug } => {
                assert_eq!(program, "abc123");
                assert_eq!(new_slug, "new-slug");
            }
            _ => panic!("expected Rename"),
        }
    }

    #[test]
    fn test_rename_empty_slug() {
        let cli = Cli::try_parse_from(&["paw-rs", "rename", "abc123", ""]).unwrap();
        match &cli.command {
            Command::Rename { program, new_slug } => {
                assert_eq!(program, "abc123");
                assert_eq!(new_slug, "");
            }
            _ => panic!("expected Rename"),
        }
    }

    #[test]
    fn test_rename_missing_arg_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "rename", "abc123"]).unwrap_err();
        assert!(err.to_string().contains("requires") || err.to_string().contains("arg"), "error: {err}");
    }

    #[test]
    fn test_rename_no_args_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "rename"]).unwrap_err();
        assert!(err.to_string().contains("program") || err.to_string().contains("arg") || err.to_string().contains("requires"), "error: {err}");
    }

    // ── Info (positional arg) ────────────────────────────────────────

    #[test]
    fn test_info() {
        let cli = Cli::try_parse_from(&["paw-rs", "info", "abc123"]).unwrap();
        match &cli.command {
            Command::Info { program } => {
                assert_eq!(program, "abc123");
            }
            _ => panic!("expected Info"),
        }
    }

    #[test]
    fn test_info_missing_arg_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "info"]).unwrap_err();
        assert!(err.to_string().contains("program") || err.to_string().contains("required"), "error: {err}");
    }

    // ── Global flags ─────────────────────────────────────────────────

    #[test]
    fn test_global_json_with_compile() {
        let cli = Cli::try_parse_from(&["paw-rs", "--json", "compile", "--spec", "x"]).unwrap();
        assert!(cli.json);
        match &cli.command {
            Command::Compile { .. } => {}
            _ => panic!("expected Compile"),
        }
    }

    #[test]
    fn test_global_json_with_run() {
        let cli = Cli::try_parse_from(&[
            "paw-rs", "--json", "run",
            "--program", "x", "--input", "y",
        ]).unwrap();
        assert!(cli.json);
        match &cli.command {
            Command::Run { .. } => {}
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn test_global_json_not_set() {
        let cli = Cli::try_parse_from(&["paw-rs", "compile", "--spec", "x"]).unwrap();
        assert!(!cli.json);
    }

    #[test]
    fn test_global_api_url() {
        let cli = Cli::try_parse_from(&[
            "paw-rs", "--api-url", "https://custom.example.com",
            "compile", "--spec", "x",
        ]).unwrap();
        assert_eq!(cli.api_url.as_deref(), Some("https://custom.example.com"));
    }

    #[test]
    fn test_global_api_key() {
        let cli = Cli::try_parse_from(&[
            "paw-rs", "--api-key", "sk_test",
            "compile", "--spec", "x",
        ]).unwrap();
        assert_eq!(cli.api_key.as_deref(), Some("sk_test"));
    }

    #[test]
    fn test_global_both() {
        let cli = Cli::try_parse_from(&[
            "paw-rs",
            "--api-url", "https://a.com",
            "--api-key", "sk_test",
            "compile", "--spec", "x",
        ]).unwrap();
        assert_eq!(cli.api_url.as_deref(), Some("https://a.com"));
        assert_eq!(cli.api_key.as_deref(), Some("sk_test"));
    }

    #[test]
    fn test_json_does_not_apply_to_login() {
        let cli = Cli::try_parse_from(&["paw-rs", "--json", "login"]).unwrap();
        assert!(cli.json);
    }

    // ── UTF-8 safe preview ───────────────────────────────────────────

    #[test]
    fn test_utf8_preview_ascii() {
        assert_eq!(preview("hello world"), "hello world");
    }

    #[test]
    fn test_utf8_preview_short() {
        assert_eq!(preview("ab"), "ab");
    }

    #[test]
    fn test_utf8_preview_exact_80() {
        let s = "a".repeat(80);
        let p = preview(&s);
        assert_eq!(p.chars().count(), 80);
        assert_eq!(p, s);
    }

    #[test]
    fn test_utf8_preview_long() {
        let s = "a".repeat(200);
        assert_eq!(preview(&s).chars().count(), 80);
    }

    #[test]
    fn test_utf8_preview_multi_byte() {
        let s = "你好世界".repeat(30);
        let p = preview(&s);
        assert_eq!(p.chars().count(), 80);
        assert!(p.chars().all(|c| c.len_utf8() <= 3));
    }

    #[test]
    fn test_utf8_preview_emoji() {
        let s = "😀".repeat(100);
        let p = preview(&s);
        assert_eq!(p.chars().count(), 80);
    }

    #[test]
    fn test_utf8_preview_mixed() {
        assert_eq!(preview("abc😀def"), "abc😀def");
    }

    // ── Unknown / missing subcommand ─────────────────────────────────

    #[test]
    fn test_unknown_subcommand_fails() {
        let err = Cli::try_parse_from(&["paw-rs", "unknown"]).unwrap_err();
        assert!(err.kind() == clap::error::ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn test_no_args_fails() {
        let err = Cli::try_parse_from(&["paw-rs"]).unwrap_err();
        assert!(err.to_string().contains("subcommand") || err.to_string().contains("Usage"), "error: {err}");
    }
}
