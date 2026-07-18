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
use paw_core::{CompileRequest, BundleMeta, PawClient, PawConfig, login};

#[derive(Parser)]
#[command(name = "paw-rs", about = "ProgramAsWeights: compile and run neural programs")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(long, global = true, help = "PAW server URL")]
    api_url: Option<String>,

    #[arg(long, global = true, help = "API key")]
    api_key: Option<String>,
}

#[derive(Subcommand)]
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

        #[arg(long, help = "JSON output")]
        json: bool,
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
_verbose: bool,

        #[arg(long, help = "JSON output")]
        json: bool,
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

        #[arg(help = "New slug (empty string to remove)")]
        new_slug: String,

        #[arg(long, help = "JSON output")]
        json: bool,
    },

    /// Show program info
    Info {
        #[arg(help = "Program ID or slug")]
        program: String,

        #[arg(long, help = "JSON output")]
        json: bool,
    },
}

fn apply_auth_overrides(api_url: Option<&str>, api_key: Option<&str>) {
    // SAFETY: Setting env vars before any other code runs is safe in single-threaded startup.
    if let Some(url) = api_url {
        unsafe { std::env::set_var("PAW_API_URL", url); }
    }
    if let Some(key) = api_key {
        unsafe { std::env::set_var("PAW_API_KEY", key); }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    apply_auth_overrides(cli.api_url.as_deref(), cli.api_key.as_deref());

    let result = match &cli.command {
        Command::Compile { spec, compiler, slug, private, json } => {
            cmd_compile(spec, compiler.as_deref(), slug.as_deref(), *private, *json).await
        }
        Command::Run { program, input, max_tokens, temperature, _verbose: _, json } => {
            cmd_run(program, input, *max_tokens, *temperature, false, *json).await
        }
        Command::Login { key } => {
            cmd_login(key.as_deref())
        }
        Command::Rename { program, new_slug, json } => {
            cmd_rename(program, new_slug, *json).await
        }
        Command::Info { program, json } => {
            cmd_info(program, *json).await
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
        let preview = if spec.len() > 80 { &spec[..80] } else { spec };
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

async fn cmd_run(program_ref: &str, input: &str, max_tokens: Option<usize>, temperature: Option<f64>, _verbose: bool, json: bool) -> Result<i32, paw_core::Error> {
    let config = PawConfig::from_env();

    let builder = paw_rs::PawFn::builder()
        .config(config)
        .slug(program_ref);

    // Try to resolve as a program ID first, fallback to slug.
    let mut func = match builder.load().await {
        Ok(f) => f,
        Err(_) => {
            // If slug loading failed, try as a direct program ID (download by ID).
            let client = PawClient::new(&PawConfig::from_env());
            let dir = client.download_paw(program_ref).await?;
            let candle_config = paw_candle::PawCandleConfig::builder()
                .core(PawConfig::from_env())
                .build();
            let inner = paw_candle::PawFnLoader::new(dir)
                .config(candle_config)
                .load()?;
            paw_rs::PawFn::from_inner(inner)
        }
    };

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
    match key {
        Some(k) => {
            login(k).map_err(|e| paw_core::Error::Io(e))?;
            println!("API key saved.");
        }
        None => {
            login("").map_err(|e| paw_core::Error::Io(e))?;
            println!("Login page opened in your browser.");
        }
    }
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

fn extra_str(meta: &BundleMeta, key: &str, default: &str) -> String {
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
