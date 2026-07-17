/// Example: download a .paw bundle, inspect and repack with PawFormatReader/PawFormatWriter.
///
/// Works on public programs without an API key.
/// Usage:
///   cargo run --example download_and_save
use paw_core::prelude::*;

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

    // ── Read tensors with PawFormatReader ──────────────────────────────
    let paw_file = std::fs::read_dir(&dir)?.find_map(|e| {
        let p = e.ok()?.path();
        (p.extension()? == "paw").then_some(p)
    });
    let (tensors, meta) = match paw_file {
        Some(ref p) => PawFormatReader::load(p)?,
        None => return Err(Error::Other("no .paw file in bundle".into())),
    };
    println!("paw tensors: {}, version: {}", tensors.len(), meta.format_version);

    // ── Repack and verify with PawFormatWriter ────────────────────────
    let tmp = std::env::temp_dir().join("repacked.paw");
    PawFormatWriter::save(&tmp, tensors, &meta)?;
    let (reloaded, _) = PawFormatReader::load(&tmp)?;
    println!("repack verified: {} tensors", reloaded.len());
    std::fs::remove_file(&tmp).ok();

    // ── Bundle metadata ──────────────────────────────────────────────
    println!("interpreter: {}", bundle.interpreter_model());
    println!("template: {} chars", bundle.prompt_template.len());
    println!(
        "adapter: {} KB",
        std::fs::metadata(&bundle.adapter_path)
            .map(|m| m.len() / 1024)
            .unwrap_or(0)
    );

    Ok(())
}
