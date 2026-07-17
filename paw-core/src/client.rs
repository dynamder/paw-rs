use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use tracing::{debug, info, warn};

use crate::cache::CacheManager;
use crate::config::PawConfig;
use crate::error::{Error, Result};
use crate::types::*;

// ── Shared HTTP helpers ────────────────────────────────────────────────

/// Recursively remove null entries from a JSON value (for lenient deserialization).
fn strip_nulls(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let cleaned: serde_json::Map<_, _> = map
                .into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls(v)))
                .collect();
            serde_json::Value::Object(cleaned)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(strip_nulls).collect())
        }
        other => other,
    }
}

fn build_headers(api_key: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Some(key) = api_key {
        if let Ok(val) = HeaderValue::from_str(key) {
            headers.insert("X-API-Key", val);
        }
    }
    headers
}

fn api_url(base: &str, path: &str) -> String {
    format!("{base}{path}")
}

async fn check_response(resp: reqwest::Response) -> Result<reqwest::Response> {
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message: text,
        });
    }
    Ok(resp)
}

async fn api_get<T: DeserializeOwned>(
    http: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<T> {
    let resp = http.get(url).headers(build_headers(api_key)).send().await?;
    let resp = check_response(resp).await?;
    Ok(resp.json().await?)
}

// ── Compile Request ────────────────────────────────────────────────────

/// A compiled compilation request with readable fields.
///
/// Create via [`CompileRequestBuilder`]:
///
/// ```rust,no_run
/// use paw_core::CompileRequest;
/// let req = CompileRequest::builder()
///     .spec("Classify sentiment")
///     .compiler("default")
///     .build()
///     .unwrap();
/// println!("{}", req.spec); // fields are pub
/// ```
#[derive(Debug, Clone)]
pub struct CompileRequest {
    pub spec: String,
    pub compiler: Option<String>,
    pub name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub public: bool,
    pub slug: Option<String>,
    pub ephemeral: bool,
}

impl CompileRequest {
    /// Start building a compilation request.
    pub fn builder() -> CompileRequestBuilder {
        CompileRequestBuilder::default()
    }
}

/// Builder for [`CompileRequest`].
///
/// # Example
///
/// ```rust,no_run
/// use paw_core::CompileRequest;
/// let req = CompileRequest::builder()
///     .spec("Classify sentiment")
///     .public(true)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Default)]
pub struct CompileRequestBuilder {
    spec: Option<String>,
    compiler: Option<String>,
    name: Option<String>,
    tags: Option<Vec<String>>,
    public: bool,
    slug: Option<String>,
    ephemeral: bool,
}

impl CompileRequestBuilder {
    /// **Required.** The program spec text.
    pub fn spec(mut self, spec: impl Into<String>) -> Self {
        self.spec = Some(spec.into());
        self
    }

    /// Compiler to use (default: server default).
    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }

    /// Human-readable name for the program.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Tags for the program.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Whether the program is public (default: `true`).
    pub fn public(mut self, public: bool) -> Self {
        self.public = public;
        self
    }

    /// Slug for the program.
    pub fn slug(mut self, slug: impl Into<String>) -> Self {
        self.slug = Some(slug.into());
        self
    }

    /// Mark as ephemeral (not stored on the server).
    pub fn ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    /// Build the request.
    pub fn build(self) -> Result<CompileRequest> {
        Ok(CompileRequest {
            spec: self
                .spec
                .ok_or_else(|| Error::Config("spec is required".into()))?,
            compiler: self.compiler,
            name: self.name,
            tags: self.tags,
            public: self.public,
            slug: self.slug,
            ephemeral: self.ephemeral,
        })
    }
}

// ── High-Level Client ──────────────────────────────────────────────────

/// High-level PAW REST API client.
///
/// Methods mirror the Python SDK behavior. For fine-grained operations,
/// call [`.raw()`](PawClient::raw) to enter the raw-ops layer.
pub struct PawClient {
    api_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
    cache: CacheManager,
}

impl PawClient {
    pub fn new(config: &PawConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_url: config.api_url().trim_end_matches('/').to_string(),
            api_key: config.effective_api_key(),
            http,
            cache: CacheManager::new(config),
        }
    }

    pub fn from_env() -> Self {
        Self::new(&PawConfig::from_env())
    }

    /// Switch to the raw-ops layer for fine-grained control.
    pub fn raw(&self) -> RawPawClient<'_> {
        RawPawClient {
            api_url: &self.api_url,
            api_key: self.api_key.as_deref(),
            http: &self.http,
            cache: &self.cache,
        }
    }

    fn ak(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    // ── High-Level Public API ─────────────────────────────────────────

    pub async fn compile(&self, request: CompileRequest) -> Result<Program> {
        let url = api_url(&self.api_url, "/api/v1/compile");
        debug!("POST {url} with spec ({} chars)", request.spec.len());

        let mut body = serde_json::json!({
            "spec": request.spec,
            "public": request.public,
        });

        if let Some(ref c) = request.compiler {
            body["compiler"] = serde_json::json!(c);
        }
        if let Some(ref n) = request.name {
            body["name"] = serde_json::json!(n);
        }
        if let Some(ref t) = request.tags {
            body["tags"] = serde_json::json!(t);
        }
        if let Some(ref s) = request.slug {
            body["slug"] = serde_json::json!(s);
        }
        if request.ephemeral {
            body["ephemeral"] = serde_json::json!(true);
        }

        let resp = self
            .http
            .post(&url)
            .headers(build_headers(self.ak()))
            .json(&body)
            .send()
            .await?;
        let resp = check_response(resp).await?;
        let value: serde_json::Value = resp.json().await?;
        let cleaned = strip_nulls(value);
        Ok(serde_json::from_value(cleaned)?)
    }

    pub async fn resolve_slug(&self, slug: &str) -> Result<String> {
        let url = api_url(&self.api_url, &format!("/api/v1/programs/resolve/{slug}"));
        debug!("GET {url}");
        let data: SlugResolveResponse = api_get(&self.http, &url, self.ak()).await?;
        Ok(data.program_id)
    }

    /// Download a .paw bundle, caching and extracting it.
    ///
    /// Behavior matches the Python SDK:
    /// 1. Check local cache
    /// 2. HTTP download with 202 polling
    /// 3. Extract the .paw archive
    /// 4. Hydrate runtime manifest
    /// 5. Return the extracted program directory
    pub async fn download_paw(&self, program_id: &str) -> Result<PathBuf> {
        let raw = self.raw();

        if let Some(dir) = raw.cache().get(program_id) {
            info!("Program {program_id} already cached");
            return Ok(dir);
        }

        let bytes = raw.fetch_paw_bytes(program_id).await?;
        let program_dir = raw.cache().store(program_id, &bytes)?;
        raw.hydrate_runtime_manifest(&program_dir).await;

        info!(
            "Program {program_id} ready at {path}",
            path = program_dir.display()
        );
        Ok(program_dir)
    }

    pub async fn get_program_meta(&self, program_id: &str) -> Result<BundleMeta> {
        let url = api_url(&self.api_url, &format!("/api/v1/programs/{program_id}"));
        api_get(&self.http, &url, self.ak()).await
    }

    pub async fn get_runtime_manifest(&self, runtime_id: &str) -> Result<RuntimeManifest> {
        if let Some(cached) = self.cache.get_cached_runtime_manifest(runtime_id) {
            return Ok(cached);
        }
        let url = api_url(
            &self.api_url,
            &format!("/api/v1/models/runtimes/{runtime_id}"),
        );
        let manifest: RuntimeManifest = api_get(&self.http, &url, self.ak()).await?;
        self.cache.save_runtime_manifest(&manifest);
        Ok(manifest)
    }

    pub async fn list_compilers(&self) -> Result<Vec<CompilerInfo>> {
        let url = api_url(&self.api_url, "/api/v1/models/compilers");
        let data: CompilerList = api_get(&self.http, &url, self.ak()).await?;
        Ok(data.compilers)
    }

    pub async fn list_slug_versions(&self, slug: &str) -> Result<VersionList> {
        let url = api_url(&self.api_url, &format!("/api/v1/programs/{slug}/versions"));
        api_get(&self.http, &url, self.ak()).await
    }

    pub async fn list_programs(&self, sort: &str, per_page: u32, page: u32) -> Result<ProgramList> {
        let url = api_url(&self.api_url, "/api/v1/programs");
        let resp = self
            .http
            .get(&url)
            .headers(build_headers(self.ak()))
            .query(&[
                ("mine", "true"),
                ("sort", sort),
                ("per_page", &per_page.to_string()),
                ("page", &page.to_string()),
            ])
            .send()
            .await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }
}

// ── Raw / Fine-Grained Client ──────────────────────────────────────────

/// Fine-grained operations for callers who want explicit control.
///
/// Accessed via [`PawClient::raw()`]. Each method does exactly one thing:
///
/// | Method | Responsibility |
/// |--------|---------------|
/// | `fetch_paw_bytes(id)` | Pure HTTP: poll + download, returns raw zip bytes |
/// | `cache()` | Access [`CacheManager`] for storage-only operations |
/// | `hydrate_runtime_manifest(dir)` | Orchestration: read meta.json, HTTP fetch manifest, write back |
pub struct RawPawClient<'a> {
    api_url: &'a str,
    api_key: Option<&'a str>,
    http: &'a reqwest::Client,
    cache: &'a CacheManager,
}

impl<'a> RawPawClient<'a> {
    /// Access the cache layer for storage-only operations.
    pub fn cache(&self) -> &'a CacheManager {
        self.cache
    }

    // ── Raw HTTP: no caching, no disk I/O ──────────────────────────────

    /// Download the raw .paw ZIP bytes for a program.
    ///
    /// Pure HTTP: polls with 202/404 retry, returns raw bytes.
    /// No cache check, no file writing.
    pub async fn fetch_paw_bytes(&self, program_id: &str) -> Result<Vec<u8>> {
        info!("Downloading program {program_id}");
        let max_wait: u64 = 60;
        let mut elapsed: u64 = 0;
        let mut waiting_logged = false;
        let url = |id: &str| api_url(self.api_url, &format!("/api/v1/programs/{id}/download"));

        let resp = loop {
            let resp = self
                .http
                .get(&url(program_id))
                .headers(build_headers(self.api_key))
                .send()
                .await?;

            match resp.status() {
                StatusCode::ACCEPTED => {
                    if !waiting_logged {
                        info!("Waiting for program {program_id} to be ready...");
                        waiting_logged = true;
                    }
                    let retry_after = resp
                        .headers()
                        .get("Retry-After")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(3);
                    tokio::time::sleep(Duration::from_secs(retry_after)).await;
                    elapsed += retry_after;
                    if elapsed >= max_wait {
                        return Err(Error::Timeout(max_wait));
                    }
                    continue;
                }
                StatusCode::NOT_FOUND => {
                    let body = resp.text().await.unwrap_or_default();
                    if body.to_lowercase().contains("not found") {
                        return Err(Error::NotFound(format!(
                            "Program {program_id} not found on server"
                        )));
                    }
                    if elapsed < max_wait - 3 {
                        if !waiting_logged {
                            info!("Waiting for program {program_id} to be ready...");
                            waiting_logged = true;
                        }
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        elapsed += 3;
                        continue;
                    }
                    return Err(Error::Api {
                        status: 404,
                        message: body,
                    });
                }
                s if s.is_success() => break resp,
                s => {
                    let text = resp.text().await.unwrap_or_default();
                    return Err(Error::Api {
                        status: s.as_u16(),
                        message: text,
                    });
                }
            }
        };

        Ok(resp.bytes().await?.to_vec())
    }

    /// Fetch a runtime manifest from the server (no cache check).
    pub async fn fetch_runtime_manifest(&self, runtime_id: &str) -> Result<RuntimeManifest> {
        let url = api_url(
            self.api_url,
            &format!("/api/v1/models/runtimes/{runtime_id}"),
        );
        api_get(self.http, &url, self.api_key).await
    }

    // ── Orchestration ─────────────────────────────────────────────────

    /// Hydrate runtime manifest into `meta.json`.
    ///
    /// Reads the program's `meta.json`, fetches the runtime manifest if
    /// needed (checking cache first, then HTTP), and writes it back.
    pub async fn hydrate_runtime_manifest(&self, program_dir: &Path) {
        let meta_path = program_dir.join("meta.json");
        let meta_content = match std::fs::read_to_string(&meta_path) {
            Ok(c) => c,
            Err(_) => return,
        };

        let meta: BundleMeta = match serde_json::from_str(&meta_content) {
            Ok(m) => m,
            Err(_) => return,
        };

        if meta.runtime.as_ref().is_some_and(|r| {
            r.local_sdk
                .as_ref()
                .and_then(|s| s.base_model.as_ref())
                .is_some()
        }) {
            return;
        }

        let runtime_id = match meta.runtime_id.as_deref() {
            Some(id) => id,
            None => return,
        };

        let manifest = self
            .cache
            .get_cached_runtime_manifest(runtime_id)
            .or_else(|| {
                let rt = tokio::runtime::Runtime::new().ok()?;
                rt.block_on(self.fetch_runtime_manifest(runtime_id)).ok()
            });

        let manifest = match manifest {
            Some(m) => m,
            None => {
                warn!("Failed to resolve runtime manifest for {runtime_id}");
                return;
            }
        };

        if self.cache.hydrate_manifest(program_dir, &manifest).is_err() {
            warn!("Failed to write hydrated manifest");
        }
    }
}
