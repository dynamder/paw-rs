use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    let opt = Option::<T>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

// ── Newtype wrappers ──────────────────────────────────────────────────────

/// A hex-encoded program hash ID (16+ hex characters).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ProgramId(String);

impl ProgramId {
    /// Create a `ProgramId` if the string looks like a valid hex hash.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let s = id.into();
        if Self::is_hash_id(&s) {
            Ok(Self(s))
        } else {
            Err(Error::Other(format!("Invalid program ID: {s}")))
        }
    }

    /// Create a `ProgramId` without validation (for trusted input).
    pub fn new_unchecked(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Check if a string looks like a hex hash ID.
    pub fn is_hash_id(s: &str) -> bool {
        s.len() >= 16 && s.chars().all(|c| c.is_ascii_hexdigit())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ProgramId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProgramId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for ProgramId {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

impl From<ProgramId> for String {
    fn from(id: ProgramId) -> Self {
        id.0
    }
}

/// A human-readable program slug (e.g. `"email-triage"`, `"da03/my-classifier@v2"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Slug(String);

impl Slug {
    pub fn new(slug: impl Into<String>) -> Self {
        Self(slug.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Slug {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Slug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for Slug {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

impl From<Slug> for String {
    fn from(s: Slug) -> Self {
        s.0
    }
}

// ── API Response Types ───────────────────────────────────────────────────

/// Outcome of a compilation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    #[serde(alias = "program_id")]
    pub id: String,
    pub status: String,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub compiler_snapshot: Option<String>,
    #[serde(default)]
    pub compiler_kind: Option<String>,
    #[serde(default)]
    pub pseudo_program_strategy: Option<String>,
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub runtime_manifest_version: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub timings: Option<HashMap<String, f64>>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub version_action: Option<String>,
}

impl Program {
    pub fn display_label(&self) -> String {
        let mut label = self.id.clone();
        if let Some(slug) = &self.slug {
            label = slug.clone();
            if let Some(v) = self.version
                && v > 1
            {
                label.push_str(&format!(" v{v}"));
            }
        }
        label
    }
}

/// Metadata extracted from a .paw bundle's `meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMeta {
    pub spec: String,
    #[serde(default)]
    pub interpreter: String,
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub runtime_manifest_version: Option<u32>,
    #[serde(default)]
    pub runtime: Option<RuntimeManifest>,
    #[serde(default)]
    pub pseudo_program: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub source_info: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub format_version: Option<u32>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub adapter_format: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Runtime manifest describing a base model and its capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub runtime_id: String,
    pub manifest_version: u32,
    pub display_name: String,
    pub interpreter: String,
    pub adapter_format: String,
    #[serde(default)]
    pub local_sdk: Option<LocalSdkInfo>,
}

/// Local SDK configuration inside a runtime manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSdkInfo {
    pub supported: bool,
    #[serde(default)]
    pub base_model: Option<BaseModelInfo>,
    #[serde(default)]
    pub n_ctx: u32,
}

/// Base model download information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelInfo {
    pub provider: String,
    pub repo: String,
    #[serde(rename = "file")]
    pub filename: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
}

impl BaseModelInfo {
    pub fn download_url(&self) -> String {
        self.url.clone().unwrap_or_else(|| {
            format!(
                "https://huggingface.co/{}/resolve/main/{}",
                self.repo, self.filename
            )
        })
    }
}

/// A compiler available on the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerInfo {
    pub name: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub runtime: Option<String>,
}

/// List of compilers returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerList {
    pub compilers: Vec<CompilerInfo>,
}

/// Version history for a slug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionList {
    pub slug: String,
    #[serde(default)]
    pub main_version: Option<u32>,
    pub versions: Vec<VersionEntry>,
}

/// A single version entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    pub version: u32,
    pub program_id: String,
    #[serde(default)]
    pub is_main: bool,
    #[serde(default)]
    pub created_at: Option<String>,
}

/// List of programs for the authenticated user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramList {
    pub programs: Vec<ProgramSummary>,
    pub total: u32,
    pub page: u32,
    pub per_page: u32,
}

/// Summary of a program in the user's list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramSummary {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub spec: Option<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub created_at: Option<String>,
}

/// Slug resolution response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlugResolveResponse {
    pub program_id: String,
}
