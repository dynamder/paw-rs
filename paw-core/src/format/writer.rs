use std::collections::HashMap;
use std::path::Path;

use serde_json::Value as JsonValue;

use crate::error::{Error, Result};
use crate::format::meta::PawFileMeta;
use crate::format::tensor::TensorData;

const PAW_MAGIC: [u8; 4] = *b"PAW\x02";
const PAW_VERSION: u32 = 2;
const HEADER_SIZE: usize = 12;

/// Binary `.paw` v2 format writer.
pub struct PawFormatWriter;

impl PawFormatWriter {
    /// Write a `.paw` file to disk.
    pub fn save<P: AsRef<Path>>(
        path: P,
        tensors: HashMap<String, TensorData>,
        metadata: &PawFileMeta,
    ) -> Result<()> {
        let meta_value = serde_json::to_value(metadata).map_err(Error::Json)?;
        let bytes = Self::to_bytes(tensors, &meta_value)?;
        std::fs::write(path.as_ref(), bytes).map_err(Error::Io)
    }

    /// Serialize `(tensors, metadata)` → bytes.
    pub fn to_bytes(tensors: HashMap<String, TensorData>, metadata: &JsonValue) -> Result<Vec<u8>> {
        let meta_bytes = serde_json::to_vec(metadata).map_err(Error::Json)?;

        let view_pairs: Vec<(&str, &TensorData)> =
            tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();

        let tensor_bytes = safetensors::serialize(view_pairs, None)
            .map_err(|e| Error::Format(format!("safetensors serialize error: {e}")))?;

        let mut output = Vec::with_capacity(HEADER_SIZE + meta_bytes.len() + tensor_bytes.len());
        output.extend_from_slice(&PAW_MAGIC);
        output.extend_from_slice(&PAW_VERSION.to_le_bytes());
        output.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(&meta_bytes);
        output.extend_from_slice(&tensor_bytes);
        Ok(output)
    }

    /// Save KV layers and LoRA weights as a `.paw` program file.
    pub fn save_program<P: AsRef<Path>>(
        filepath: P,
        kv_layers: Option<Vec<(TensorData, TensorData)>>,
        lora_weights: Option<HashMap<String, TensorData>>,
        meta: PawFileMeta,
    ) -> Result<()> {
        let mut tensors: HashMap<String, TensorData> = HashMap::new();
        let mut meta = meta;

        if let Some(layers) = kv_layers {
            meta.num_layers = layers.len() as u32;
            for (i, (k, v)) in layers.into_iter().enumerate() {
                tensors.insert(format!("layer_{i}_key"), k);
                tensors.insert(format!("layer_{i}_value"), v);
            }
        }

        if let Some(lora) = lora_weights {
            meta.has_lora = !lora.is_empty();
            for (name, t) in lora {
                tensors.insert(format!("lora_{name}"), t);
            }
        }

        Self::save(filepath, tensors, &meta)
    }

    /// Load KV cache layers from a `.paw` file.
    pub fn load_program<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Vec<(TensorData, TensorData)>, PawFileMeta)> {
        let (tensors, meta) = super::reader::PawFormatReader::load(path)?;

        let layer_count = tensors.keys().filter(|k| k.ends_with("_key")).count();
        let mut kv = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let key = tensors
                .get(&format!("layer_{i}_key"))
                .ok_or_else(|| Error::Format(format!("Missing layer_{i}_key")))?;
            let val = tensors
                .get(&format!("layer_{i}_value"))
                .ok_or_else(|| Error::Format(format!("Missing layer_{i}_value")))?;
            kv.push((key.clone(), val.clone()));
        }
        Ok((kv, meta))
    }

    /// Load LoRA weights from a `.paw` file.
    pub fn load_lora<P: AsRef<Path>>(
        path: P,
    ) -> Result<(HashMap<String, TensorData>, PawFileMeta)> {
        let (tensors, meta) = super::reader::PawFormatReader::load(path)?;

        let lora = tensors
            .into_iter()
            .filter(|(k, _)| k.starts_with("lora_"))
            .map(|(k, v)| (k[5..].to_string(), v))
            .collect();
        Ok((lora, meta))
    }
}
