use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use serde_json::Value as JsonValue;

use crate::error::{Error, Result};
use crate::format::meta::PawFileMeta;
use crate::format::tensor::TensorData;

const PAW_MAGIC: [u8; 4] = *b"PAW\x02";

/// Binary `.paw` v2 format reader.
pub struct PawFormatReader;

impl PawFormatReader {
    /// Check if a file has valid `.paw` magic bytes.
    pub fn is_paw_file<P: AsRef<Path>>(path: P) -> bool {
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).is_ok() && magic == PAW_MAGIC
    }

    /// Read a `.paw` file from disk → `(tensors, meta)`.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<(HashMap<String, TensorData>, PawFileMeta)> {
        let mut data = Vec::new();
        std::fs::File::open(path.as_ref())
            .and_then(|mut f| f.read_to_end(&mut data))
            .map_err(Error::Io)?;
        let (tensors, meta_value) = Self::from_bytes(&data)?;
        let meta: PawFileMeta = serde_json::from_value(meta_value).map_err(Error::Json)?;
        Ok((tensors, meta))
    }

    /// Read a `.paw` file and return raw metadata (untyped).
    pub fn load_raw<P: AsRef<Path>>(path: P) -> Result<(HashMap<String, TensorData>, JsonValue)> {
        let mut data = Vec::new();
        std::fs::File::open(path.as_ref())
            .and_then(|mut f| f.read_to_end(&mut data))
            .map_err(Error::Io)?;
        Self::from_bytes(&data)
    }

    /// Parse bytes → `(tensors, raw_metadata)`.
    pub fn from_bytes(data: &[u8]) -> Result<(HashMap<String, TensorData>, JsonValue)> {
        let mut cursor = std::io::Cursor::new(data);

        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|_| Error::Format("Failed to read magic".into()))?;
        if magic != PAW_MAGIC {
            return Err(Error::Format(format!("Invalid .paw magic: {magic:02x?}")));
        }

        let mut buf32 = [0u8; 4];
        cursor
            .read_exact(&mut buf32)
            .map_err(|_| Error::Format("Failed to read version".into()))?;
        let _version = u32::from_le_bytes(buf32);

        cursor
            .read_exact(&mut buf32)
            .map_err(|_| Error::Format("Failed to read meta_len".into()))?;
        let meta_len = u32::from_le_bytes(buf32) as usize;

        let mut meta_bytes = vec![0u8; meta_len];
        cursor
            .read_exact(&mut meta_bytes)
            .map_err(|_| Error::Format("Failed to read metadata".into()))?;
        let metadata: JsonValue = serde_json::from_slice(&meta_bytes).map_err(Error::Json)?;

        let tensor_slice = &data[cursor.position() as usize..];
        let safe = safetensors::SafeTensors::deserialize(tensor_slice)
            .map_err(|e| Error::Format(format!("safetensors error: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, tv) in safe.tensors() {
            tensors.insert(
                name,
                TensorData {
                    dtype: tv.dtype(),
                    shape: tv.shape().to_vec(),
                    data: tv.data().to_vec(),
                },
            );
        }

        Ok((tensors, metadata))
    }
}
