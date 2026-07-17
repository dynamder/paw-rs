use std::collections::HashMap;

use crate::format::meta::{LoRAConfig, PawFileMeta};
use crate::format::reader::PawFormatReader;
use crate::format::tensor::TensorData;
use crate::format::writer::PawFormatWriter;
use safetensors::tensor::Dtype;

fn dummy_tensor() -> TensorData {
    let data: Vec<f32> = vec![1.0; 8];
    TensorData {
        dtype: Dtype::F32,
        shape: vec![2, 4],
        data: data.into_iter().flat_map(|f| f.to_le_bytes()).collect(),
    }
}

#[test]
fn test_roundtrip_empty() {
    let p = std::env::temp_dir().join("paw_test_empty.paw");
    PawFormatWriter::save_program(&p, None, None, PawFileMeta::default()).unwrap();
    assert!(PawFormatReader::is_paw_file(&p));
    let (t, m) = PawFormatWriter::load_program(&p).unwrap();
    assert!(t.is_empty());
    assert_eq!(m.format_version, 2);
    std::fs::remove_file(&p).ok();
}

#[test]
fn test_roundtrip_kv() {
    let p = std::env::temp_dir().join("paw_test_kv.paw");
    let meta = PawFileMeta {
        interpreter_model: "Qwen/Qwen3-0.6B".into(),
        spec: "Test".into(),
        ..PawFileMeta::default()
    };

    let layers = vec![(dummy_tensor(), dummy_tensor())];
    PawFormatWriter::save_program(&p, Some(layers), None, meta).unwrap();

    let (kv, m) = PawFormatWriter::load_program(&p).unwrap();
    assert_eq!(kv.len(), 1);
    assert_eq!(m.interpreter_model, "Qwen/Qwen3-0.6B");
    assert_eq!(m.spec, "Test");
    std::fs::remove_file(&p).ok();
}

#[test]
fn test_roundtrip_lora() {
    let p = std::env::temp_dir().join("paw_test_lora.paw");
    let mut lora = HashMap::new();
    lora.insert("blk.0.attn_q.lora_a".into(), dummy_tensor());

    let meta = PawFileMeta {
        has_lora: true,
        lora_config: Some(LoRAConfig {
            rank: 64,
            alpha: 16.0,
            target_modules: vec!["attn_q".into()],
        }),
        ..PawFileMeta::default()
    };

    PawFormatWriter::save_program(&p, None, Some(lora), meta).unwrap();
    let (l, m) = PawFormatWriter::load_lora(&p).unwrap();
    assert_eq!(l.len(), 1);
    assert!(l.contains_key("blk.0.attn_q.lora_a"));
    assert_eq!(m.lora_config.unwrap().rank, 64);
    std::fs::remove_file(&p).ok();
}

#[test]
fn test_validate_ok() {
    let p = std::env::temp_dir().join("paw_test_valid.paw");
    let meta = PawFileMeta {
        interpreter_model: "Qwen/Qwen3-0.6B".into(),
        ..Default::default()
    };
    PawFormatWriter::save_program(&p, None, None, meta).unwrap();

    let (tensors, meta) = PawFormatReader::load(&p).unwrap();
    meta.validate(&tensors).unwrap();
    std::fs::remove_file(&p).ok();
}

#[test]
fn test_validate_bad_magic() {
    let p = std::env::temp_dir().join("paw_test_bad.paw");
    std::fs::write(&p, b"NOT_PAW\x00\x00\x00\x01\x00\x00\x00{}").ok();
    assert!(!PawFormatReader::is_paw_file(&p));
    assert!(PawFormatReader::load(&p).is_err());
    std::fs::remove_file(&p).ok();
}

#[test]
fn test_from_to_bytes() {
    let mut tensors = HashMap::new();
    tensors.insert("test".into(), dummy_tensor());

    let meta = serde_json::json!({"hello": "world"});
    let bytes = PawFormatWriter::to_bytes(tensors, &meta).unwrap();
    let (loaded, meta2) = PawFormatReader::from_bytes(&bytes).unwrap();
    assert!(loaded.contains_key("test"));
    assert_eq!(meta2["hello"], "world");
}

#[test]
fn test_validate_rejects_suspicious() {
    let meta = PawFileMeta {
        spec: "eval(something)".into(),
        ..Default::default()
    };
    let result = meta.validate(&HashMap::new());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Suspicious"));
}
