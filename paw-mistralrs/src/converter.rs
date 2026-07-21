use std::collections::HashMap;
use std::path::Path;

use paw_core::Error;

pub fn convert_adapter_gguf_to_safetensors(
    gguf_path: &Path,
    output_dir: &Path,
) -> Result<(), Error> {
    let mut file =
        std::fs::File::open(gguf_path).map_err(|e| Error::Other(format!("open adapter: {e}")))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| Error::Other(format!("read gguf: {e}")))?;

    let alpha = content
        .metadata
        .get("lora.alpha")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::F32(f) => Some(*f),
            candle_core::quantized::gguf_file::Value::I32(i) => Some(*i as f32),
            candle_core::quantized::gguf_file::Value::U32(u) => Some(*u as f32),
            _ => None,
        })
        .unwrap_or(16.0);

    let device = candle_core::Device::Cpu;

    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    let mut target_modules: Vec<String> = Vec::new();
    let mut rank: Option<usize> = None;

    for (name, _info) in &content.tensor_infos {
        let qtensor = content
            .tensor(&mut file, name, &device)
            .map_err(|e| Error::Other(format!("load {name}: {e}")))?;
        let t = qtensor
            .dequantize(&device)
            .map_err(|e| Error::Other(format!("dequantize {name}: {e}")))?;
        let data: Vec<f32> = t
            .flatten_all()
            .map_err(|e| Error::Other(format!("flatten {name}: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Other(format!("to_vec1 {name}: {e}")))?;

        let shape: Vec<usize> = t.shape().dims().to_vec();
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let base_name = name
            .strip_suffix(".lora_a")
            .or_else(|| name.strip_suffix(".lora_b"))
            .and_then(|n| n.strip_suffix(".weight"));

        if let Some(name) = base_name {
            let module = name.rsplit('.').next().unwrap_or(name);
            if !target_modules.contains(&module.to_string()) {
                target_modules.push(module.to_string());
            }
        }

        if let Some(a_name) = name.strip_suffix(".lora_a") {
            if let Some(base) = a_name.strip_suffix(".weight") {
                let savename = format!("{base}.lora_A.weight");
                tensors.push((savename, raw, shape));
                if rank.is_none() {
                    rank = t.shape().dims().first().copied();
                }
            }
        } else if let Some(b_name) = name.strip_suffix(".lora_b") {
            if let Some(base) = b_name.strip_suffix(".weight") {
                let savename = format!("{base}.lora_B.weight");
                tensors.push((savename, raw, shape));
            }
        }
    }

    let rank_val = rank.unwrap_or(16) as u64;
    let alpha_val = alpha as u64;

    let config = serde_json::json!({
        "lora_alpha": alpha_val,
        "lora_dropout": 0.0,
        "r": rank_val,
        "bias": "none",
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "unknown"
    });

    std::fs::create_dir_all(output_dir)
        .map_err(|e| Error::Other(format!("create output dir: {e}")))?;

    let mut tensor_data: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    for (name, raw, shape) in &tensors {
        let dtype = safetensors::Dtype::F32;
        let view = safetensors::tensor::TensorView::new(dtype, shape.clone(), raw.as_slice())
            .map_err(|e| Error::Other(format!("create tensor view: {e}")))?;
        tensor_data.insert(name.clone(), view);
    }

    let safetensors_bytes = safetensors::serialize(tensor_data, None)
        .map_err(|e| Error::Other(format!("serialize safetensors: {e}")))?;

    std::fs::write(
        output_dir.join("adapter_model.safetensors"),
        safetensors_bytes,
    )
    .map_err(|e| Error::Other(format!("write safetensors: {e}")))?;

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| Error::Other(format!("serialize config: {e}")))?;
    std::fs::write(output_dir.join("adapter_config.json"), config_json)
        .map_err(|e| Error::Other(format!("write config: {e}")))?;

    Ok(())
}
