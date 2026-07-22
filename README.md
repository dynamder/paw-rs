# PAW RS — ProgramAsWeights Rust SDK

非官方 Rust SDK，用于在 Rust 项目中嵌入 ProgramAsWeights 推理。

> [English version](./README.en.md)

---

## 安装

```toml
[dependencies]
paw-rs = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

默认后端为 **llama.cpp**（CPU，Qwen3-0.6B 30 tokens ≈ 240ms）。
如需 GPU 加速或 `PawFn<T>` 静态类型 API，启用 `candle` 后端：

```toml
paw-rs = { version = "0.1", default-features = false, features = ["candle", "cuda"] }
```

---

## SDK 快速开始

### 动态分发（Builder，任意后端）

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFnBuilder::builder()
        .slug("email-triage")
        .load()
        .await?;
    let result = f.run("Urgent: server is down!")?;
    println!("{result}"); // "immediate"
    Ok(())
}
```

### 编译新程序

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFnBuilder::builder()
        .spec("Classify sentiment: return POSITIVE or NEGATIVE")
        .compile()
        .await?;
    let result = f.run("I love this product!")?;
    println!("{result}"); // "POSITIVE"
    Ok(())
}
```

### 静态类型 + 模型共享

多个 `PawFn<T, B>` 共享同一份基座模型。两个后端均支持：

```rust
use paw_rs::prelude::*;
use paw_rs::paw_core::{Qwen3_0_6B, Candle};

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut a = PawFn::<Qwen3_0_6B, Candle>::load_slug("email-triage").await?;
    let mut b = PawFn::<Qwen3_0_6B, Candle>::compile_spec(
        "Classify sentiment", "paw-4b-qwen3-0.6b",
    ).await?;
    println!("{}", a.run("Server is down!")?);
    println!("{}", b.run("I love this product!")?);
    Ok(())
}
```

### 自定义参数推理

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFnBuilder::builder()
        .slug("email-triage")
        .load()
        .await?;

    let opts = paw_core::PawRuntimeOptions {
        max_tokens: Some(100),
        temperature: 0.7,
        ..Default::default()
    };
    let result = f.run_with("What should I do?", &opts)?;
    println!("{result}");
    Ok(())
}
```

### 模型副本与并发控制

`max_model_copies` 控制基模副本数和最大并发数：

```rust
use paw_llamacpp::{PawFnLoader, PawLlamaCppConfig};

// 默认 1 份：所有 PawFunction 共享基模，串行执行，内存最小
let config = PawLlamaCppConfig::default();

// 最多 4 份：懒加载，最多 4 个 PawFunction 可同时推理
let config = PawLlamaCppConfig::builder()
    .core(paw_config)
    .max_model_copies(4)
    .build();

let a = PawFnLoader::new("program_a_dir").config(config.clone()).load()?;
let b = PawFnLoader::new("program_b_dir").config(config).load()?;
// a 和 b 共用同一份基模池（若 interpreter 相同）
// 初始只有 1 份模型，运行中按需懒加载最多 4 份
```

Candle 后端同样支持：

```rust
use paw_candle::{PawFnLoader, PawCandleConfig};

let config = PawCandleConfig::builder()
    .core(paw_config)
    .max_model_copies(4)
    .build();
```

---

## Feature Flags

| flag | 说明 |
|------|------|
| `llamacpp` | llama.cpp 后端（默认） |
| `candle` | Candle 后端（需 `default-features = false`） |
| `cuda` | NVIDIA GPU（转发到已启用的后端） |
| `metal` | Apple Silicon GPU |
| `mkl` | Intel MKL CPU 加速（仅 candle） |

```bash
# llama.cpp CPU（默认）
cargo run -- run --program email-triage --input "test"

# candle + CUDA GPU
cargo run --no-default-features --features candle,cuda -- run --program email-triage --input "test"

# candle + MKL（CPU 加速）
cargo run --no-default-features --features candle,mkl -- run --program email-triage --input "test"
```

---

## 性能

| 后端 | Qwen3 (10 tokens) | 单份模型内存 | GPU 支持 |
|------|------------------|------------|---------|
| llama.cpp (CPU) | ~240ms | 588 MB | CUDA / Metal / Vulkan |
| candle (CPU, native) | ~680ms | 588 MB | CUDA / Metal |
| candle (CUDA) | ~200ms | 588 MB + VRAM | CUDA |

多个 PawFunction 共享基模时的内存对比（以 4 个为例）：

| 配置 | llama.cpp | candle |
|------|-----------|--------|
| 无共享（每实例 1 份） | ~2.4 GB | ~2.4 GB |
| 共享基模，串行（默认 `max_model_copies=1`） | ~588 MB | ~588 MB |
| 共享基模，4 路并行（`max_model_copies=4`） | ~2.4 GB | ~2.4 GB |

---

## 架构

| crate | 说明 |
|-------|------|
| `paw-core` | `InterpreterModel` / `Backend` trait, `PawFnTrait`, `PawRuntimeOptions`, HTTP 客户端, 缓存 |
| `paw-candle` | `CandleBackend`, `Qwen3Model`, `Gpt2Model`, 全局基模池 + 懒加载 |
| `paw-llamacpp` | `LlamaCppBackend`, CPU ~2.8x 快于 candle, 全局基模池 + 懒加载 |
| `paw-rs` | `PawFn<T, B>`, `PawFnBuilder`, CLI |

---

## Examples

| 示例 | Crate | 说明 | 需 API key |
|------|-------|------|-----------|
| `high_level` | `paw-rs` | Builder: 编译→推理 | 是 |
| `low_level` | `paw-rs` | Builder: 加载→推理 | 是 |
| `typed_api` | `paw-rs` | 静态类型 + 模型共享 | 是 |
| `qwen3_inference` | `paw-candle` | 加载已有程序推理 | 否 |
| `llamacpp_benchmark` | `paw-llamacpp` | llama.cpp 延迟测试 | 否 |
| `parallel_benchmark` | `paw-llamacpp` | 并行推理吞吐量测试 | 否 |
| `verify_backend` | `paw-llamacpp` | 后端正确性验证 | 否 |
| `compare_ref` | `paw-candle` | Candle 参考输出（交叉比对用） | 否 |
| `compare_test` | `paw-llamacpp` | Llamacpp 测试输出（交叉比对用） | 否 |
| `verify_bundle` | `paw-candle` | LoRA 前向验证 | 否 |
| `download_and_save` | `paw-core` | Bundle 格式 roundtrip | 否 |

```bash
# Builder（默认 llamacpp 后端）
PAW_API_KEY=sk_... cargo run --example high_level -p paw-rs

# 静态类型 + 模型共享（candle）
PAW_API_KEY=sk_... cargo run --example typed_api -p paw-rs --features candle

# llama.cpp 压测（无需 API key）
cargo run --release --example llamacpp_benchmark -p paw-llamacpp

# 并行推理测试
cargo run --release --example parallel_benchmark -p paw-llamacpp -- fccdea9da515e3f20dd6 4 3 30
```
