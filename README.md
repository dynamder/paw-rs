# PAW RS — 非官方 ProgramAsWeights Rust SDK

非官方 Rust SDK，用于在 Rust 项目中嵌入 ProgramAsWeights 推理。

**⚠️ 注意**：此 SDK 非官方维护。CPU 推理速度较官方 Python SDK（llama.cpp 后端）慢约 4-5x。

> [English version](./README.en.md)

## 快速开始

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .slug("email-triage")
        .load()
        .await?;
    let result = f.run("Urgent: server is down!")?;
    println!("{result}"); // "immediate"
    Ok(())
}
```

## 安装

```toml
[dependencies]
paw-rs = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## CLI

```bash
# 加载已有程序并运行
paw-rs run --program email-triage --input "Is this urgent?"

# 编译新程序
paw-rs compile --spec "Classify sentiment as positive or negative"

# 查询程序信息
paw-rs info email-triage

# 全局选项
paw-rs --api-url https://api.programasweights.com --api-key paw_sk_xxx run --program ...
```

## 性能

| | Python SDK (llama.cpp) | Rust SDK (candle CPU) |
|---|---|---|
| Qwen3 (30 tokens) | ~200ms | ~900ms |
| GPT-2 (10 tokens) | — | ~140ms |
| 模型内存 | ~600MB | ~600MB |

## 支持模型

| 模型 | ID | GGUF 大小 |
|---|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 594 MB |
| GPT-2 (124M) | `gpt2` | 134 MB |

## Feature flags

| flag | 说明 |
|---|---|
| `cuda` | NVIDIA GPU 加速 (`--features cuda`) |
| `metal` | Apple Silicon GPU 加速 |

## 架构

| crate | 说明 |
|---|---|
| `paw-core` | HTTP 客户端、缓存、bundle 解析 |
| `paw-candle` | candle 推理引擎、模型加载、LoRA |
| `paw-rs` | 高层 API (`PawFn`/`PawFnBuilder`) + CLI |

底层 crate 通过 `paw_rs::paw_core` 和 `paw_rs::paw_candle` 直接访问。

## 相关链接

- [ProgramAsWeights 官网](https://programasweights.com)
- [Python SDK 文档](https://programasweights.readthedocs.io)
- [candle 框架](https://github.com/huggingface/candle)
