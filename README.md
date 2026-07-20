# PAW RS — ProgramAsWeights Rust SDK

非官方 Rust SDK，用于在 Rust 项目中嵌入 ProgramAsWeights 推理。

> [English version](./README.en.md)

**⚠️ 注意**：此 SDK 非官方维护。CPU 推理速度较官方 Python SDK（llama.cpp 后端）慢约 4–5x。可通过 `--features cuda` 启用 GPU 加速。

---

## 安装

```toml
[dependencies]
paw-rs = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

---

## SDK 快速开始

### 加载已有程序并推理

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .slug("email-triage")
        .load()
        .await?;
    let result = f.run("Urgent: server is down!")?;
    println!("{result}"); // "immediate"
    Ok(())
}
```

### 编译新程序并推理

```rust
use paw_rs::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .spec("Classify sentiment: return POSITIVE or NEGATIVE")
        .compile()
        .await?;
    let result = f.run("I love this product!")?;
    println!("{result}"); // "POSITIVE"
    Ok(())
}
```

### 自定义参数推理

```rust
use paw_rs::prelude::*;
use paw_rs::paw_candle::PawRuntimeOptions;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let mut f = PawFn::builder()
        .slug("email-triage")
        .load()
        .await?;

    let opts = PawRuntimeOptions {
        max_tokens: Some(100),
        temperature: 0.7,
        ..Default::default()
    };
    let result = f.run_with("What should I do about this?", &opts)?;
    println!("{result}");
    Ok(())
}
```

### 自定义配置

```rust
use paw_rs::prelude::*;
use paw_rs::paw_core::PawConfig;
use paw_rs::paw_candle::DevicePreference;

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let config = PawConfig::builder()
        .api_url("https://custom.example.com")
        .api_key("paw_sk_xxx")
        .n_ctx(4096)
        .verbose(true)
        .build()?;

    let mut f = PawFn::builder()
        .config(config)
        .device(DevicePreference::Cpu)
        .slug("email-triage")
        .load()
        .await?;

    let result = f.run("Is this urgent?")?;
    println!("{result}");
    Ok(())
}
```

### 底层 API：手动加载

```rust
use paw_rs::paw_core::{PawClient, PawConfig};
use paw_rs::paw_candle::{PawFnLoader, PawCandleConfig, PawRuntimeOptions};

#[tokio::main]
async fn main() -> std::result::Result<(), paw_core::Error> {
    let config = PawConfig::from_env();
    let client = PawClient::new(&config);

    // 下载 .paw 程序包
    let dir = client.download_paw("some-program-id").await?;

    // 本地加载模型
    let mut func = PawFnLoader::new(dir)
        .config(PawCandleConfig::default())
        .load()?;

    let result = func.run("hello", &PawRuntimeOptions::default())?;
    println!("{result}");
    Ok(())
}
```

---

## CLI

以下是 `paw-rs` 命令行工具的完整使用方式：

### 登录认证

```bash
# 交互式登录（打开浏览器 + 输入 API key）
paw-rs login

# 直接提供 API key
paw-rs login paw_sk_your_api_key

# 全局 --api-key 也可用于所有命令
paw-rs --api-key paw_sk_xxx run --program email-triage --input "test"
```

### 编译程序

```bash
# 最简编译
paw-rs compile --spec "Classify message urgency as low, medium, or high"

# 指定编译器、slug、设为私有
paw-rs compile \
  --spec "Extract key points from text" \
  --compiler paw-4b-qwen3-0.6b \
  --slug my-extractor \
  --private

# JSON 输出（适合脚本 / agent 集成）
paw-rs --json compile --spec "Classify sentiment"
```

### 运行程序

```bash
# 通过 slug 运行
paw-rs run --program email-triage --input "The server is on fire!"

# 通过程序 ID 运行
paw-rs run --program a1b2c3d4e5f6a1b2 --input "hello"

# 带参数运行
paw-rs run \
  --program email-triage \
  --input "What's the deadline?" \
  --max-tokens 256 \
  --temperature 0.5 \
  --verbose

# JSON 输出
paw-rs --json run --program email-triage --input "test"
# → {"program":"email-triage","input":"test","output":"immediate"}
```

### 修改程序 slug

```bash
# 设置或修改 slug（位置参数）
paw-rs rename a1b2c3d4e5f6a1b2 my-slug

# 移除 slug（传空字符串）
paw-rs rename a1b2c3d4e5f6a1b2 ""

# JSON 输出
paw-rs rename a1b2c3d4e5f6a1b2 my-slug --json
```

### 查询程序信息

```bash
# 查询程序元数据（位置参数）
paw-rs info email-triage

# 通过 ID 查询
paw-rs info a1b2c3d4e5f6a1b2

# JSON 输出
paw-rs info email-triage --json
```

### 全局选项

```bash
# 自定义服务器
paw-rs --api-url https://api.custom.com compile --spec "..."

# 全局 API key（适用于所有命令）
paw-rs --api-key paw_sk_xxx info my-program

# JSON 模式（适用于所有命令）
paw-rs --json compile --spec "Classify urgency"
paw-rs --json run --program email-triage --input "test"
paw-rs --json info my-program
```

### Agent / 脚本工作流示例

```bash
# 编译 → 获取 ID → 运行
PROGRAM_ID=$(paw-rs --json compile --spec "Classify urgency" | jq -r '.program_id')
paw-rs run --program "$PROGRAM_ID" --input "Please review by EOD" --json | jq -r '.output'
```

---

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `PAW_API_URL` | PAW 服务器地址 | `https://programasweights.com` |
| `PAW_API_KEY` | API 密钥 | (无) |
| `PAW_CACHE_DIR` | 缓存目录 | `~/.cache/programasweights/` |
| `PAW_CONFIG_DIR` | 配置目录 | `~/.config/programasweights/` |
| `PAW_N_CTX` | 上下文窗口大小 | `2048` |
| `PAW_GPU_LAYERS` | GPU 层数 (`-1`=全部, `0`=CPU) | `-1` |
| `PAW_VERBOSE` | 详细日志 (`1`/`true`) | `false` |
| `PAW_OFFLINE` | 离线模式 | `false` |

---

## 支持模型

| 模型 | ID | GGUF 大小 |
|---|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 594 MB |
| GPT-2 (124M) | `gpt2` | 134 MB |

---

## Feature flags

| flag | 说明 |
|---|---|
| `cuda` | NVIDIA GPU 加速 (`--features cuda`) |
| `metal` | Apple Silicon GPU 加速 |

```bash
cargo run --features cuda -- run --program email-triage --input "test"
```

---

## 性能

| | Python SDK (llama.cpp) | Rust SDK (candle CPU) |
|---|---|---|
| Qwen3 (30 tokens) | ~200ms | ~900ms |
| GPT-2 (10 tokens) | — | ~140ms |
| 模型内存 | ~600MB | ~600MB |

---

## Examples

以下示例位于 `<crate>/examples/`，可直接运行：

| 示例 | Crate | 说明 | 需 API key |
|------|-------|------|-----------|
| [`high_level`](paw-rs/examples/high_level.rs) | `paw-rs` | 高层 API：编译→推理（一键） | 是 |
| [`low_level`](paw-rs/examples/low_level.rs) | `paw-rs` | 底层 API：6 步手动流程 | 是 |
| [`qwen3_inference`](paw-candle/examples/qwen3_inference.rs) | `paw-candle` | 加载已有程序并推理 | 否 |
| [`gpt2_inference`](paw-candle/examples/gpt2_inference.rs) | `paw-candle` | 编译→下载→推理（GPT-2） | 是 |
| [`download_and_save`](paw-core/examples/download_and_save.rs) | `paw-core` | 下载 bundle + 二进制格式 roundtrip | 否 |
| [`verify_bundle`](paw-candle/examples/verify_bundle.rs) | `paw-candle` | 加载 LoRA → forward pass 校验 | 否 |

```bash
# 高层 API 示例（需 API key）
PAW_API_KEY=paw_sk_... cargo run --example high_level -p paw-rs

# 底层 API 示例
PAW_API_KEY=paw_sk_... cargo run --example low_level -p paw-rs

# 加载已有程序（无需 API key）
cargo run --release --example qwen3_inference -p paw-candle
```

---

## 架构

| crate | 说明 |
|---|---|
| `paw-core` | HTTP 客户端、缓存管理、bundle 解析、类型定义 |
| `paw-candle` | Candle 推理引擎、量化模型加载、LoRA 适配器 |
| `paw-rs` | 高层 API (`PawFn` / `PawFnBuilder`) + CLI 工具 + examples |

底层 crate 通过 `paw_rs::paw_core` 和 `paw_rs::paw_candle` 直接访问。完整示例见 `paw-rs/examples/`、`paw-candle/examples/`、`paw-core/examples/`。

---

## 相关链接

- [ProgramAsWeights 官网](https://programasweights.com)
- [Python SDK 文档](https://programasweights.readthedocs.io)
- [Candle 框架](https://github.com/huggingface/candle)
