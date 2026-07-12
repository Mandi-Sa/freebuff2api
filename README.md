# freebuff2api

Codebuff Freebuff 的 OpenAI-compatible API

## 接口

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /healthz`

## 配置

### 获取 Token

无需安装 Freebuff / Codebuff CLI，可以直接打开公开页面自动获取 token：

```text
https://freebuff.071129.xyz/
```

使用方式：

1. 打开上面的地址
2. 选择 Freebuff
3. 点击“开始认证”，在跳转页面完成授权
4. 回到页面复制展示的 token
5. 将复制结果写入本项目 `.env`

示例：

```dotenv
FREEBUFF_TOKEN=你的 Freebuff Bearer token
```

多账号可用英文逗号分隔；并发请求会优先分配到空闲账号，避免单个
Freebuff 账号的全局 active free session 被并发切模型请求互相覆盖：

```dotenv
FREEBUFF_TOKEN=token-a,token-b,token-c
```

多账号还会按一天 24 小时平均分时使用：N 个 token 时，每个 token 使用
`24 / N` 小时后切换到下一个；即使没有请求，后台也会在窗口边界（时区由
`FREEBUFF_SCHEDULE_UTC_OFFSET` 决定）按点切换，并删除上一个 token 的上游会话。

### 模型准入

按**底层会话模型**判定，只允许两类，其余一律拒绝（400）：

- **唯一 premium 模型** `FREEBUFF_PREMIUM_MODEL`（默认 `moonshotai/kimi-k2.7-code`）；
- **unlimited 白名单** `FREEBUFF_UNLIMITED_MODEL`（逗号分隔，可多个）。

判定看的是模型实际使用的会话模型（`session_id`），因此借壳的 gemini 也会放行：
`gemini-3.1-pro` 借 kimi → 当 premium；两个 `gemini-*-flash-lite` 借 deepseek-flash →
当 unlimited。`/v1/models` 也只列出允许的模型。

### 调度规则

- **premium 请求**：锁定当前时段 token，繁忙只排队、不跨 token，也不跨模型；额度耗尽
  时直接返回 `premium quota exhausted ... resets at ...`。
- **unlimited 请求**：当前时段 token 繁忙或正持有 premium 会话时，回退到其他空闲 token。
  unlimited 会话不计额度，保留复用，待窗口轮换或切换模型时再删除。

### premium 计费块复用

premium 上游按 6 分钟一块阶梯计费（每块约 0.1，5/天）。同一 token 的 premium 会话在
一个计费块内**复用**（多次请求只算一块），后台看守在每个块边界前
`FREEBUFF_DESTROY_LEAD_SECONDS` 秒、token 空闲时销毁会话；销毁时正忙则顺延到下一块。

### 额度统计

每次会话响应里的 premium 用量（`used/limit/resetAt`）会被解析、落盘到
`FREEBUFF_QUOTA_FILE`，并在变化时打印日志。`GET /admin/quota` 返回各 token 的额度快照
（reset 后 `effective_used` 归零）。

日志统一带 `[请求id t序号/总数 模式]` 前缀（模式 `U`=unlimited 绿、`P`=premium 黄），
便于并发归组。

复制 `.env.example` 为 `.env`，然后填写上游 token：

```powershell
Copy-Item .env.example .env
```

`.env` 示例：

```dotenv
FREEBUFF_TOKEN=你的 Freebuff Bearer token
FREEBUFF_API_KEY=本地 OpenAI API key，可留空
FREEBUFF_AD_PROVIDERS=gravity,carbon
FREEBUFF_PROXY_ENABLED=false
FREEBUFF_PROXY_URL=
FREEBUFF_DEBUG=false
FREEBUFF_LOG_LEVEL=INFO
FREEBUFF_LOG_BODY_CHARS=2000
FREEBUFF_LOG_COLOR=true
FREEBUFF_HOST=0.0.0.0
FREEBUFF_PORT=8000
```

默认不启用代理，所有上游请求直连，且不会读取系统 `HTTP_PROXY` / `HTTPS_PROXY`。

需要让所有上游请求经过代理时，在 `.env` 中开启：

```dotenv
FREEBUFF_PROXY_ENABLED=true
FREEBUFF_PROXY_URL=http://127.0.0.1:7890
```

支持 HTTP 和 SOCKS 代理，例如：

```dotenv
FREEBUFF_PROXY_URL=http://127.0.0.1:7890
FREEBUFF_PROXY_URL=socks5://127.0.0.1:1080
FREEBUFF_PROXY_URL=socks5h://127.0.0.1:1080
```

当前内置 Freebuff 模型：

- `deepseek/deepseek-v4-flash`
- `deepseek/deepseek-v4-pro`
- `moonshotai/kimi-k2.7-code`
- `minimax/minimax-m3`
- `google/gemini-2.5-flash-lite`
- `google/gemini-3.1-flash-lite`
- `google/gemini-3.1-pro-preview`
- `mimo/mimo-v2.5`
- `mimo/mimo-v2.5-pro`

调试空返回或上游异常时：

```dotenv
FREEBUFF_DEBUG=true
FREEBUFF_LOG_LEVEL=DEBUG
FREEBUFF_LOG_BODY_CHARS=0
```

## 运行

```powershell
uv sync
uv run freebuff2api
```

或：

```powershell
python -m pip install -e .
python main.py
```

## 调用示例

```powershell
curl http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:FREEBUFF_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

流式：

```powershell
curl -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:FREEBUFF_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "写一个 Python 快排"}],
    "stream": true
  }'
```

## 感谢

> [FreeBuff](https://freebuff.com)
