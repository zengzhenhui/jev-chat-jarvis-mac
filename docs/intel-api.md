# Intel / API-only 安装与验收

## 范围

- 支持目标：macOS 13+，Intel x86_64 / Apple Silicon arm64，Python 3.12（源码与 app 相同）。Linux 仅用于离线回归，不能运行原生悬浮窗。
- 基础安装仅依赖 NumPy 与 macOS PyObjC；torch、laya、transformers、huggingface-hub 是带 macOS arm64 标记的 `local` 可选依赖。
- `uv.lock` 同时记录可选依赖不表示会安装它们。`uv sync --locked` 不选 local；只有 `--extra local` 才安装本地栈。不要用 `--all-extras` 当作 Intel 安装命令。
- 无内置密钥/作者中转。判断、排序、生成会向配置的服务发送聊天正文、启用的上下文与人工背景；可能收费。通用模型的风险与排序分数是启发式自评，不是校准后的概率。

## 启动

1. 安装 uv（https://docs.astral.sh/uv/getting-started/installation/），进入本仓库目录。启动器也会在缺失时下载官方 uv 安装脚本。
2. 创建 `~/.config/jev-jarvis/env`，权限设为 600。不要把真实密钥提交到仓库。示例（服务地址、模型按实际提供商填写）：

   ```sh
   export JUDGE_BACKEND=api
   export OPENAI_API_KEY="你的密钥"
   export OPENAI_BASE_URL="https://api.openai.com/v1"
   export OPENAI_MODEL="你的服务支持的非思考且支持 JSON 输出的模型"
   ```

   Anthropic 兼容服务可改用 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`。OpenAI 组优先；OpenAI 兼容端点需支持 `response_format: {"type":"json_object"}`。设置页连接测试也会验证判断所需结构化 JSON。可选 `TYPESAFE_API_KEY` 独立做判断与排序。无密钥的本机 Ollama 可填占位密钥 `ollama` 与 `http://localhost:11434/v1`，需自行运行服务与安装模型。
3. `./start.command`。设置中的模型修改需退出重开。默认不下载数 GB 模型。
4. 根据所用聊天应用授予屏幕录制 / 辅助功能权限，再重启。只读识别；发送始终由你手动完成。

## Apple Silicon 本地模式

在用户 env 中设置 `export JUDGE_BACKEND=local`，重新启动；两个启动器都会明确选择 `--extra local`。单独运行本地模型自测也要加 `uv run --extra local`。本地 torch 2.14 wheel 需要原生 ARM 与 macOS 14+；Intel / Rosetta 即使保留旧 local 设置也按 API 模式打开，以便进入设置修正。生成仍需用户配置 API。

## 构建 app

在任一受支持的 Mac 安装 Xcode Command Line Tools 后执行 `./packaging/build_app.sh`。clang 同时编译 arm64 与 x86_64，`lipo -verify_arch` 验证双切片；Info.plist 声明两个架构，最低 macOS 13。`./packaging/release.sh` 再次核验压缩包中的两种切片。Linux 不能产出或验证此原生 app。

包仍为启动器与源码，不捆绑 Python 或依赖。首次联网安装锁定依赖，后续启动仍同步 lock，防止升级或切换模式后残留错误依赖。环境目录为 `~/Library/Application Support/jev-jarvis/venv-<架构>`；旧版 `venv` 不复用。

## 必做的 Intel 实机验收（当前尚未执行）

1. `uname -m` 为 x86_64；`uv sync --locked` 成功，`uv pip list` 不含 torch/laya/transformers。
2. 无 key 启动显示配置引导，不下载本地模型、不发出聊天 API 请求。
3. 配置自己的 key 后，先用设置页固定问候语「测试连接」，再用非敏感测试聊天确认意图、风险、候选与排序。
4. 验证 API 401、超时、非法 JSON 时错误可见，不加载本地模型，下一条有效请求可以恢复。
5. 授权后分别验证 OCR / AX 可用、窗口切换、剪贴板回退；确认不会自动发送。
6. `lipo -archs jev-jarvis.app/Contents/MacOS/jev-jarvis` 同时包含 arm64 与 x86_64；从 Finder 原生启动成功。签名、公证、Gatekeeper 与 TCC 行为仍需真机验收。

## 依赖资料

- [uv 可选依赖与锁定](https://docs.astral.sh/uv/concepts/projects/sync/)
- [PyObjC Cocoa 12.2.2 文件与 universal2 wheels](https://pypi.org/project/pyobjc-framework-Cocoa/12.2.2/)

以上平台声明是实现目标，离线单元测试不能替代 macOS 实机验证。
