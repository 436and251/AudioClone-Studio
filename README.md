# Voice Dataset Builder

一个面向 GPT-SoVITS 等 TTS 训练流程的本地语音数据集构建工具。

它可以把 **本地音频 / 视频文件、YouTube 链接、Bilibili 链接** 自动处理为干净的人声训练片段，并生成 GPT-SoVITS 可直接使用的数据清单。

项目同时提供：

- **桌面 GUI**：适合日常使用，支持输入路径 / URL、模型管理、阶段进度、错误提示与任务续接；
- **CLI**：适合调试、批量处理和自动化脚本。

你只需要 **输入链接** 和 **确认地址**:
![image](assets/gui_example1.png)

> 当前默认假设：单个素材中**主要**只有一个目标说话人，可自动去除 BGM，但暂不具备多说话人分轨识别提取。


---

## 写在前面

本项目旨在大幅消减训练时寻找、整理、构建数据的时间成本和人力成本；鼓励使用者传播或者按需对其进行特化改造，但是！**请勿进行商用、诈骗等用途**。

本项目只是提供了一个快速批量构建voice clone等text-to-speech(tts) 任务训练数据的小工具；无法对任何使用者的任何行为有约束力，因此不对任何负面甚至非法行为和相应后果负任何责任。

---

## 主要功能以及技术概要

- YouTube / Bilibili 音频获取
- Demucs 人声分离
- FFmpeg 音频标准化
- faster-whisper 多语言 ASR
- 自动生成适合 TTS 的短语音片段
- ASR 置信度与时长过滤
- GPT-SoVITS `dataset.list` 和通用数据格式导出

典型处理流程：

```text
Audio / Video / URL
        ↓
Source acquisition
        ↓
Vocal separation
        ↓
Audio normalization
        ↓
ASR transcription
        ↓
Segmentation & filtering
        ↓
clips / transcript / manifest / dataset.list
```

---

## 项目结构

```text
.
├── voice_dataset_builder.py     # GUI 入口
├── build_dataset.py             # CLI 入口
├── requirements.txt
├── requirements-gui.txt
├── assets/
├── tts_builder/
│   ├── pipeline.py
│   ├── separator.py
│   ├── transcriber.py
│   ├── segmenter.py
│   ├── dataset.py
│   ├── cache.py
│   ├── sources/
│   └── gui/
└── tests/
```

---

# 快速开始

## 1. 环境要求

推荐环境：

```text
Windows 10 / 11
Python 3.12
NVIDIA GPU + CUDA（推荐，但不是必须）
FFmpeg
```

CPU 模式也可以运行，但 Demucs 和 ASR 会明显更慢。

---

## 2. 安装 Python 环境

推荐使用 `uv`。

安装 uv：

```powershell
winget install --id=astral-sh.uv -e
```

进入项目目录后创建环境：

```powershell
uv venv --python 3.12
```

激活环境：

```powershell
.venv\Scripts\activate.ps1
```

安装核心依赖：

```powershell
uv pip install -r requirements.txt
```

安装 GUI 依赖：

```powershell
uv pip install -r requirements-gui.txt
```

如果你已经有兼容的 GPT-SoVITS / PyTorch 环境，也可以直接复用现有虚拟环境。

---

## 3. 安装 FFmpeg

Windows 推荐：

```powershell
winget install --id Gyan.FFmpeg -e
```

安装后重新打开终端并确认：

```powershell
ffmpeg -version
ffprobe -version
```

只要这两个命令能正常输出版本信息，开发态就不需要手动复制 `ffmpeg.exe`。

---

# 使用桌面 GUI

启动：

```powershell
python voice_dataset_builder.py
```

首次启动只进行本机环境检查，不会立即联网下载模型。

GUI 中可以设置：

- 输入文件 / URL
- Speaker 名称
- Language
- ASR Model
- 当前任务输出目录
- 默认输出目录
- 模型存储根目录

开始任务后，界面会展示：

```text
Prepare
Source
Vocal Separation
Normalize
ASR
Segment
Export
```

各阶段会显示运行中、已完成、缓存命中或失败状态。

---

# 可选：接入训练模块（AudioClone Studio）

训练模块是可选的。未配置训练模块时，程序保持原来的 `Voice Dataset Builder` 单页模式，素材挖掘、CLI 和缓存逻辑不依赖训练仓，也不会启动训练环境。

配置至少一个通过协议检查的训练模块后，重启 GUI 会进入 `AudioClone Studio`，侧栏显示“素材挖掘”和“训练”。训练框架在训练页内部选择。

## 推荐的环境边界

AudioMiner 与训练模块使用各自的虚拟环境，避免 Torch、CUDA 和前端依赖互相污染。AudioMiner 只会用配置中的绝对 Python 路径启动训练子进程，不会把当前环境的 `PYTHONPATH`、`PYTHONHOME`、`HF_HOME` 或 `TORCH_HOME` 传给训练模块。

## 首次接入 voice-pipeline

`voice-pipeline` 不是需要提前常驻启动的服务。AudioClone Studio 会在开始训练时，使用你指定的训练环境启动独立子进程；关闭 GUI 前不需要另开一个终端运行 pipeline。

先在 PowerShell 中确认训练模块及预训练权重可用：

```powershell
Set-Location 'D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline'
$pipelinePython = 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe'
& $pipelinePython -m voice_pipeline module describe --json
& $pipelinePython -m voice_pipeline models verify --project-root . --profile v2ProPlus
```

第一条命令应输出包含 `"protocol_version":1` 和 `"gpt-sovits-v2proplus"` 的 JSON；第二条命令应确认 v2ProPlus 权重完整。如果第一条提示找不到 `voice_pipeline`，在同一目录执行一次：

```powershell
uv pip install --python $pipelinePython -e . --no-deps
```

然后启动 AudioMiner：

```powershell
Set-Location 'D:\Python_program_codes\AudioMiner(voice-clone)'
.\venv\Scripts\Activate.ps1
python .\voice_dataset_builder.py
```

在 AudioMiner 的“设置 → 训练模块”中点击“添加”，按当前目录结构填写：

```text
名称：GPT-SoVITS
项目目录：D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline
Python：D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe
模块入口：voice_pipeline
```

点击“检查连接”。出现 `GPT-SoVITS v2ProPlus` 后保存设置、关闭并重新启动 GUI。窗口名称会变为 `AudioClone Studio`，左侧出现“素材挖掘”和“训练”。启动时只有显式配置且握手成功的模块会进入训练页面；删除全部训练模块配置并重启后，会恢复 standalone 单页模式。

## 完整工作流

1. 在“素材挖掘”中生成 `dataset.list`，完成后点击“继续训练”；也可以在训练页直接选择已有的 `dataset.list`。
2. “项目目录”选择该目标人的数据目录。`dataset.list`、参考音频和输出目录都必须位于这个项目目录内；这是目标人之间隔离数据、缓存、任务和权重的边界。
3. “项目名称”使用字母、数字、下划线或连字符，例如 `Acane`；选择 `cuda:0`、`fp16`，并勾选预处理、S2、S1、自动评测。
4. 如果启用自动评测，选择项目目录内的参考音频，填写与音频一致的参考文本并选择对应语言。确认高级参数后点击“开始”。
5. 预处理、S1、S2、评测等阶段状态会显示在训练页；S1/S2 显示模块上报的实际进度。错误、协议异常和子进程输出会立即进入错误区和 Activity 日志。
6. 自动评测完成后，候选页只展示 `A`、`B`、`C`。每个候选提供中文、日文、英文试听；必须人工确认后才能晋升最终模型。

每次 GUI 任务的协议快照、事件和日志保存在所选目标人项目的 `jobs/<job_id>/` 下。训练模块自身的阶段缓存和 checkpoint 仍由训练项目管理；失败后使用相同目标人项目和输入重新启动，模块可按其状态继续。当前 GUI 不会自动删除失败任务、试听候选或训练输出。

## 语言切换

在设置中可切换中文、English、日本語。普通界面文案受语言系统管理；`GPT-SoVITS`、框架名、路径、`A/B/C` 等专有名称或技术标识保持不变。

> 本仓库只维护源码运行方式，不再提供或维护 PyInstaller/EXE 打包流程。

---

## 模型存储

GUI 中的 `Model storage root` 是模型总目录，例如：

```text
D:\AI_Cache\VoiceDatasetBuilder
```

程序内部会使用：

```text
D:\AI_Cache\VoiceDatasetBuilder\huggingface
D:\AI_Cache\VoiceDatasetBuilder\torch
```

其中：

- Hugging Face：faster-whisper 模型
- Torch：Demucs 模型

第一次真正使用模型时才会下载。

下载过程中如果网络中断，已有缓存会保留，重新 Retry 时不会主动删除已下载内容。

---

# 使用 CLI

CLI 入口：

```powershell
python build_dataset.py -h
```

## 本地音频

```powershell
python build_dataset.py input.m4a --speaker target --language ja
```

## YouTube

```powershell
python build_dataset.py "https://www.youtube.com/watch?v=..." --speaker target --language ja
```

## Bilibili

```powershell
python build_dataset.py "https://www.bilibili.com/video/BV..." --speaker target --language ja
```

## 已经是纯人声音频

可跳过 Demucs：

```powershell
python build_dataset.py vocals.wav --speaker target --language ja --skip-separation
```

默认 ASR 模型：

```text
large-v3-turbo
```

调试时可改为：

```powershell
--asr-model small
```

---

# 中断恢复与缓存

工具会自动保留可复用阶段结果。

任务失败或停止后，重新执行同一个 source 时，会尽量从最近可复用阶段继续，而不是从头处理。

运行过程中可能保留：

```text
source audio
separated vocals
normalized audio
ASR result
state information
```

任务成功后会自动执行 compact 清理：

保留：

```text
source audio
ASR JSON
state JSON
final clips
manifest.jsonl
dataset.list
```

删除体积较大的临时中间文件，例如：

```text
separated vocals
normalized full-length WAV
```

如需强制重新处理某个 source：

```powershell
--fresh
```

调试时希望保留全部中间文件：

```powershell
--keep-temp
```

---

# 输出结果

默认会生成：

```text
clips/
transcripts/
manifest.jsonl
dataset.list
```

## `clips/*.wav`

最终用于训练的短语音片段。

默认切片目标：

```text
推荐：4 ~ 8 秒
硬限制：3 ~ 12 秒
```

## `manifest.jsonl`

通用数据格式，例如：

```json
{"audio":"clips/sample_0001.wav","speaker":"target","language":"ja","text":"今日はいい天気ですね。","confidence":0.93}
```

## `dataset.list`

GPT-SoVITS 格式：

```text
ABSOLUTE_WAV_PATH|target|ja|今日はいい天気ですね。
```

---

# GPU 与 CPU

检测到 NVIDIA CUDA 时，程序会优先使用 GPU。

推荐正式构建使用：

```text
large-v3-turbo
```

无 NVIDIA GPU 时可以使用 CPU 模式。

如果 Demucs 显存不足，可使用：

```powershell
--separator-device cpu
```

如果 ASR 需要使用 CPU：

```powershell
--asr-device cpu
```

---

# 已知限制

当前版本**不处理**：

- 多说话人精确区分，请尽量保证视频/音频里只有一个主说话人
- Bilibili 会员 / 登录限制内容的自动认证，请保证已有自己的账号
- 自动读取浏览器 Cookie（安全第一）
- LLM 文本纠错（复杂语句偶现识别错误）
- 手工音频编辑 / 波形剪辑

欢迎大家下载下来自己根据需求魔改，待开发玩法应该还是挺多的！
e.g.可以仅提取音频，阶段缓存会保留的；或者拿来做乐器分轨提取......
---
