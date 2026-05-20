# PianoSolo to MIDI - 钢琴转录工具

将钢琴独奏 mp3 音频转录为 MIDI 文件。

基于字节跳动的高分辨率钢琴转录系统 [piano_transcription_inference](https://github.com/bytedance/piano_transcription_inference)。

## 快速开始

### 1. 安装依赖

先安装 PyTorch（CUDA 版本，用于 GPU 加速）
* 去 [pytorch](https://pytorch.org/get-started/locally/) 选你的 CUDA 版本生成安装命令，例如：
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 再装其他依赖
pip install -r requirements.txt
```

### 2. 放入音频

把 `.mp3` 文件放到 `workspace/mp3s/` 目录。

### 3. 运行

双击 `Start.bat`，或手动执行：

```bash
python audios_to_midis.py transcribe_file \
    --input ./workspace/mp3s \
    --output ./workspace/midis
```

转录完成的 `.mid` 文件会输出到 `workspace/midis/`。

### 4. 单文件转录

```bash
python audios_to_midis.py transcribe_file --input ./workspace/mp3s/my_song.mp3 --output ./workspace/midis/my_song.mid
```

### 5. MIDI 音符对齐

转录后可以对 MIDI 音符进行对齐（类似量化），将几乎同时响起的音符的起始时间对齐，让节奏更紧凑：

```bash
python audios_to_midis.py transcribe_file \
    --input ./workspace/mp3s \
    --output ./workspace/midis \
    --align-strength 1.0 \
    --align-threshold 0.05
```

- `--align-strength`: 对齐强度，0.0 = 不做对齐（默认），1.0 = 完全对齐（组内音符完全同步），0.5 = 移动到一半位置。推荐从 0.5 开始尝试
- `--align-threshold`: 时间窗口（秒），在此窗口内的音符被视为一组并对齐。默认 0.05（50ms）

对齐会保持音符的时值不变（结束时间随起始时间同步偏移），踏板事件不受影响。

### 6. MIDI 多音轨声部分离

转录后可以使用 GNN 模型自动将单轨 MIDI 按声部拆分为多轨（如左右手分离）：

```bash
python audios_to_midis.py transcribe_file \
    --input ./workspace/mp3s \
    --output ./workspace/midis \
    --separate-voices
```

- `--separate-voices`: 启用声部分离，输出文件名为 `原文件名_separated.mid`
- `--svsep-model`: 预训练模型路径（默认使用内置模型，首次运行自动下载 34.8MB）
- `--sep-mode`: 分离模式（默认 `voice`）
  - `voice` — 按声部拆分，每个声部独立一轨（最多 4-6 轨）
  - `staff` — 按左右手拆分为两轨（Right Hand + Left Hand）
- 可同时使用 `--align-strength` 先对齐再分离
- 基于 [piano_svsep](https://github.com/CPJKU/piano_svsep)（ISMIR 2024 最佳论文提名），使用图神经网络进行声部/谱表分离

**voice 模式**输出 MIDI 包含多个音轨：
- Track 0: 速度和拍号
- Track 1-2: 上谱表声部（右手）
- Track 5-6: 下谱表声部（左手）
- Pedal Track: 踏板事件

**staff 模式**输出 MIDI 包含三个音轨：
- Track 0: 速度和拍号
- Track 1: Right Hand (upper staff)
- Track 2: Left Hand (lower staff)
- Pedal Track: 踏板事件

## 注意事项

- **需要 NVIDIA 显卡 + CUDA 版 PyTorch** 才能 GPU 加速，CPU 也能跑但很慢
- 仅支持**钢琴独奏**音频，含人声或其它乐器的音频转录效果较差
- 原始模型来自 [GiantMIDI-Piano 论文](https://arxiv.org/abs/2010.07061) (Kong et al., 2020)

## License

CC BY 4.0
