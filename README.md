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

## 注意事项

- **需要 NVIDIA 显卡 + CUDA 版 PyTorch** 才能 GPU 加速，CPU 也能跑但很慢
- 仅支持**钢琴独奏**音频，含人声或其它乐器的音频转录效果较差
- 原始模型来自 [GiantMIDI-Piano 论文](https://arxiv.org/abs/2010.07061) (Kong et al., 2020)

## License

CC BY 4.0
