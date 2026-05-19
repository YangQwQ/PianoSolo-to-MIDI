# PianoSolo to MIDI - 钢琴转录工具

将钢琴独奏 mp3 音频转录为 MIDI 文件。

基于字节跳动的高分辨率钢琴转录系统 [piano_transcription_inference](https://github.com/bytedance/piano_transcription_inference)。

## 快速开始

### 1. 安装依赖

```bash
# 先安装 PyTorch（CUDA 版本，用于 GPU 加速）
# 去 https://pytorch.org/ 选你的 CUDA 版本生成安装命令，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu126

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
python audios_to_midis.py transcribe_file \
    --input ./workspace/mp3s/my_song.mp3 \
    --output ./workspace/midis/my_song.mid
```

## 注意事项

- **需要 NVIDIA 显卡 + CUDA 版 PyTorch** 才能 GPU 加速，CPU 也能跑但很慢
- 仅支持**钢琴独奏**音频，含人声或其它乐器的音频转录效果较差
- 原始模型来自 [GiantMIDI-Piano 论文](https://arxiv.org/abs/2010.07061) (Kong et al., 2020)

## License

CC BY 4.0
