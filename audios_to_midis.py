import os
import re
import builtins
from contextlib import contextmanager
import argparse

import numpy as np
import audioread
import librosa
import torch
import piano_transcription_inference
import piano_transcription_inference.utilities as piu
import midi_align
import midi_separate


# ── patch: 兼容新版 librosa（0.10+ 废弃了 core.audio API） ──
def _patched_load_audio(path, sr=22050, mono=True, offset=0.0, duration=None,
    dtype=np.float32, res_type='kaiser_best',
    backends=[audioread.ffdec.FFmpegAudioFile]):
    """Fixed load_audio using librosa 0.10+ API."""

    y = []
    with audioread.audio_open(os.path.realpath(path), backends=backends) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels
        s_end = np.inf if duration is None else \
            s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0
        for frame in input_file:
            frame = librosa.util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                continue
            if s_end < n_prev:
                break
            if s_end < n:
                frame = frame[:s_end - n_prev]
            if n_prev <= s_start <= n:
                frame = frame[(s_start - n_prev):]

            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = librosa.to_mono(y)
        if sr is not None:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=sr,
                                 res_type=res_type)
        else:
            sr = sr_native

    y = np.ascontiguousarray(y, dtype=dtype)
    return (y, sr)


piu.load_audio = _patched_load_audio
piano_transcription_inference.load_audio = _patched_load_audio


@contextmanager
def _progress_bar(filename=''):
    """Suppress segment-by-segment prints and show a single inline progress bar."""
    original_print = builtins.print
    state = {'total': 0, 'current': 0}

    def _print(*args, **kwargs):
        text = ' '.join(str(a) for a in args)
        m = re.match(r'Segment\s+(\d+)\s+/\s+(\d+)', text)
        if m:
            state['current'] = int(m.group(1))
            state['total'] = int(m.group(2))
            pct = state['current'] / max(state['total'], 1)
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '=' * filled + '-' * (bar_len - filled)
            original_print(
                f'\r  [{bar}] {state["current"]}/{state["total"]} 片段',
                end='', flush=True)
        elif text.startswith('Write out to '):
            # Suppress the library's own "Write out to" message
            pass
        else:
            original_print(*args, **kwargs)

    builtins.print = _print
    try:
        yield
    finally:
        builtins.print = original_print
        if state['total'] > 0:
            original_print()  # final newline after progress bar


def transcribe_file(args):
    """Transcribe mp3 file(s) to midi. Supports single file or directory input."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Collect mp3 paths
    if os.path.isdir(args.input):
        mp3_paths = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith('.mp3')
        ])
    else:
        mp3_paths = [args.input]

    if not mp3_paths:
        print('No mp3 files found')
        return

    # Output: if input is a directory, output is treated as a directory
    output_is_dir = os.path.isdir(args.input) or os.path.isdir(args.output) \
        or (not os.path.splitext(args.output)[1])
    if output_is_dir:
        os.makedirs(args.output, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f'设备: {device}, 文件数: {len(mp3_paths)}')

    transcriptor = piano_transcription_inference.PianoTranscription(device=device)

    for i, mp3_path in enumerate(mp3_paths):
        filename = os.path.basename(mp3_path)
        print(f'\n[{i + 1}/{len(mp3_paths)}] {filename}')

        if output_is_dir:
            midi_path = os.path.join(args.output,
                os.path.splitext(filename)[0] + '.mid')
        else:
            midi_path = args.output

        (audio, _) = piano_transcription_inference.load_audio(
            mp3_path, sr=piano_transcription_inference.sample_rate, mono=True)

        with _progress_bar(filename):
            transcriptor.transcribe(audio, midi_path)

        if args.align_strength > 0:
            midi_align.align_midi_file(midi_path, args.align_strength, args.align_threshold)

        if args.separate_voices:
            sep_path = os.path.splitext(midi_path)[0] + '_separated.mid'
            midi_separate.separate_midi_voices(midi_path, sep_path, args.svsep_model, device=device, mode=args.sep_mode)
            print(f'  -> {sep_path}')

        print(f'  -> {midi_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe mp3 to midi using piano_transcription_inference')
    subparsers = parser.add_subparsers(dest='mode')

    parser_transcribe_file = subparsers.add_parser('transcribe_file')
    parser_transcribe_file.add_argument('--input', type=str, required=True,
        help='mp3 file path or directory containing mp3 files')
    parser_transcribe_file.add_argument('--output', type=str, required=True,
        help='output midi file path or directory')
    parser_transcribe_file.add_argument('--align-strength', type=float, default=0.0,
        help='Align notes that start almost simultaneously. '
             '0.0 = no change, 1.0 = full quantization. Default: 0.0 (disabled)')
    parser_transcribe_file.add_argument('--align-threshold', type=float, default=0.05,
        help='Time window in seconds for grouping notes. Default: 0.05 (50ms)')
    parser_transcribe_file.add_argument('--separate-voices', action='store_true', default=False,
        help='Separate MIDI into multiple tracks by voice (using piano_svsep GNN)')
    parser_transcribe_file.add_argument('--svsep-model', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'piano_svsep_repo', 'pretrained_models', 'model.ckpt'),
        help='Path to piano_svsep model checkpoint')
    parser_transcribe_file.add_argument('--sep-mode', type=str, default='voice',
        choices=['voice', 'staff'],
        help='Separation mode: "voice" = one track per voice (default), '
             '"staff" = left/right hand only (2 tracks)')

    args = parser.parse_args()

    if args.mode == 'transcribe_file':
        transcribe_file(args)
    else:
        parser.print_help()
