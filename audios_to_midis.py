import os
import argparse
import torch
import piano_transcription_inference


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

    transcriptor = piano_transcription_inference.PianoTranscription(device=device)

    for mp3_path in mp3_paths:
        print('Processing:', mp3_path)

        if output_is_dir:
            midi_path = os.path.join(args.output,
                os.path.splitext(os.path.basename(mp3_path))[0] + '.mid')
        else:
            midi_path = args.output

        (audio, _) = piano_transcription_inference.load_audio(
            mp3_path, sr=piano_transcription_inference.sample_rate, mono=True)

        transcriptor.transcribe(audio, midi_path)
        print('Saved to:', midi_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe mp3 to midi using piano_transcription_inference')
    subparsers = parser.add_subparsers(dest='mode')

    parser_transcribe_file = subparsers.add_parser('transcribe_file')
    parser_transcribe_file.add_argument('--input', type=str, required=True,
        help='mp3 file path or directory containing mp3 files')
    parser_transcribe_file.add_argument('--output', type=str, required=True,
        help='output midi file path or directory')

    args = parser.parse_args()

    if args.mode == 'transcribe_file':
        transcribe_file(args)
    else:
        parser.print_help()
