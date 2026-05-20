"""MIDI voice/staff separation using piano_svsep GNN model."""

import os
import urllib.request
import numpy as np
import torch
import torch_geometric as pyg
import partitura as pt
from mido import Message, MidiFile, MidiTrack, MetaMessage

from piano_svsep.models.pl_models import PLPianoSVSep
from piano_svsep.utils import (
    hetero_graph_from_note_array,
    get_vocsep_features,
    score_graph_to_pyg,
    HeteroScoreGraph,
    remove_ties_acros_barlines,
    get_measurewise_pot_edges,
    get_pot_chord_edges,
    get_truth_chords_edges,
    get_measurewise_truth_edges,
    assign_voices,
)

from midi_align import parse_midi


def _prepare_score(score):
    """Prepare a partitura Score for voice separation (adapted from piano_svsep)."""
    if len(score) > 1:
        score = pt.score.Score(pt.score.merge_parts(score.parts))

    tie_couples = remove_ties_acros_barlines(score, return_ids=True)
    part = score[0]

    for beam in list(part.iter_all(pt.score.Beam)):
        for note in beam.notes:
            note.beam = None
        part.remove(beam)
    for rest in list(part.iter_all(pt.score.Rest)):
        part.remove(rest)
    for tuplet in list(part.iter_all(pt.score.Tuplet)):
        if isinstance(tuplet.start_note, pt.score.Rest) or isinstance(tuplet.end_note, pt.score.Rest):
            part.remove(tuplet)
    for gn in list(part.iter_all(pt.score.GraceNote)):
        part.remove(gn)

    note_array = part.note_array(
        include_time_signature=True,
        include_grace_notes=True,
        include_staff=True,
    )

    mn_map = part.measure_number_map
    note_measures = mn_map(note_array["onset_div"])

    nodes, edges = hetero_graph_from_note_array(note_array, pot_edge_dist=0)
    note_features = get_vocsep_features(note_array)
    hg = HeteroScoreGraph(note_features, edges, name="sep", labels=None, note_array=note_array)

    pot_edges = get_measurewise_pot_edges(note_array, note_measures)
    pot_chord_edges = get_pot_chord_edges(note_array, hg.get_edges_of_type("onset").numpy())
    setattr(hg, "pot_edges", torch.tensor(pot_edges))
    setattr(hg, "pot_chord_edges", torch.tensor(pot_chord_edges))

    truth_chords_edges = get_truth_chords_edges(note_array, pot_chord_edges)
    polyphonic_truth_edges = get_measurewise_truth_edges(note_array, note_measures)
    setattr(hg, "truth_chord_edges", torch.tensor(truth_chords_edges).long())
    setattr(hg, "truth_edges", torch.tensor(polyphonic_truth_edges).long())

    pg_graph = score_graph_to_pyg(hg)
    return pg_graph, score, tie_couples


def _get_voice_labels(midi_path, model, device='cpu'):
    """Run GNN prediction and return per-note voice/staff assignments from partitura.

    predict_step returns (edge_indices, staff_labels), not per-note voice IDs.
    We call assign_voices() to propagate edge info into per-note voice assignments,
    then extract them from the partitura note_array.

    Returns:
        voice_labels: np.ndarray of voice numbers (one per note, in partitura order)
        staff_labels: np.ndarray of staff numbers
        note_array: the partitura note_array with onset/pitch for matching
    """
    score = pt.load_score(midi_path, force_note_ids=True)
    pg_graph, score, _tied_notes = _prepare_score(score)
    pg_graph = pyg.data.Batch.from_data_list([pg_graph])
    pg_graph = pg_graph.to(device)

    model.module.eval()
    with torch.no_grad():
        # predict_step returns (pred_edges, staff_logits) — edges connect notes in same voice
        pred_edges, pred_staff = model.predict_step(pg_graph)

    part = score[0]
    assign_voices(part, pred_edges, pred_staff)

    na = part.note_array(include_staff=True)
    voice_labels = na["voice"].astype(int)
    staff_labels = na["staff"].astype(int)

    return voice_labels, staff_labels, na


MODEL_URL = 'https://github.com/CPJKU/piano_svsep/raw/main/pretrained_models/model.ckpt'


def _download_model(model_path):
    """Download the pretrained model checkpoint if missing."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f'  Downloading model checkpoint (34.8 MB)...')
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f'  Model saved to {model_path}')


def separate_midi_voices(midi_path, output_path, model_path, device='cpu', mode='voice'):
    """Separate a single-track piano MIDI into multiple tracks.

    Args:
        midi_path: path to input MIDI file
        output_path: path for output multi-track MIDI
        model_path: path to piano_svsep model checkpoint
        device: 'cpu' or 'cuda'
        mode: 'voice' = one track per voice (4-6 tracks),
              'staff' = one track per staff / left+right hand (2 tracks)
    """
    if not os.path.exists(model_path):
        _download_model(model_path)

    pl_model = PLPianoSVSep.load_from_checkpoint(
        model_path, map_location=device, strict=False, weights_only=False)

    # Get voice labels from GNN
    voice_labels, staff_labels, pt_notes = _get_voice_labels(midi_path, pl_model, device)
    print(f'  GNN predicted {len(voice_labels)} notes, '
          f'{len(np.unique(voice_labels))} voices')

    # Parse MIDI with mido for accurate timing
    note_events, pedal_events, tempo_info = parse_midi(midi_path)

    if len(note_events) == 0:
        print('  No notes to separate')
        return

    # Match partitura notes with mido notes by (pitch, onset_time)
    ticks_per_second = tempo_info['ticks_per_second']

    # Build lookup from partitura: (pitch, rounded onset_sec) -> voice, staff
    # partitura uses onset_div, convert to seconds via ticks_per_second
    voice_map = {}
    for i in range(len(pt_notes)):
        onset_sec = float(pt_notes[i]['onset_div']) / ticks_per_second
        pitch = int(pt_notes[i]['pitch'])
        key = (pitch, round(onset_sec, 3))
        voice_map[key] = (int(voice_labels[i]), int(staff_labels[i]))

    # Match mido notes to partitura voices
    matched = 0
    for ev in note_events:
        key = (ev['note'], round(ev['onset'], 3))
        if key in voice_map:
            ev['voice'], ev['staff'] = voice_map[key]
            matched += 1
        else:
            ev['voice'] = 1
            ev['staff'] = 1

    if matched < len(note_events):
        print(f'  Note: matched {matched}/{len(note_events)} notes with voice labels')

    _write_multitrack_midi(note_events, pedal_events, tempo_info, output_path, mode)
    voices_found = set(ev.get('voice', 1) for ev in note_events)
    if mode == 'staff':
        print(f'  Mode: staff → {len(voices_found)} voice(s) grouped into 2 tracks (left/right hand)')
    else:
        print(f'  Output: {len(voices_found)} tracks (voices: {sorted(voices_found)})')


def _write_multitrack_midi(note_events, pedal_events, tempo_info, midi_path, mode='voice'):
    """Write note events grouped by voice or staff to separate MIDI tracks.

    Args:
        mode: 'voice' = one track per voice, 'staff' = one track per staff (left/right hand)
    """
    ticks_per_beat = tempo_info['ticks_per_beat']
    microseconds_per_beat = tempo_info['microseconds_per_beat']
    ticks_per_second = tempo_info['ticks_per_second']

    if mode == 'staff':
        # Group by staff: staff 1 = upper (right hand), staff 2 = lower (left hand)
        groups = {}
        for ev in note_events:
            s = ev.get('staff', 1)
            groups.setdefault(s, []).append(ev)
        group_names = {
            1: 'Right Hand (upper staff)',
            2: 'Left Hand (lower staff)',
        }
    else:
        # Group by individual voice
        groups = {}
        for ev in note_events:
            v = ev.get('voice', 1)
            groups.setdefault(v, []).append(ev)
        group_names = {
            1: 'Voice 1 (upper staff)',
            2: 'Voice 2 (upper staff)',
            5: 'Voice 5 (lower staff)',
            6: 'Voice 6 (lower staff)',
        }

    sorted_groups = sorted(groups.keys())
    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0: tempo and time signature
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # All tracks share time origin 0 so notes keep their absolute positions
    # across tracks. Each track's first note gets a positive delta-time equal
    # to its absolute onset, which is valid in MIDI.
    global_start = 0.0

    for g in sorted_groups:
        track = MidiTrack()
        name = group_names.get(g, f'Voice {g}')
        track.append(MetaMessage('track_name', name=name, time=0))

        message_roll = []
        for ev in groups[g]:
            message_roll.append({'time': ev['onset'], 'note': ev['note'], 'velocity': ev['velocity']})
            message_roll.append({'time': ev['offset'], 'note': ev['note'], 'velocity': 0})

        message_roll.sort(key=lambda m: m['time'])

        previous_ticks = 0
        for msg in message_roll:
            this_ticks = int(round((msg['time'] - global_start) * ticks_per_second))
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            track.append(Message('note_on', note=msg['note'], velocity=msg['velocity'], time=diff_ticks))

        track.append(MetaMessage('end_of_track', time=1))
        midi_file.tracks.append(track)

    if pedal_events:
        track = MidiTrack()
        track.append(MetaMessage('track_name', name='Pedal', time=0))
        message_roll = []
        for ev in pedal_events:
            message_roll.append({'time': ev['onset'], 'control': 64, 'value': 127})
            message_roll.append({'time': ev['offset'], 'control': 64, 'value': 0})
        message_roll.sort(key=lambda m: m['time'])

        previous_ticks = 0
        for msg in message_roll:
            this_ticks = int(round((msg['time'] - global_start) * ticks_per_second))
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            track.append(Message('control_change', channel=0, control=msg['control'],
                                 value=msg['value'], time=diff_ticks))

        track.append(MetaMessage('end_of_track', time=1))
        midi_file.tracks.append(track)

    midi_file.save(midi_path)
