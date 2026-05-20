"""MIDI note onset alignment (quantization)."""

from mido import Message, MidiFile, MidiTrack, MetaMessage


def parse_midi(midi_path):
    """Parse a MIDI file into note events, pedal events, and tempo info.

    Returns:
        note_events: list of dicts with keys 'onset', 'offset', 'note', 'velocity'
        pedal_events: list of dicts with keys 'onset', 'offset'
        tempo_info: dict with 'ticks_per_beat', 'microseconds_per_beat', 'ticks_per_second'
    """
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    # Read tempo from track 0
    microseconds_per_beat = 500000  # default 120 BPM
    for msg in midi_file.tracks[0]:
        if msg.type == 'set_tempo':
            microseconds_per_beat = msg.tempo
            break

    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second
    tempo_info = {
        'ticks_per_beat': ticks_per_beat,
        'microseconds_per_beat': microseconds_per_beat,
        'ticks_per_second': ticks_per_second,
    }

    note_events = []
    pedal_events = []
    pending_onsets = {}       # note -> {'onset': float, 'velocity': int}
    pending_pedal_onset = None

    abs_ticks = 0
    for msg in midi_file.tracks[1]:
        abs_ticks += msg.time
        abs_time = abs_ticks / ticks_per_second

        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.note in pending_onsets:
                # Overlapping same pitch: close the previous note now
                prev = pending_onsets.pop(msg.note)
                note_events.append({
                    'onset': prev['onset'], 'offset': abs_time,
                    'note': msg.note, 'velocity': prev['velocity'],
                })
            pending_onsets[msg.note] = {'onset': abs_time, 'velocity': msg.velocity}

        elif msg.type in ('note_on', 'note_off'):
            # note_on velocity=0 or note_off
            if msg.note in pending_onsets:
                prev = pending_onsets.pop(msg.note)
                note_events.append({
                    'onset': prev['onset'], 'offset': abs_time,
                    'note': msg.note, 'velocity': prev['velocity'],
                })

        elif msg.type == 'control_change' and msg.control == 64:
            if msg.value >= 64:
                pending_pedal_onset = abs_time
            else:
                if pending_pedal_onset is not None:
                    pedal_events.append({'onset': pending_pedal_onset, 'offset': abs_time})
                    pending_pedal_onset = None

    # Close any notes still pending at end of track
    for note, prev in pending_onsets.items():
        note_events.append({
            'onset': prev['onset'], 'offset': abs_time,
            'note': note, 'velocity': prev['velocity'],
        })

    return note_events, pedal_events, tempo_info


def group_notes(note_events, threshold):
    """Group notes whose onset times fall within `threshold` seconds of each other.

    Uses a first-note anchor window: the group starts with the earliest ungrouped
    note, and all subsequent notes within the window join the same group.

    Returns a list of groups, each group is a list of (index, event) tuples.
    """
    if not note_events:
        return []

    indexed = list(enumerate(note_events))
    indexed.sort(key=lambda x: x[1]['onset'])

    groups = []
    current_group = [indexed[0]]
    anchor = indexed[0][1]['onset']

    for item in indexed[1:]:
        if item[1]['onset'] - anchor <= threshold + 1e-9:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
            anchor = item[1]['onset']

    groups.append(current_group)
    return groups


def align_notes(note_events, strength, threshold):
    """Apply onset alignment to note events.

    Args:
        note_events: list of note event dicts
        strength: 0.0 = no change, 1.0 = full quantization
        threshold: time window in seconds for grouping

    Returns a new list of note event dicts with aligned onsets and offsets.
    Duration is preserved (offset shifts by the same delta as onset).
    """
    if strength <= 0 or not note_events:
        return [dict(e) for e in note_events]

    groups = group_notes(note_events, threshold)
    result = [dict(e) for e in note_events]

    for group in groups:
        if len(group) < 2:
            continue

        onsets = [item[1]['onset'] for item in group]
        target = sum(onsets) / len(onsets)

        for idx, event in group:
            delta = (target - event['onset']) * strength
            result[idx]['onset'] = event['onset'] + delta
            result[idx]['offset'] = event['offset'] + delta

    result.sort(key=lambda e: e['onset'])
    return result


def write_aligned_midi(note_events, pedal_events, tempo_info, midi_path):
    """Write aligned note events (and original pedal events) to a MIDI file."""
    ticks_per_beat = tempo_info['ticks_per_beat']
    microseconds_per_beat = tempo_info['microseconds_per_beat']
    ticks_per_second = tempo_info['ticks_per_second']

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0: tempo and time signature
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1: note and pedal events
    track1 = MidiTrack()
    message_roll = []

    for ev in note_events:
        message_roll.append({'time': ev['onset'], 'note': ev['note'], 'velocity': ev['velocity']})
        message_roll.append({'time': ev['offset'], 'note': ev['note'], 'velocity': 0})

    for ev in pedal_events:
        message_roll.append({'time': ev['onset'], 'control': 64, 'value': 127})
        message_roll.append({'time': ev['offset'], 'control': 64, 'value': 0})

    message_roll.sort(key=lambda m: m['time'])

    # Use the earliest event time as the reference start to avoid negative ticks
    if message_roll:
        start_time = message_roll[0]['time']
    else:
        start_time = 0.0

    previous_ticks = 0
    for msg in message_roll:
        this_ticks = int(round((msg['time'] - start_time) * ticks_per_second))
        diff_ticks = this_ticks - previous_ticks
        previous_ticks = this_ticks
        if 'note' in msg:
            track1.append(Message('note_on', note=msg['note'], velocity=msg['velocity'], time=diff_ticks))
        elif 'control' in msg:
            track1.append(Message('control_change', channel=0, control=msg['control'], value=msg['value'], time=diff_ticks))

    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)
    midi_file.save(midi_path)


def align_midi_file(midi_path, strength, threshold):
    """Parse, align, and rewrite a MIDI file in place."""
    note_events, pedal_events, tempo_info = parse_midi(midi_path)
    if not note_events:
        return
    aligned = align_notes(note_events, strength, threshold)
    write_aligned_midi(aligned, pedal_events, tempo_info, midi_path)
