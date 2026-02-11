"""
Synthetic audio generator with hierarchical temporal + frequency annotations.

Generates random multi-instrument songs, renders them to audio, computes mel
spectrograms, and produces hierarchical bounding-box annotations for training
an audio OCR model.

Hierarchy
─────────
  full_mix
  ├── instrument_part  (one instrument's temporal span + frequency signature)
  │   ├── phrase        (temporally adjacent notes grouped together)
  │   │   ├── note      (single sound event: start/end frames + freq mask)
  │   │   └── …
  │   └── …
  └── …

Each node carries:
    start_frame, end_frame  — temporal bounds on the mel spectrogram
    freq_mask               — boolean vector (n_mels,) of active mel bins

Two synthesis back-ends:
    1. SimpleSynth   — additive synthesis with ADSR envelopes (default, zero deps)
    2. ArturiaSynth  — renders through Arturia VST plug-ins via DawDreamer (optional)

Usage:
    python3.10 generate_audio.py --num_songs 50 --out_dir data/audio --visualise
"""

import os
import json
import math
import wave
import struct
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from retina import fit_to_retina, image_to_tensor

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SPEC_FLOOR_DB = -80.0          # silence floor in dB

AUDIO_CLASSES = [
    'none',              # 0 — nothing left to detect
    'full_mix',          # 1
    'instrument_part',   # 2
    'phrase',            # 3
    'note',              # 4
]
AUDIO_CLASS_TO_ID = {c: i for i, c in enumerate(AUDIO_CLASSES)}

INSTRUMENT_NAMES = ['piano', 'strings', 'organ', 'brass']


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class NoteEvent:
    pitch: int             # MIDI note number (e.g. 60 = C4)
    start_sec: float       # onset time in seconds
    duration_sec: float    # note duration in seconds
    velocity: int          # 0–127
    instrument: str        # timbre / instrument name

    @property
    def end_sec(self):
        return self.start_sec + self.duration_sec

    @property
    def freq_hz(self):
        return 440.0 * 2.0 ** ((self.pitch - 69) / 12.0)


@dataclass
class AudioBBox:
    """Hierarchical annotation node for a region of the spectrogram."""
    start_frame: int
    end_frame: int
    freq_mask: list                          # bool list, length = n_mels
    label: str
    children: List['AudioBBox'] = field(default_factory=list)

    def to_dict(self):
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'freq_mask': self.freq_mask,
            'label': self.label,
            'children': [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d):
        return AudioBBox(
            start_frame=d['start_frame'],
            end_frame=d['end_frame'],
            freq_mask=d['freq_mask'],
            label=d['label'],
            children=[AudioBBox.from_dict(c) for c in d.get('children', [])],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Mel spectrogram (pure numpy — no librosa dependency)
# ──────────────────────────────────────────────────────────────────────────────
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    mel_lo = hz_to_mel(fmin)
    mel_hi = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # (n_fft//2 + 1,)
    fb = np.zeros((n_mels, len(freq_bins)))

    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (freq_bins >= lo) & (freq_bins <= ctr)
        dn = (freq_bins > ctr) & (freq_bins <= hi)
        if ctr > lo:
            fb[i, up] = (freq_bins[up] - lo) / (ctr - lo)
        if hi > ctr:
            fb[i, dn] = (hi - freq_bins[dn]) / (hi - ctr)
    return fb


def compute_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE,
                            n_fft: int = N_FFT, hop: int = HOP_LENGTH,
                            n_mels: int = N_MELS) -> np.ndarray:
    """
    Return mel spectrogram in dB, shape (n_mels, n_frames).
    """
    # zero-pad so we get at least 1 frame
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    n_frames = 1 + (len(audio) - n_fft) // hop
    window = np.hanning(n_fft)

    power = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        frame = audio[s:s + n_fft] * window
        spectrum = np.fft.rfft(frame)
        power[:, i] = np.abs(spectrum) ** 2

    # floor to avoid log(0); suppress benign overflow in filterbank matmul
    np.clip(power, 1e-20, None, out=power)

    mel_fb = _mel_filterbank(sr, n_fft, n_mels)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mel_power = mel_fb @ power
    np.nan_to_num(mel_power, nan=1e-10, posinf=1e-10, neginf=1e-10,
                  copy=False)
    np.clip(mel_power, 1e-10, None, out=mel_power)
    mel_db = 10.0 * np.log10(mel_power)
    return mel_db


# ──────────────────────────────────────────────────────────────────────────────
# Built-in additive synthesiser
# ──────────────────────────────────────────────────────────────────────────────
TIMBRES = {
    'piano': {
        'harmonics': [(1, 1.0), (2, 0.50), (3, 0.25), (4, 0.10), (5, 0.05)],
        'attack': 0.005, 'decay': 0.20, 'sustain': 0.60, 'release': 0.30,
    },
    'strings': {
        'harmonics': [(1, 1.0), (2, 0.80), (3, 0.65), (4, 0.50),
                      (5, 0.35), (6, 0.25), (7, 0.15)],
        'attack': 0.08, 'decay': 0.10, 'sustain': 0.80, 'release': 0.50,
    },
    'organ': {
        'harmonics': [(1, 1.0), (2, 1.0), (3, 0.50), (4, 0.50),
                      (5, 0.25), (6, 0.25)],
        'attack': 0.01, 'decay': 0.01, 'sustain': 1.0, 'release': 0.05,
    },
    'brass': {
        'harmonics': [(1, 1.0), (2, 0.70), (3, 0.85), (4, 0.50),
                      (5, 0.60), (6, 0.30)],
        'attack': 0.03, 'decay': 0.15, 'sustain': 0.70, 'release': 0.15,
    },
}


class SimpleSynth:
    """Additive synthesis with per-instrument timbre and ADSR envelope."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr

    def render_note(self, event: NoteEvent) -> np.ndarray:
        """Render a single note to a mono audio array (full song length not
        needed — just the note duration + release tail)."""
        t_info = TIMBRES.get(event.instrument, TIMBRES['piano'])
        release = t_info['release']
        total_dur = event.duration_sec + release
        n_samples = int(total_dur * self.sr)
        t = np.arange(n_samples, dtype=np.float64) / self.sr

        # additive harmonics
        sig = np.zeros(n_samples, dtype=np.float64)
        for h, amp in t_info['harmonics']:
            freq = event.freq_hz * h
            if freq >= self.sr / 2:
                break
            sig += amp * np.sin(2.0 * np.pi * freq * t)

        # ADSR envelope
        a = int(t_info['attack'] * self.sr)
        d = int(t_info['decay'] * self.sr)
        s_lvl = t_info['sustain']
        r = int(release * self.sr)
        sus = max(0, int(event.duration_sec * self.sr) - a - d)

        env = np.zeros(n_samples, dtype=np.float64)
        # attack
        end_a = min(a, n_samples)
        env[:end_a] = np.linspace(0, 1, end_a)
        # decay
        end_d = min(a + d, n_samples)
        env[end_a:end_d] = np.linspace(1, s_lvl, end_d - end_a)
        # sustain
        end_s = min(a + d + sus, n_samples)
        env[end_d:end_s] = s_lvl
        # release
        end_r = min(end_s + r, n_samples)
        if end_r > end_s:
            env[end_s:end_r] = np.linspace(s_lvl, 0, end_r - end_s)

        return sig * env * (event.velocity / 127.0)

    def render_song(self, events: List[NoteEvent],
                    duration_sec: float) -> np.ndarray:
        """Mix all notes into a single mono array of length duration_sec."""
        n_total = int(duration_sec * self.sr)
        mix = np.zeros(n_total, dtype=np.float64)
        for ev in events:
            note_audio = self.render_note(ev)
            start_sample = int(ev.start_sec * self.sr)
            end_sample = min(start_sample + len(note_audio), n_total)
            seg_len = end_sample - start_sample
            mix[start_sample:end_sample] += note_audio[:seg_len]
        # normalise
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.9
        return mix


# ──────────────────────────────────────────────────────────────────────────────
# DawDreamer / Arturia back-end (optional)
# ──────────────────────────────────────────────────────────────────────────────
class ArturiaSynth:
    """
    Render notes through Arturia (or any) VST plug-ins using DawDreamer.

    Requires:
        pip install dawdreamer

    Provide a mapping  instrument_name → VST path, e.g.:
        {
            'piano':   '/Library/Audio/Plug-Ins/VST3/Arturia/Piano V3.vst3',
            'strings': '/Library/Audio/Plug-Ins/VST3/Arturia/Analog Lab V.vst3',
        }
    """

    def __init__(self, vst_paths: Dict[str, str], sr: int = SAMPLE_RATE,
                 bpm: float = 120.0, buffer_size: int = 512):
        try:
            import dawdreamer as daw
        except ImportError:
            raise ImportError(
                'DawDreamer is required for ArturiaSynth.  '
                'Install with:  pip install dawdreamer')
        self.daw = daw
        self.sr = sr
        self.bpm = bpm
        self.buffer_size = buffer_size
        self.vst_paths = vst_paths

    def _seconds_to_beats(self, sec: float) -> float:
        return sec * self.bpm / 60.0

    def render_note(self, event: NoteEvent) -> np.ndarray:
        """Render a single note in isolation through the VST."""
        vst_path = self.vst_paths.get(event.instrument)
        if vst_path is None:
            raise ValueError(f'No VST path configured for "{event.instrument}"')

        engine = self.daw.RenderEngine(self.sr, self.buffer_size)
        synth = engine.make_plugin_processor('synth', vst_path)

        start_beat = 0.0
        dur_beats = self._seconds_to_beats(event.duration_sec)
        synth.add_midi_note(event.pitch, event.velocity,
                            start_beat, dur_beats)

        engine.load_graph([(synth, [])])
        render_dur = event.duration_sec + 1.0   # extra for release
        engine.render(render_dur)

        audio = engine.get_audio()              # (channels, samples)
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio
        return mono.astype(np.float64)

    def render_song(self, events: List[NoteEvent],
                    duration_sec: float) -> np.ndarray:
        """Render all notes by instrument, mixing the results."""
        # Group events by instrument
        by_inst: Dict[str, List[NoteEvent]] = {}
        for ev in events:
            by_inst.setdefault(ev.instrument, []).append(ev)

        n_total = int(duration_sec * self.sr)
        mix = np.zeros(n_total, dtype=np.float64)

        for inst, inst_events in by_inst.items():
            vst_path = self.vst_paths.get(inst)
            if vst_path is None:
                continue

            engine = self.daw.RenderEngine(self.sr, self.buffer_size)
            synth = engine.make_plugin_processor('synth', vst_path)

            for ev in inst_events:
                start_beat = self._seconds_to_beats(ev.start_sec)
                dur_beats = self._seconds_to_beats(ev.duration_sec)
                synth.add_midi_note(ev.pitch, ev.velocity,
                                    start_beat, dur_beats)

            engine.set_bpm(self.bpm)
            engine.load_graph([(synth, [])])
            engine.render(duration_sec)

            audio = engine.get_audio()
            mono = audio.mean(axis=0) if audio.ndim > 1 else audio
            seg_len = min(len(mono), n_total)
            mix[:seg_len] += mono[:seg_len].astype(np.float64)

        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.9
        return mix


# ──────────────────────────────────────────────────────────────────────────────
# Random song generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_random_song(num_instruments: int = 2,
                         duration_sec: float = 8.0,
                         notes_per_instrument: Tuple[int, int] = (5, 15),
                         pitch_range: Tuple[int, int] = (48, 84),
                         seed: int = None) -> Tuple[List[NoteEvent], float]:
    """
    Generate random note events for several instruments.
    Returns (events, total_duration_sec).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_instruments = min(num_instruments, len(INSTRUMENT_NAMES))
    instruments = random.sample(INSTRUMENT_NAMES, num_instruments)

    events: List[NoteEvent] = []
    max_end = 0.0

    for inst in instruments:
        n_notes = random.randint(*notes_per_instrument)
        t = random.uniform(0.0, 0.5)          # start offset

        for _ in range(n_notes):
            pitch = random.randint(*pitch_range)
            dur = random.uniform(0.2, 1.5)
            vel = random.randint(60, 120)

            events.append(NoteEvent(
                pitch=pitch, start_sec=round(t, 4),
                duration_sec=round(dur, 4), velocity=vel,
                instrument=inst))

            max_end = max(max_end, t + dur)
            # advance time (some overlap allowed)
            gap = random.uniform(-0.1, 0.6)
            t += dur + gap
            t = max(0, t)

    total_dur = max(duration_sec, max_end + 0.5)
    return events, round(total_dur, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Frequency-mask computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_note_freq_mask(note_audio: np.ndarray,
                           sr: int = SAMPLE_RATE,
                           n_fft: int = N_FFT,
                           hop: int = HOP_LENGTH,
                           n_mels: int = N_MELS,
                           threshold_db: float = 15.0) -> List[bool]:
    """
    Compute which mel bins are significantly active for a solo-rendered note.
    Returns a boolean list of length n_mels.
    """
    spec = compute_mel_spectrogram(note_audio, sr, n_fft, hop, n_mels)
    # average energy across time
    freq_profile = spec.mean(axis=1)                  # (n_mels,)
    floor = freq_profile.min()
    mask = (freq_profile > floor + threshold_db).tolist()
    return mask


def compute_note_temporal_bounds(note_audio: np.ndarray,
                                 start_sec: float,
                                 sr: int = SAMPLE_RATE,
                                 hop: int = HOP_LENGTH,
                                 threshold_db: float = 10.0) -> Tuple[int, int]:
    """
    Compute start_frame and end_frame of audible energy for a rendered note,
    relative to the global spectrogram timeline.
    """
    spec = compute_mel_spectrogram(note_audio, sr, hop=hop)
    # energy per frame
    energy = spec.mean(axis=0)                        # (n_frames,)
    floor = energy.min()
    active = energy > floor + threshold_db

    if not active.any():
        # fallback: use MIDI timing
        sf = int(start_sec * sr / hop)
        ef = sf + spec.shape[1]
        return sf, ef

    indices = np.where(active)[0]
    local_start = int(indices[0])
    local_end = int(indices[-1]) + 1

    global_offset = int(start_sec * sr / hop)
    return global_offset + local_start, global_offset + local_end


# ──────────────────────────────────────────────────────────────────────────────
# Phrase splitting (temporal proximity)
# ──────────────────────────────────────────────────────────────────────────────

def _group_into_phrases(note_bboxes: List[AudioBBox],
                        events: List[NoteEvent],
                        gap_sec: float = 0.5,
                        hop: int = HOP_LENGTH,
                        sr: int = SAMPLE_RATE) -> List[AudioBBox]:
    """
    Group note AudioBBoxes into phrases.  A new phrase starts whenever the
    temporal gap between consecutive notes exceeds `gap_sec`.
    """
    if not note_bboxes:
        return []

    gap_frames = int(gap_sec * sr / hop)

    # sort by start_frame
    indexed = sorted(enumerate(note_bboxes), key=lambda x: x[1].start_frame)

    phrases: List[AudioBBox] = []
    cur_notes = [indexed[0][1]]

    for i in range(1, len(indexed)):
        prev = indexed[i - 1][1]
        curr = indexed[i][1]
        if curr.start_frame - prev.end_frame > gap_frames:
            phrases.append(_make_phrase(cur_notes))
            cur_notes = [curr]
        else:
            cur_notes.append(curr)

    if cur_notes:
        phrases.append(_make_phrase(cur_notes))

    return phrases


def _make_phrase(notes: List[AudioBBox]) -> AudioBBox:
    start = min(n.start_frame for n in notes)
    end = max(n.end_frame for n in notes)
    # union of frequency masks
    n_mels = len(notes[0].freq_mask)
    union_mask = [False] * n_mels
    for n in notes:
        for j in range(n_mels):
            union_mask[j] = union_mask[j] or n.freq_mask[j]
    return AudioBBox(start, end, union_mask, 'phrase', children=list(notes))


# ──────────────────────────────────────────────────────────────────────────────
# Annotation tree builder
# ──────────────────────────────────────────────────────────────────────────────

def build_annotation_tree(events: List[NoteEvent],
                          synth,
                          total_frames: int,
                          sr: int = SAMPLE_RATE,
                          hop: int = HOP_LENGTH,
                          n_mels: int = N_MELS) -> AudioBBox:
    """
    Render each note in isolation, compute its freq_mask and temporal bounds,
    then build the hierarchical AudioBBox tree.
    """
    # ── per-note annotations ──
    note_bboxes_by_inst: Dict[str, List[Tuple[AudioBBox, NoteEvent]]] = {}

    for ev in events:
        note_audio = synth.render_note(ev)
        fmask = compute_note_freq_mask(note_audio, sr, N_FFT, hop, n_mels)
        sf, ef = compute_note_temporal_bounds(note_audio, ev.start_sec,
                                              sr, hop)
        sf = max(0, sf)
        ef = min(total_frames, ef)

        nb = AudioBBox(sf, ef, fmask, 'note')
        note_bboxes_by_inst.setdefault(ev.instrument, []).append((nb, ev))

    # ── instrument parts ──
    inst_parts: List[AudioBBox] = []
    for inst, items in note_bboxes_by_inst.items():
        note_bboxes = [nb for nb, _ in items]
        note_events = [ev for _, ev in items]

        phrases = _group_into_phrases(note_bboxes, note_events,
                                      gap_sec=0.5, hop=hop, sr=sr)

        # instrument envelope
        inst_start = min(p.start_frame for p in phrases)
        inst_end = max(p.end_frame for p in phrases)
        inst_mask = [False] * n_mels
        for p in phrases:
            for j in range(n_mels):
                inst_mask[j] = inst_mask[j] or p.freq_mask[j]

        inst_parts.append(AudioBBox(
            inst_start, inst_end, inst_mask,
            'instrument_part', children=phrases))

    # ── full mix (root) ──
    all_mask = [True] * n_mels
    root = AudioBBox(0, total_frames, all_mask, 'full_mix',
                     children=inst_parts)
    return root


# ──────────────────────────────────────────────────────────────────────────────
# Teacher-forcing training-sample extraction
# ──────────────────────────────────────────────────────────────────────────────

def spectrogram_to_image(spec: np.ndarray, vmin: float = SPEC_FLOOR_DB,
                         vmax: float = 0.0) -> Image.Image:
    """
    Convert a mel spectrogram (n_mels × n_frames, dB) to a PIL RGB image.
    Low frequencies at the bottom (image convention: row 0 = top → flip).
    """
    normed = (spec - vmin) / max(vmax - vmin, 1e-6)
    normed = np.clip(normed, 0.0, 1.0)
    normed = np.flipud(normed)                        # low freq → bottom
    grey = (normed * 255).astype(np.uint8)
    return Image.fromarray(np.stack([grey, grey, grey], axis=-1))


def extract_audio_training_samples(spec: np.ndarray,
                                   root: AudioBBox,
                                   retina_size: int = 1024
                                   ) -> List[dict]:
    """
    Walk the AudioBBox tree and produce teacher-forcing training tuples.

    Mirrors the sheet-music approach:
      • At each node, iterate children (sorted by start_frame).
      • Target = child's (start_norm, end_norm, freq_mask, class).
      • Mask child's freq bins in [start:end] on the working spectrogram.
      • After all children → 'none' sentinel.
      • Zoom into each child (crop time, apply freq_mask) and recurse.

    Each sample dict:
        spec_image     — PIL Image of the current spectrogram view
        target_start   — float in [0, 1]
        target_end     — float in [0, 1]
        target_mask    — list[float] of length n_mels (0 or 1)
        target_class   — string label
    """
    samples: List[dict] = []
    _audio_traverse(spec, root, samples)
    return samples


def _audio_traverse(spec: np.ndarray, node: AudioBBox, samples: list):
    if not node.children:
        return

    n_mels, n_frames = spec.shape
    sf, ef = node.start_frame, node.end_frame
    sf = max(0, sf)
    ef = min(n_frames, ef)
    if ef <= sf:
        return

    # Crop to node region and apply node's own mask
    node_spec = spec[:, sf:ef].copy()
    node_mask = np.array(node.freq_mask, dtype=bool)
    node_spec[~node_mask, :] = SPEC_FLOOR_DB
    node_frames = ef - sf

    children = sorted(node.children, key=lambda c: c.start_frame)
    working = node_spec.copy()

    for child in children:
        # Normalised temporal bounds relative to node crop
        t_start = max(0.0, min(1.0, (child.start_frame - sf) / node_frames))
        t_end = max(0.0, min(1.0, (child.end_frame - sf) / node_frames))

        # Frequency mask as float list
        fmask_float = [1.0 if m else 0.0 for m in child.freq_mask]

        samples.append({
            'spec_image': spectrogram_to_image(working),
            'target_start': t_start,
            'target_end': t_end,
            'target_mask': fmask_float,
            'target_class': child.label,
        })

        # Mask child out of working spectrogram
        c_mask = np.array(child.freq_mask, dtype=bool)
        c_sf = max(0, child.start_frame - sf)
        c_ef = min(node_frames, child.end_frame - sf)
        working[c_mask, c_sf:c_ef] = SPEC_FLOOR_DB

    # Sentinel: nothing left
    samples.append({
        'spec_image': spectrogram_to_image(working),
        'target_start': 0.0,
        'target_end': 0.0,
        'target_mask': [0.0] * n_mels,
        'target_class': 'none',
    })

    # Recurse into each child (zoom in)
    for child in children:
        c_sf = max(0, child.start_frame - sf)
        c_ef = min(node_frames, child.end_frame - sf)
        if c_ef <= c_sf:
            continue
        zoom = node_spec[:, c_sf:c_ef].copy()
        # keep only child's freq bins
        c_mask = np.array(child.freq_mask, dtype=bool)
        zoom[~c_mask, :] = SPEC_FLOOR_DB
        # Build a sub-AudioBBox with children offset to local time
        local_children = []
        for gc in child.children:
            local_sf = max(0, gc.start_frame - child.start_frame)
            local_ef = max(0, gc.end_frame - child.start_frame)
            local_ef = min(local_ef, c_ef - c_sf)
            local_children.append(AudioBBox(
                local_sf, local_ef, gc.freq_mask, gc.label,
                children=gc.children))
        local_node = AudioBBox(
            0, zoom.shape[1], child.freq_mask, child.label,
            children=local_children)
        _audio_traverse(zoom, local_node, samples)


# ──────────────────────────────────────────────────────────────────────────────
# WAV I/O (stdlib only)
# ──────────────────────────────────────────────────────────────────────────────

def save_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    pcm = (audio * 32767).clip(-32767, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
]


def draw_audio_bbox_tree(spec_img: Image.Image, root: AudioBBox,
                         n_frames: int, depth: int = 0,
                         max_depth: int = 99) -> Image.Image:
    """Overlay temporal regions + frequency highlights on a spectrogram image."""
    overlay = spec_img.copy().convert('RGBA')
    n_mels = spec_img.height
    _draw_audio_recursive(overlay, root, n_frames, n_mels, depth, max_depth)
    return overlay.convert('RGB')


def _draw_audio_recursive(overlay, node, n_frames, n_mels,
                          depth, max_depth):
    if depth > max_depth:
        return
    colour = PALETTE[depth % len(PALETTE)]
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = overlay.size
    x1 = int(node.start_frame / max(n_frames, 1) * img_w)
    x2 = int(node.end_frame / max(n_frames, 1) * img_w)
    x1, x2 = max(0, x1), min(img_w - 1, x2)

    # ensure valid rectangle (x2 > x1)
    if x2 <= x1:
        x2 = x1 + 1

    # draw temporal bounds as a rectangle spanning the image height
    draw.rectangle([x1, 0, x2, img_h - 1], outline=colour + (200,), width=2)

    # highlight active freq bins with semi-transparent overlay
    mask = node.freq_mask
    for j, active in enumerate(mask):
        if active:
            y = img_h - 1 - j   # flip (low freq = bottom)
            draw.line([(x1, y), (x2, y)], fill=colour + (40,), width=1)

    for child in node.children:
        _draw_audio_recursive(overlay, child, n_frames, n_mels,
                              depth + 1, max_depth)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic audio with hierarchical annotations.')
    parser.add_argument('--num_songs', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='data/audio')
    parser.add_argument('--num_instruments', type=int, default=2)
    parser.add_argument('--duration', type=float, default=8.0,
                        help='Target duration in seconds.')
    parser.add_argument('--retina_size', type=int, default=1024)
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backend', type=str, default='simple',
                        choices=['simple', 'arturia'],
                        help='Synthesis back-end.')
    parser.add_argument('--vst_config', type=str, default=None,
                        help='JSON file mapping instrument names to VST paths '
                             '(required for --backend arturia).')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'wav'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'spec'), exist_ok=True)
    if args.visualise:
        os.makedirs(os.path.join(args.out_dir, 'vis'), exist_ok=True)

    # Synthesis back-end
    if args.backend == 'arturia':
        if args.vst_config is None:
            parser.error('--vst_config required for arturia backend')
        with open(args.vst_config) as f:
            vst_paths = json.load(f)
        synth = ArturiaSynth(vst_paths)
    else:
        synth = SimpleSynth()

    annotations = []
    total_samples = 0

    for i in range(args.num_songs):
        events, dur = generate_random_song(
            num_instruments=args.num_instruments,
            duration_sec=args.duration,
            seed=args.seed + i)

        # Render full mix
        mix_audio = synth.render_song(events, dur)
        wav_name = f'song_{i:05d}.wav'
        save_wav(os.path.join(args.out_dir, 'wav', wav_name), mix_audio)

        # Mel spectrogram
        full_spec = compute_mel_spectrogram(mix_audio)
        n_frames = full_spec.shape[1]

        # Save spectrogram image
        spec_img = spectrogram_to_image(full_spec)
        spec_name = f'song_{i:05d}.png'
        spec_img.save(os.path.join(args.out_dir, 'spec', spec_name))

        # Build annotation tree
        tree = build_annotation_tree(events, synth, n_frames)

        ann = {
            'wav': wav_name,
            'spec': spec_name,
            'n_frames': n_frames,
            'n_mels': N_MELS,
            'duration_sec': dur,
            'sample_rate': SAMPLE_RATE,
            'hop_length': HOP_LENGTH,
            'bbox_tree': tree.to_dict(),
        }
        annotations.append(ann)

        # Training samples
        samples = extract_audio_training_samples(full_spec, tree,
                                                 args.retina_size)
        total_samples += len(samples)

        if args.visualise:
            vis = draw_audio_bbox_tree(spec_img, tree, n_frames)
            vis.save(os.path.join(args.out_dir, 'vis', spec_name))

        if (i + 1) % 5 == 0 or i == 0:
            print(f'  [{i+1}/{args.num_songs}]  notes={len(events)}  '
                  f'frames={n_frames}  samples_so_far={total_samples}')

    ann_path = os.path.join(args.out_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f'\nDone.  {args.num_songs} songs → {total_samples} training samples.')
    print(f'Annotations: {ann_path}')


if __name__ == '__main__':
    main()
