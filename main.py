#!/usr/bin/env python3
import argparse
import contextlib
import functools
import itertools
import json
import math
import re
import sys
import threading
import time
from fractions import Fraction as F
from pathlib import Path

try:
    import mido
except ImportError:
    sys.exit("mido is not installed. run pip install mido python-rtmidi")

MASTER = 0
MEMBERS = list(range(1, 16))

RPN_NRPN_CCS = {6, 38, 96, 97, 98, 99, 100, 101}
CHANNEL_MODE_CCS = set(range(120, 128))

CONFIG_PATH = Path(__file__).resolve().with_suffix('.json')

JI_RATIOS = {5: {
    0: (F(1),), 1: (F(16, 15),), 2: (F(9, 8), F(10, 9)), 3: (F(6, 5),),
    4: (F(5, 4),), 5: (F(4, 3),), 6: (F(45, 32), F(64, 45)), 7: (F(3, 2),),
    8: (F(8, 5),), 9: (F(5, 3),), 10: (F(9, 5), F(16, 9)), 11: (F(15, 8),),
}}
JI_RATIOS[7] = {**JI_RATIOS[5],
                6: JI_RATIOS[5][6] + (F(7, 5), F(10, 7)),
                10: JI_RATIOS[5][10] + (F(7, 4),)}

def _odd_part(n):
    while n % 2 == 0:
        n //= 2
    return n


def _complexity(ratio):
    return math.log2(_odd_part(ratio.numerator) * _odd_part(ratio.denominator))


@functools.lru_cache(maxsize=4096)
def ji_tuning(pcs, bass, limit):
    pcs = sorted(pcs)
    table = JI_RATIOS[limit]
    pairs = list(itertools.combinations(pcs, 2))
    best_score, best_cents = None, None
    for root in pcs:
        steps = {pc: (pc - root) % 12 for pc in pcs}
        for combo in itertools.product(*(table[steps[pc]] for pc in pcs)):
            ratio = dict(zip(pcs, combo))
            complexity = sum(_complexity(ratio[b] / ratio[a]) for a, b in pairs)
            cents = {pc: 1200 * math.log2(ratio[pc]) - 100 * steps[pc] for pc in pcs}
            score = (round(complexity, 6), root != bass, sum(map(abs, cents.values())))
            if best_score is None or score < best_score:
                best_score, best_cents = score, cents
    mean = sum(best_cents.values()) / len(best_cents)
    return {pc: c - mean for pc, c in best_cents.items()}


class MpeDistributor:
    def __init__(self, send, steal=True, verbose=False, ji_limit=None, bend_range=48, pb_scale=1.0):
        self.send = send
        self.steal = steal
        self.verbose = verbose
        self.ji_limit = ji_limit
        self.bend_range = bend_range
        self.pb_scale = pb_scale
        self.lock = threading.Lock()
        self.next_idx = 0
        self.channel_note = {}
        self.note_channels = {}
        self.in_state = {}
        self.ji_cents = {}

    def _log(self, text):
        if self.verbose:
            print(text, flush=True)

    def _allocate(self):
        n = len(MEMBERS)
        for i in range(n):
            ch = MEMBERS[(self.next_idx + i) % n]
            if ch not in self.channel_note:
                self.next_idx = (MEMBERS.index(ch) + 1) % n
                return ch

        if not self.steal:
            return None

        # All channels busy: steal the one holding the oldest note.
        ch = min(self.channel_note, key=lambda c: self.channel_note[c][2])
        source, note, _ = self.channel_note[ch]
        self._log(f"steal ch{ch + 1} (note {note})")
        self.send(mido.Message('note_off', channel=ch, note=note, velocity=0))
        self._release(source, note, ch)
        self.next_idx = (MEMBERS.index(ch) + 1) % n
        return ch

    def _release(self, source, note, out_ch):
        self.channel_note.pop(out_ch, None)
        self.ji_cents.pop(out_ch, None)
        stack = self.note_channels.get((source, note))
        if stack and out_ch in stack:
            stack.remove(out_ch)
            if not stack:
                del self.note_channels[(source, note)]

    def _out_channels_for(self, source):
        return [c for c, (s, _, _) in self.channel_note.items() if s == source]

    def _bend_msg(self, out_ch, source):
        msg = self.in_state.get(source, {}).get('pitchwheel')
        pitch = round(msg.pitch * self.pb_scale) if msg else 0
        pitch += round(self.ji_cents.get(out_ch, 0.0) * 8192 / (100 * self.bend_range))
        return mido.Message('pitchwheel', channel=out_ch, pitch=max(-8192, min(8191, pitch)))

    def _prime_channel(self, out_ch, source):
        state = self.in_state.get(source, {})
        self.send(self._bend_msg(out_ch, source))
        if 'aftertouch' not in state:
            self.send(mido.Message('aftertouch', channel=out_ch, value=0))
        for key, msg in state.items():
            if key != 'pitchwheel':
                self.send(msg.copy(channel=out_ch))

    def _retune(self, new_ch):

        held = {ch: note for ch, (_, note, _) in self.channel_note.items()}
        if len(held) < 2:
            return
        tuning = ji_tuning(frozenset(n % 12 for n in held.values()),
                           min(held.values()) % 12, self.ji_limit)
        for ch, note in held.items():
            cents = tuning[note % 12]
            changed = self.ji_cents.get(ch) != cents
            self.ji_cents[ch] = cents
            if changed and ch != new_ch:
                self.send(self._bend_msg(ch, self.channel_note[ch][0]))
        self._log("ji:" + ", ".join(f"{note} {self.ji_cents[ch]:+.1f}c"
                                     for ch, note in sorted(held.items(), key=lambda x: x[1])))

    def handle(self, msg, port=None):
        with self.lock:
            self._handle(msg, port)

    def _handle(self, msg, port):
        if not hasattr(msg, 'channel'):
            self.send(msg)
            return

        in_ch = msg.channel
        source = (port, in_ch)
        t = msg.type

        if t == 'note_on' and msg.velocity > 0:
            out_ch = self._allocate()
            if out_ch is None:
                self._log(f"drop note {msg.note}, no free channel")
                return
            self.channel_note[out_ch] = (source, msg.note, time.monotonic())
            self.note_channels.setdefault((source, msg.note), []).append(out_ch)
            if self.ji_limit:
                self._retune(out_ch)
            self._prime_channel(out_ch, source)
            self.send(msg.copy(channel=out_ch))
            self._log(f"{port or 'in'} ch{in_ch + 1} note {msg.note} -> out ch{out_ch + 1}")
            return

        if t == 'note_off' or (t == 'note_on' and msg.velocity == 0):
            stack = self.note_channels.get((source, msg.note))
            if not stack:
                return
            out_ch = stack[0]
            self.send(msg.copy(channel=out_ch))
            self._release(source, msg.note, out_ch)
            self._log(f"{port or 'in'} ch{in_ch + 1} note {msg.note} off (out ch{out_ch + 1})")
            return

        if t == 'polytouch':
            for out_ch in self.note_channels.get((source, msg.note), []):
                self.send(mido.Message('aftertouch', channel=out_ch, value=msg.value))
            return

        if in_ch == MASTER:
            if t == 'pitchwheel':
                msg = msg.copy(pitch=round(msg.pitch * self.pb_scale))
            self.send(msg)
            return

        if t == 'control_change' and (msg.control in RPN_NRPN_CCS or msg.control in CHANNEL_MODE_CCS):
            for out_ch in MEMBERS:
                self.send(msg.copy(channel=out_ch))
            return

        if t in ('pitchwheel', 'aftertouch', 'control_change'):
            key = ('cc', msg.control) if t == 'control_change' else t
            self.in_state.setdefault(source, {})[key] = msg
            for out_ch in self._out_channels_for(source):
                self.send(self._bend_msg(out_ch, source) if t == 'pitchwheel'
                          else msg.copy(channel=out_ch))
            return

        self.send(msg.copy(channel=MASTER))

    def all_notes_off(self):
        with self.lock:
            for out_ch, (_, note, _) in list(self.channel_note.items()):
                self.send(mido.Message('note_off', channel=out_ch, note=note, velocity=0))
            self.channel_note.clear()
            self.note_channels.clear()
            self.ji_cents.clear()

def send_rpn(send, channel, rpn, msb, lsb=0):
    send(mido.Message('control_change', channel=channel, control=101, value=(rpn >> 7) & 0x7F))
    send(mido.Message('control_change', channel=channel, control=100, value=rpn & 0x7F))
    send(mido.Message('control_change', channel=channel, control=6, value=msb))
    send(mido.Message('control_change', channel=channel, control=38, value=lsb))
    send(mido.Message('control_change', channel=channel, control=101, value=127))
    send(mido.Message('control_change', channel=channel, control=100, value=127))

def unit_float(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not between 0.0 and 1.0")
    return value


def ask_numbers(prompt, count, multiple):
    while True:
        try:
            text = input(prompt).strip()
        except EOFError:
            sys.exit("\nNo selection made.")
        parts = [p for p in re.split(r'[\s,]+', text) if p]
        if not parts:
            continue
        if not all(p.isdigit() and int(p) < count for p in parts):
            print(f"enter numbers from 0 to {count - 1}")
            continue
        picks = list(dict.fromkeys(int(p) for p in parts))
        if not multiple and len(picks) > 1:
            print("pick only one output")
            continue
        return picks


def prompt_ports(in_names, out_names):
    if not in_names:
        sys.exit("no midi input ports found")
    if not out_names:
        sys.exit("no midi output ports found")

    print("\nMIDI inputs:")
    for i, n in enumerate(in_names):
        print(f"  [{i}] {n}")
    ins = ask_numbers("select inputs. separated by commas or spaces: ",
                      len(in_names), multiple=True)

    print("\nMIDI outputs:")
    for i, n in enumerate(out_names):
        print(f"  [{i}] {n}")
    out = ask_numbers("select output: ", len(out_names), multiple=False)

    return [in_names[i] for i in ins], out_names[out[0]]


def find_port(saved, names):
    if saved in names:
        return saved
    base = re.sub(r'\s+\d+$', '', saved)
    for n in names:
        if re.sub(r'\s+\d+$', '', n) == base:
            return n
    return None


def load_config():
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg['inputs'], cfg['output']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_config(inputs, output):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'inputs': inputs, 'output': output}, f, indent=2)
    print(f"Saved midi selection to {CONFIG_PATH}")


def choose_ports(reprompt):
    in_names = mido.get_input_names()
    out_names = mido.get_output_names()

    saved = None if reprompt else load_config()
    if saved:
        saved_ins, saved_out = saved
        ins = [p for p in (find_port(s, in_names) for s in saved_ins) if p]
        out = find_port(saved_out, out_names)
        for s in saved_ins:
            if not find_port(s, in_names):
                print(f"saved input '{s}' is not connected so skipping it")
        if ins and out:
            return ins, out
        print("the saved midi ports aren't available, choose again or plug the devices in and restart")

    ins, out = prompt_ports(in_names, out_names)
    save_config(ins, out)
    return ins, out


def main():
    ap = argparse.ArgumentParser(description="Spread midi across mpe channels.")
    ap.add_argument('--reprompt-midi', action='store_true',
                    help="choose the midi ports again and save the new choice")
    ap.add_argument('--send-mcm', action='store_true',
                    help="at startup, send mpe configuration Message (lower zone, 15 member channels)")
    ap.add_argument('--bend-range', type=int, metavar='SEMITONES',
                    help="at startup, set pitch bend range of member channels 2-16, default = 48"
                         "--ji also uses it to convert cents to pitch bend")
    ap.add_argument('--no-steal', action='store_true',
                    help="drop new notes when all 15 channels are busy default = steal the oldest note")
    ap.add_argument('--ji', action='store_true',
                    help="retune notes played together to pure ji intervals with per-note pitch bend")
    ap.add_argument('--ji-limit', type=int, choices=(5, 7), default=5,
                    help="with --ji: 5 = 5-limit ratios, 7 = also allow 7/4, 7/5, 10/7")
    ap.add_argument('--pbscale', type=unit_float, default=1.0, metavar='0.0-1.0',
                    help="scale the controller's pitch bend by this factor. default = 1")
    ap.add_argument('-v', '--verbose', action='store_true', help="print each channel assignment")
    args = ap.parse_args()

    try:
        in_names, out_name = choose_ports(args.reprompt_midi)
    except ImportError:
        sys.exit("python-rtmidi, run pip install python-rtmidi")

    with mido.open_output(out_name) as outport, contextlib.ExitStack() as stack:
        send = outport.send

        if args.send_mcm:
            send_rpn(send, MASTER, 6, len(MEMBERS))
        if args.bend_range is not None:
            for ch in MEMBERS:
                send_rpn(send, ch, 0, args.bend_range) 

        bend_range = args.bend_range or 48
        dist = MpeDistributor(send, steal=not args.no_steal, verbose=args.verbose,
                              ji_limit=args.ji_limit if args.ji else None,
                              bend_range=bend_range, pb_scale=args.pbscale)

        for name in in_names:
            stack.enter_context(mido.open_input(
                name, callback=lambda msg, port=name: dist.handle(msg, port)))

        print("in: " + "\n".join(in_names))
        try:
            while True:
                time.sleep(0.5)
        finally:
            dist.all_notes_off()


if __name__ == '__main__':
    main()