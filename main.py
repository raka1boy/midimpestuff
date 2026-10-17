#!/usr/bin/env python3
import argparse
import contextlib
import functools
import heapq
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
TIMBRE_CC = 74

CONFIG_PATH = Path(__file__).resolve().with_suffix('.json')
UNSAVED_FLAGS = {'help', 'reprompt_midi', 'reset_flags'}

JI_TOLERANCE = 35.0
JI_PER_STEP = 4
JI_SLACK = 3.0
JI_MAX_HEIGHT = 2 ** 24


def _odd_part(n):
    while n % 2 == 0:
        n //= 2
    return n


def _complexity(ratio):
    return math.log2(_odd_part(ratio.numerator) * _odd_part(ratio.denominator))


def _odd_primes(limit):
    return [p for p in range(3, limit + 1, 2)
            if all(p % q for q in range(3, math.isqrt(p) + 1, 2))]


def _octave_reduce(ratio):
    ratio *= F(2) ** (ratio.denominator.bit_length() - ratio.numerator.bit_length())
    return ratio * 2 if ratio < 1 else ratio


@functools.lru_cache(maxsize=None)
def ji_ratios(limit, tolerance=JI_TOLERANCE):
    primes = _odd_primes(limit)
    found = {step: [] for step in range(1, 12)}
    heap, seen = [1], {1}
    while heap:
        m = heapq.heappop(heap)
        height = math.log2(m)
        if m > JI_MAX_HEIGHT or all(len(v) >= JI_PER_STEP or (v and height > v[0][1] + JI_SLACK)
                                    for v in found.values()):
            break
        for p in primes:
            if m * p not in seen:
                seen.add(m * p)
                heapq.heappush(heap, m * p)
        powers = []
        for p in primes:
            pe = 1
            while m % (pe * p) == 0:
                pe *= p
            if pe > 1:
                powers.append(pe)
        for split in itertools.product((False, True), repeat=len(powers)):
            num = math.prod(pe for pe, up in zip(powers, split) if up)
            ratio = _octave_reduce(F(num, m // num))
            cents = 1200 * math.log2(ratio)
            step = round(cents / 100)
            v = found.get(step)
            if (v is not None and abs(cents - 100 * step) <= tolerance and len(v) < JI_PER_STEP
                    and (not v or height <= v[0][1] + JI_SLACK)):
                v.append((ratio, height))
    for step, v in found.items():
        if not v:
            raise ValueError(f"no {limit}-limit ratio within {tolerance:g} cents of {step} semitones")
    return {0: (F(1),), **{step: tuple(r for r, _ in v) for step, v in found.items()}}


@functools.lru_cache(maxsize=None)
def _ji_space(limit, tolerance):
    table = ji_ratios(limit, tolerance)
    ratios = sorted({r for rs in table.values() for r in rs})
    index = {r: i for i, r in enumerate(ratios)}
    dist = [[_complexity(b / a) for b in ratios] for a in ratios]
    return ratios, {step: [index[r] for r in rs] for step, rs in table.items()}, dist


@functools.lru_cache(maxsize=4096)
def ji_tuning(pcs, bass, limit, tolerance=JI_TOLERANCE):
    ratios, table, dist = _ji_space(limit, tolerance)
    pcs = sorted(pcs)
    best_score, best_cents = None, None

    for root in pcs:
        steps = {pc: (pc - root) % 12 for pc in pcs}
        order = sorted(pcs, key=lambda pc: len(table[steps[pc]]))
        cands = [table[steps[pc]] for pc in order]
        n = len(order)
        floor = [[min(dist[a][b] for a in cands[i] for b in cands[j]) for j in range(n)]
                 for i in range(n)]
        tail = [sum(floor[i][j] for i in range(k, n) for j in range(i + 1, n)) for k in range(n + 1)]
        chosen = [0] * n

        def search(i, cost, rest):
            nonlocal best_score, best_cents
            if i == n:
                cents = {pc: 1200 * math.log2(ratios[c]) - 100 * steps[pc]
                         for pc, c in zip(order, chosen)}
                score = (round(cost, 6), root != bass, sum(map(abs, cents.values())))
                if best_score is None or score < best_score:
                    best_score, best_cents = score, cents
                return
            if best_score and round(cost + sum(map(min, rest)) + tail[i], 6) > best_score[0]:
                return
            for c, extra in sorted(zip(cands[i], rest[0]), key=lambda x: x[1]):
                chosen[i] = c
                row = dist[c]
                search(i + 1, cost + extra,
                       [[acc + row[b] for acc, b in zip(r, cands[k])]
                        for k, r in enumerate(rest[1:], i + 1)])

        search(0, 0.0, [[0.0] * len(c) for c in cands])

    mean = sum(best_cents.values()) / len(best_cents)
    return {pc: c - mean for pc, c in best_cents.items()}


class MpeDistributor:
    def __init__(self, send, steal=True, verbose=False, ji_limit=None, ji_tolerance=JI_TOLERANCE,
                 bend_range=48, pb_scale=1.0, preserve=(), pressure_smoothing=0.0, unipolar_timbre=False):
        self.send = send
        self.preserve = set(preserve)
        self.unipolar_timbre = unipolar_timbre
        self.pressure_smoothing = pressure_smoothing
        self.steal = steal
        self.verbose = verbose
        self.ji_limit = ji_limit
        self.ji_tolerance = ji_tolerance
        self.bend_range = bend_range
        self.pb_scale = pb_scale
        self.lock = threading.Lock()
        self.next_idx = 0
        self.channel_note = {}
        self.note_channels = {}
        self.in_state = {}
        self.ji_cents = {}
        self.lingering = {}
        self.pressure = {}

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
        if self.preserve:
            self.lingering[out_ch] = source
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

    def _send_pressure(self, out_ch, value):
        prev = self.pressure.get(out_ch)
        if prev is not None:
            value = prev * self.pressure_smoothing + value * (1 - self.pressure_smoothing)
        self.pressure[out_ch] = value
        self.send(mido.Message('aftertouch', channel=out_ch, value=round(value)))

    def _prime_channel(self, out_ch, source):
        state = self.in_state.get(source, {})
        self.send(self._bend_msg(out_ch, source))
        self.pressure.pop(out_ch, None)
        self._send_pressure(out_ch, state['aftertouch'].value if 'aftertouch' in state else 0)
        for key, msg in state.items():
            if key not in ('pitchwheel', 'aftertouch'):
                self.send(msg.copy(channel=out_ch))

    def _retune(self, new_ch):

        held = {ch: note for ch, (_, note, _) in self.channel_note.items()}
        if len(held) < 2:
            return
        tuning = ji_tuning(frozenset(n % 12 for n in held.values()),
                           min(held.values()) % 12, self.ji_limit, self.ji_tolerance)
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

        if self.unipolar_timbre and t == 'control_change' and msg.control == TIMBRE_CC:
            msg = msg.copy(value=min(127, abs(msg.value - 64) * 2))

        if t == 'note_on' and msg.velocity > 0:
            out_ch = self._allocate()
            if out_ch is None:
                self._log(f"drop note {msg.note}, no free channel")
                return
            self.lingering.pop(out_ch, None)
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
                self._send_pressure(out_ch, msg.value)
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
            targets = self._out_channels_for(source)
            if key in self.preserve:
                targets += [c for c, s in self.lingering.items() if s == source]
            for out_ch in targets:
                if t == 'pitchwheel':
                    self.send(self._bend_msg(out_ch, source))
                elif t == 'aftertouch':
                    self._send_pressure(out_ch, msg.value)
                else:
                    self.send(msg.copy(channel=out_ch))
            return

        self.send(msg.copy(channel=MASTER))

    def all_notes_off(self):
        with self.lock:
            for out_ch, (_, note, _) in list(self.channel_note.items()):
                self.send(mido.Message('note_off', channel=out_ch, note=note, velocity=0))
            self.channel_note.clear()
            self.note_channels.clear()
            self.ji_cents.clear()
            self.lingering.clear()
            self.pressure.clear()

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


def ji_limit_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number")
    if value < 3:
        raise argparse.ArgumentTypeError(f"{value} is below 3, the smallest useful limit")
    return value


def cents_tolerance(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number")
    if not 0.0 < value <= 50.0:
        raise argparse.ArgumentTypeError(f"{value} is not above 0 and at most 50")
    return value


def ask_numbers(prompt, count, multiple):
    while True:
        try:
            text = input(prompt).strip()
        except EOFError:
            sys.exit("\nno selection made")
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
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def save_config(**updates):
    cfg = load_config()
    cfg.update(updates)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)


def apply_saved_flags(ap, saved):
    if not isinstance(saved, dict):
        return
    defaults, seen = {}, set(UNSAVED_FLAGS)
    for action in ap._actions:
        if action.dest not in saved or action.dest in seen:
            continue
        seen.add(action.dest)
        value = saved[action.dest]
        try:
            if isinstance(action.default, bool) and not isinstance(value, bool):
                raise ValueError
            if value is not None and action.type:
                value = action.type(str(value))
        except (argparse.ArgumentTypeError, ValueError):
            print(f"ignoring saved {action.dest} = {value!r}, it's not valid")
            continue
        defaults[action.dest] = value
    ap.set_defaults(**defaults)


def choose_ports(reprompt):
    in_names = mido.get_input_names()
    out_names = mido.get_output_names()

    cfg = {} if reprompt else load_config()
    saved_ins, saved_out = cfg.get('inputs'), cfg.get('output')
    if (isinstance(saved_ins, list) and all(isinstance(s, str) for s in saved_ins)
            and isinstance(saved_out, str)):
        ins = [p for p in (find_port(s, in_names) for s in saved_ins) if p]
        out = find_port(saved_out, out_names)
        for s in saved_ins:
            if not find_port(s, in_names):
                print(f"saved input '{s}' is not connected so skipping it")
        if ins and out:
            return ins, out
        print("the saved midi ports aren't available, choose again or plug the devices in and restart")

    ins, out = prompt_ports(in_names, out_names)
    save_config(inputs=ins, output=out)
    print(f"saved midi selection to {CONFIG_PATH}")
    return ins, out


def main():
    ap = argparse.ArgumentParser(description="Spread midi across mpe channels.")
    ap.add_argument('--reprompt-midi', action='store_true',
                    help="choose the midi ports again and save the new choice")
    ap.add_argument('--reset-flags', action='store_true',
                    help="ignore the flags saved from the last run and start from the built-in defaults")
    ap.add_argument('--send-mcm', action=argparse.BooleanOptionalAction, default=False,
                    help="at startup, send mpe configuration Message (lower zone, 15 member channels)")
    ap.add_argument('--bend-range', type=int, metavar='SEMITONES',
                    help="at startup, set pitch bend range of member channels 2-16, default = 48"
                         "--ji also uses it to convert cents to pitch bend")
    ap.add_argument('--no-bend-range', dest='bend_range', action='store_const', const=None,
                    help="don't set the pitch bend range at startup")
    ap.add_argument('--steal', action=argparse.BooleanOptionalAction, default=True,
                    help="steal the oldest note when all 15 channels are busy, --no-steal drops new notes instead")
    ap.add_argument('--ji', type=ji_limit_int, nargs='?', const=5, metavar='PRIME',
                    help="retune notes played together to pure ji intervals with per-note pitch bend, "
                         "using ratios built from primes up to PRIME (3 = pythagorean, 5, 7, 11, 13 etc). "
                         "default = 5")
    ap.add_argument('--no-ji', dest='ji', action='store_const', const=None,
                    help="turn ji retuning off")
    ap.add_argument('--ji-tolerance', type=cents_tolerance, default=JI_TOLERANCE, metavar='CENTS',
                    help="with --ji: only use ratios within this many cents of the key played"
                         f"default = {JI_TOLERANCE:g}")
    ap.add_argument('--pbscale', type=unit_float, default=1.0, metavar='0.0-1.0',
                    help="scale the controller's pitch bend by this factor. default = 1")
    ap.add_argument('--preserve-pressure', type=unit_float, nargs='?', const=0.5, metavar='SMOOTHING',
                    help="keep sending pressure to a channel after its note is released, until the channel is reused. "
                         "pressure is smoothed by smoothing 0.0-1.0 and default = 0.5 so a lifted finger "
                         "leaves the channel with a non-zero pressure. 0 = no smoothing")
    ap.add_argument('--no-preserve-pressure', dest='preserve_pressure', action='store_const', const=None,
                    help="stop sending pressure to channels after their note is released")
    ap.add_argument('--preserve-timbre', action=argparse.BooleanOptionalAction, default=False,
                    help=f"keep sending timbre cc to a channel after its note is released, "
                         "until the channel is reused")
    ap.add_argument('--unipolar-timbre', action=argparse.BooleanOptionalAction, default=False,
                    help="make timbre cc positive-only. distance from the centre =64 in either direction "
                         "is scaled to 0-127, so the centre sends 0")
    ap.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False,
                    help="print each channel assignment and just more stuff in general")

    if '--reset-flags' not in sys.argv[1:]:
        apply_saved_flags(ap, load_config().get('flags'))
    args = ap.parse_args()

    if args.ji:
        try:
            table = ji_ratios(args.ji, args.ji_tolerance)
        except ValueError as e:
            sys.exit(f"{e}. raise --ji-tolerance or the --ji limit")
        if args.verbose:
            print(f"{args.ji}-limit ratios:")
            for step, rs in table.items():
                print(f"{step:2}: " + ", ".join(
                    f"{r} ({1200 * math.log2(r) - 100 * step:+.1f}c)" for r in rs))

    save_config(flags={k: v for k, v in vars(args).items() if k not in UNSAVED_FLAGS})

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
        preserve_pressure = args.preserve_pressure is not None
        preserve = ['aftertouch'] * preserve_pressure + [('cc', TIMBRE_CC)] * args.preserve_timbre
        dist = MpeDistributor(send, steal=args.steal, verbose=args.verbose,
                              ji_limit=args.ji,
                              ji_tolerance=args.ji_tolerance, bend_range=bend_range, pb_scale=args.pbscale,
                              preserve=preserve,
                              pressure_smoothing=args.preserve_pressure if preserve_pressure else 0.0,
                              unipolar_timbre=args.unipolar_timbre)

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