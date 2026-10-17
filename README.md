# midistuff

lil midi mpe utility. main functionaity is in spreading sequential notes in their own midi channels, so that expressive controllers playing does not affect previous notes with long release time.
there's also --ji which will retune intervals to their ji equivalent.

requires python 3.9+, mido python-rtmidi
On windows, a virtual MIDI port such as [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) if you want to send the output into a daw.

on the first run you choose the input ports and one output port. that is then saved to main.json next to the script and reused on later runs. If windows renumbers a port, the saved choice still matches it.

the flags you run with are saved to main.json too and become the defaults for the next run, so python main.py --ji 7 once and then plain python main.py keeps 7-limit JI on. flags you pass override the saved ones. turn a saved switch off with its --no- form (--no-ji, --no-preserve-pressure, --no-bend-range eetc) or use --reset-flags to go back to the built-in defaults.

press Ctrl+C to quit

plain mpe channel spreading - python main.py

configure the synth at startup: MPE on, bend range 48 semitones - python main.py --send-mcm --bend-range 48

7-limit ji - python main.py --ji 7

keep pressure and timbre on released notes so their tails keep their expression - python main.py --preserve-pressure 0.7 --preserve-timbre

A non mpe controllers are also supported for ji.

--reprompt-midi chooses the midi ports again and save the new choice

--reset-flags ignores the flags saved from the last run and start from the built-in defaults

--send-mcm at startup sends the mpe Configuration Message (lower zone, 15 member channels)

--bend-range SEMITONES sets the pitch bend range of channels 2–16. --ji also uses this value to convert cents to bend. default is 48

--no-steal when all 15 channels are busy drop new notes instead of stealing the oldest one. --steal turns stealing back on

--pbscale 0.0-1.0 scales incoming pitch bend by this factor. default is 1

--ji PRIME retune notes played together to just intonation, building ratios from primes up to PRIME (3 = pythagorean, 5, 7, 11, 13 etc). default is 5

--ji-tolerance CENTS with --ji only uses ratios within this many cents of the key played. default = 35

--preserve-pressure SMOOTHING keeps sending pressure to a channel after its note is released until the channel is reused. pressure is smoothed by 0.0–1.0 (default 0.5) so that lifting a finger doesn't drop the channel straight to zero. 0 = no smoothing

--preserve-timbre keeps sending timbre to a channel after its note is released until the channel is reused

--unipolar-timbre makes timbre positive-only. the distance from the centre (64) in either direction is scaled to 0–127 so the centre sends 0

-v, --verbose prints channel assignments, js tunings and just stuff in general

with --ji every time a note starts the tool finds a just tuning for all the notes being held and retunes them with per-note pitch bend.

for each interval it chooses the simplest ratios (fewest and smallest prime factors) that are within --ji-tolerance of the 12-TET interval
when two tunings are equally simple, it prefers the one built on the lowest note
the tuning is centred so that the average offset is 0 cents, which stops chords drifting away from concert pitch

