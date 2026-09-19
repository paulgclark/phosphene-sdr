<!-- SPDX-License-Identifier: MIT -->

# `ci32_le` corpus fixtures — real SDRangel bytes, not ours

These four files are **excerpts of a real third-party recording**. They exist because a
self-encoded round trip — write with `encode_append`, read with `decode_append` — tests our
arithmetic and cannot fail for the reason that matters: a scale wrong in *both* directions
cancels. What M3-C actually needed proving is our **interpretation** of somebody else's bytes —
endianness, I/Q order, stride, and the divisor — and only a file we did not write can prove
that.

## The recording

`1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-*`, an SDRangel capture of an srsRAN Project
gNB made with a PlutoSDR: `core:datatype: "ci32_le"`, `core:sample_rate: 7680000.0`,
`core:frequency: 1876954000.0`. It is the capture that motivated D-062: before `cs32` existed,
phosphene's honest answer to it was a named `UnsupportedDatatype` refusal.

The full recording is 8.8 MB and lives outside this repository. It is **deliberately not
vendored** — this repo is public-bound, and a capture is large. Only the excerpts below are
committed, and every byte of IQ in them is verbatim from the original file.

## The files

| file | what it is |
| --- | --- |
| `sdrangel-pluto-ci32le-head.sigmf-data` | the recording's **first 16 384 samples**, byte for byte (131 072 B) |
| `sdrangel-pluto-ci32le-head.sigmf-meta` | the recording's own sidecar (see the two edits below) |
| `sdrangel-pluto-ci32le-slice70401.sigmf-data` | samples **70 401 … 70 948**, byte for byte (4 384 B) |
| `sdrangel-pluto-ci32le-slice70401.reference.cf32` | **a third party's `cf32` decode of that same slice** |

The sidecar is the original's, with exactly two edits, both because the excerpt is not the whole
recording and a fixture must not claim otherwise:

* **`core:sha512` removed** — it hashes the full 8.8 MB of data, which this 128 KB prefix is not.
* **`annotations` emptied** — 35 KB of PSS/SSS labels at sample offsets far past the prefix.

Everything else, `sdrangel:rx_bits: 24` included, is the recorder's own text untouched.

## Why the `.reference.cf32` file is the load-bearing one

It was produced from this recording by the openphy vector toolchain — a Jupyter/NumPy pipeline
that is not phosphene and shares no code with it — as the ground truth for an unrelated 5G NR
test. It is an independent decoder's reading of the same 548 samples.

Our decode of `slice70401.sigmf-data` is **bit-identical** to it in `f32`. That single fact pins
all four byte-level decisions at once: little-endian words, I before Q, 8 bytes per sample, and
the 2³¹ divisor. A wrong endianness, a swapped I/Q, a half-sample stride or a divisor off by 2⁸
all break it — and it pins them **without asserting a level**, which is what makes it the right
seal under D-071 (see below).

## The level these read at — `sdrangel:rx_bits: 24`, ruled by D-071

The recorder writes the Pluto's 12-bit codes left-shifted inside a **24-bit** field: every code in
the recording is a whole multiple of 4096, and the whole file peaks at 0.9575 × 2²³. It is a
well-driven capture at a 24-bit full scale, sitting in a 32-bit container.

`cs32` normalises by the **container's** maximum, `i32::MAX` — the only value derivable from
`ci32_le` alone, and the one that keeps D-006 true for a genuine 32-bit full-scale file. So this
recording reads **≈48 dB low** (`20·log₁₀(2⁸)` = 48.16), and **D-071 rules that this is correct
behaviour for an ambiguous input**: SigMF's datatype names the container, not where full scale
sits inside it, and nothing in the file says which. phosphene does not sniff `core:recorder` and
does not infer a scale from the observed peak — a heuristic that lifts a genuinely quiet capture
by 48 dB is far worse than a level that is honestly low, and is undetectable when wrong. An
explicit `--full-scale-bits` override is the right fix and is on the backlog.

**That is why the tests over these fixtures assert no dBFS level.** They assert the file opens and
decodes at the sidecar's declared rate and centre; a level assertion would bake one recorder's
convention into our seal. The decode itself is pinned without a level, by the bit-for-bit
comparison above.

For the record, read at the documented `i32::MAX` scale: −82 dBFS peak over the prefix, −55.8 over
the slice, −48.5 over the whole recording.

## Regenerating them

```sh
python3 - <<'PY'
import json
src = '<path to>/captures/pluto_n3_ncellid1_short/'
dst = 'crates/phosphene-sources/tests/corpus/'
data = src + '1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data'

with open(data, 'rb') as f:
    open(dst + 'sdrangel-pluto-ci32le-head.sigmf-data', 'wb').write(f.read(16384 * 8))
with open(data, 'rb') as f:
    f.seek(70401 * 8)
    open(dst + 'sdrangel-pluto-ci32le-slice70401.sigmf-data', 'wb').write(f.read(548 * 8))
open(dst + 'sdrangel-pluto-ci32le-slice70401.reference.cf32', 'wb').write(
    open(src + 'expected_slice_70401_to_70948.cf32', 'rb').read())

m = json.load(open(src + '1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta'))
g = dict(m['global']); g.pop('core:sha512', None)
open(dst + 'sdrangel-pluto-ci32le-head.sigmf-meta', 'w').write(
    json.dumps({'global': g, 'captures': m['captures'], 'annotations': []}, indent=4) + '\n')
PY
```

## The whole file, when you have it

`tests/file.rs` also carries an opt-in test over the **complete** 8.8 MB recording, skipped by
default because the file is not in the repo:

```sh
PHOSPHENE_CORPUS_CI32=<path to>/1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data \
  cargo test -p phosphene-sources --test file -- --ignored the_whole_corpus_recording
```
