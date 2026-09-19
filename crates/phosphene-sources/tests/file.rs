// SPDX-License-Identifier: MIT

//! Seal tests for the M1-E lane (lane spec §Seal):
//!
//! * a full-scale tone encoded in each of the five formats (M3-C added
//!   `cs32`) reads 0.0 dBFS ±0.1 through `phosphene-core` — the cross-format
//!   calibration invariant;
//! * siggen output written as each format and read back matches within format
//!   precision (round-trip);
//! * truncated files and odd-length input are handled without panic and
//!   reported honestly;
//! * `--loop` is seamless; replay is unthrottled when no rate is given
//!   (clarification C5) and throttled to real time when one is;
//! * FIFO and reader (pipe-like) inputs stream; stdin and non-seekable inputs
//!   reject `--loop`.

use std::io::Write as _;
use std::path::PathBuf;

use phosphene_core::{fftshift_index, SpectrumAnalyzer, WindowKind};
use phosphene_sources::format::{decode_append, encode_append};
use phosphene_sources::metadata;
use phosphene_sources::{
    Complex, Control, ControlCaps, FileSource, FileSourceConfig, Input, IqFormat, ReplayPace,
    SampleSink, SampleSource, SigGen, SigGenConfig, SinkFlow, SourceDesc, SourceError, SourceMeta,
    ToneConfig,
};

/// A sink that records everything and stops after a target sample count
/// (`usize::MAX` = drain the source).
struct CollectingSink {
    samples: Vec<Complex<f32>>,
    metas: Vec<SourceMeta>,
    target: usize,
}

impl CollectingSink {
    fn until(target: usize) -> Self {
        CollectingSink {
            samples: Vec::new(),
            metas: Vec::new(),
            target,
        }
    }

    fn drain() -> Self {
        Self::until(usize::MAX)
    }
}

impl SampleSink for CollectingSink {
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
        assert!(
            !self.metas.is_empty(),
            "push before meta_changed violates the stream contract"
        );
        self.samples.extend_from_slice(samples);
        if self.samples.len() >= self.target {
            SinkFlow::Stop
        } else {
            SinkFlow::Continue
        }
    }

    fn meta_changed(&mut self, meta: &SourceMeta) {
        self.metas.push(meta.clone());
    }
}

/// A unique temp path for this test run; removed by [`TempFile::drop`].
struct TempFile(PathBuf);

impl TempFile {
    fn with_bytes(name: &str, bytes: &[u8]) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!("phosphene-m1e-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).expect("temp file must be writable");
        TempFile(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn file_config(path: &TempFile, format: IqFormat) -> FileSourceConfig {
    FileSourceConfig::new(Input::Path(path.0.clone()), format)
}

fn drain_source(source: &mut FileSource) -> CollectingSink {
    let mut sink = CollectingSink::drain();
    source.stream(&mut sink).expect("stream must succeed");
    sink
}

/// The lane's headline seal: a full-scale complex tone encoded in each of the
/// five formats reads 0.0 dBFS ±0.1 through `phosphene-core` — one
/// calibration across every format there is (D-006).
///
/// It iterates [`IqFormat::ALL`] rather than a list of its own, so a format
/// added later (M3-C added `cs32`) is calibrated by this test the moment it
/// exists — the seal D-062 calls "the seal that matters", because a format
/// that decodes but miscalibrates yields a display wrong by a constant and
/// entirely plausible-looking.
#[test]
fn full_scale_tone_reads_0_dbfs_in_every_format() {
    let n = 1024usize;
    let bin = 100usize;
    let frames = 4usize;
    // Bin-exact full-scale tone, magnitude exactly 1.0.
    let tone: Vec<Complex<f32>> = (0..n * frames)
        .map(|i| {
            let phase = std::f64::consts::TAU * bin as f64 * i as f64 / n as f64;
            Complex::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect();

    for format in IqFormat::ALL {
        let mut bytes = Vec::new();
        encode_append(format, &tone, &mut bytes);
        let file = TempFile::with_bytes(&format!("cal-{format}.iq"), &bytes);

        let mut source = FileSource::new(file_config(&file, format)).unwrap();
        let sink = drain_source(&mut source);
        assert_eq!(sink.samples.len(), n * frames, "{format}");

        let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
        let mut dbfs = vec![0.0f32; n];
        analyzer.process(&sink.samples[..n], &mut dbfs);

        let peak_bin = fftshift_index(bin, n);
        let peak = dbfs[peak_bin];
        assert!(
            peak.abs() < 0.1,
            "{format}: full-scale tone read {peak} dBFS, expected 0.0 ±0.1"
        );
        let argmax = (0..n).max_by(|&a, &b| dbfs[a].total_cmp(&dbfs[b])).unwrap();
        assert_eq!(argmax, peak_bin, "{format}: peak landed in the wrong bin");
    }
}

/// Round-trip seal: siggen output written as each format and read back
/// matches within format precision.
#[test]
fn siggen_round_trip_through_every_format() {
    // A scene that stays inside full scale (0.5 + 0.25 < 1.0), so integer
    // encoding never clamps.
    let rate = 1_024_000.0;
    let mut config = SigGenConfig::new(rate);
    config.tones = vec![
        ToneConfig {
            offset_hz: 100_000.0,
            level_dbfs: -6.0206,
        },
        ToneConfig {
            offset_hz: -250_000.0,
            level_dbfs: -12.0412,
        },
    ];
    let mut gen = SigGen::new(config).unwrap();
    let mut original = vec![Complex::new(0.0f32, 0.0f32); 8192];
    gen.fill(&mut original);

    // Half a quantisation step per format (0 = bit-exact).
    let tolerances = [
        (IqFormat::Cf32, 0.0f32),
        (IqFormat::Cs32, 0.5 / 2_147_483_647.0),
        (IqFormat::Cs16, 0.5 / 32767.0),
        (IqFormat::Cs8, 0.5 / 127.0),
        (IqFormat::Cu8, 0.5 / 127.5),
    ];
    for (format, tol) in tolerances {
        let mut bytes = Vec::new();
        encode_append(format, &original, &mut bytes);
        let file = TempFile::with_bytes(&format!("rt-{format}.iq"), &bytes);

        let mut source = FileSource::new(file_config(&file, format)).unwrap();
        let sink = drain_source(&mut source);
        assert_eq!(sink.samples.len(), original.len(), "{format}");
        for (i, (a, b)) in original.iter().zip(&sink.samples).enumerate() {
            assert!(
                (a.re - b.re).abs() <= tol + f32::EPSILON
                    && (a.im - b.im).abs() <= tol + f32::EPSILON,
                "{format} sample {i}: wrote {a}, read {b} (tolerance {tol})"
            );
        }
    }
}

/// Truncation seal: a partial sample at EOF is a normal condition — no panic,
/// whole samples delivered, the remainder counted honestly.
#[test]
fn truncated_and_odd_length_input_is_reported_not_panicked() {
    // 10 whole cs16 samples plus 3 stray bytes.
    let mut bytes = Vec::new();
    encode_append(
        IqFormat::Cs16,
        &[Complex::new(0.25f32, -0.25f32); 10],
        &mut bytes,
    );
    bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
    let file = TempFile::with_bytes("truncated.cs16", &bytes);

    let mut source = FileSource::new(file_config(&file, IqFormat::Cs16)).unwrap();
    let sink = drain_source(&mut source);
    assert_eq!(sink.samples.len(), 10);
    assert_eq!(source.stats().samples_delivered, 10);
    assert_eq!(source.stats().trailing_bytes, 3);

    // Streaming again after exhaustion is a clean no-op, not a double-count.
    let sink = drain_source(&mut source);
    assert!(sink.samples.is_empty());
    assert_eq!(source.stats().trailing_bytes, 3);

    // An input shorter than a single sample yields zero samples, honestly.
    let file = TempFile::with_bytes("tiny.cf32", &[1, 2, 3, 4, 5, 6, 7]);
    let mut source = FileSource::new(file_config(&file, IqFormat::Cf32)).unwrap();
    let sink = drain_source(&mut source);
    assert!(sink.samples.is_empty());
    assert_eq!(source.stats().trailing_bytes, 7);

    // Even under --loop, a sampleless input terminates instead of spinning.
    let file = TempFile::with_bytes("tiny-loop.cf32", &[1, 2, 3]);
    let mut config = file_config(&file, IqFormat::Cf32);
    config.loop_replay = true;
    let mut source = FileSource::new(config).unwrap();
    let sink = drain_source(&mut source);
    assert!(sink.samples.is_empty());
}

/// `--loop` seal: the replayed stream wraps seamlessly — the concatenation is
/// exactly the file repeated, with no gap, glitch, or partial-sample bleed.
#[test]
fn loop_replay_is_seamless() {
    let pattern: Vec<Complex<f32>> = (0..100)
        .map(|i| Complex::new(i as f32, -(i as f32)))
        .collect();
    let mut bytes = Vec::new();
    encode_append(IqFormat::Cf32, &pattern, &mut bytes);
    // A trailing partial sample must be discarded on every pass, never
    // spliced into the start of the next.
    bytes.extend_from_slice(&[0xDE, 0xAD]);
    let file = TempFile::with_bytes("loop.cf32", &bytes);

    let mut config = file_config(&file, IqFormat::Cf32);
    config.loop_replay = true; // no rate: C5 unthrottled replay
    let mut source = FileSource::new(config).unwrap();
    let mut sink = CollectingSink::until(350);
    source.stream(&mut sink).unwrap();

    assert!(sink.samples.len() >= 350);
    for (i, s) in sink.samples.iter().enumerate() {
        let expected = &pattern[i % pattern.len()];
        assert_eq!(
            (s.re.to_bits(), s.im.to_bits()),
            (expected.re.to_bits(), expected.im.to_bits()),
            "loop replay diverges at sample {i}"
        );
    }
    assert!(source.stats().loops_completed >= 3);
    assert!(source.stats().trailing_bytes >= 3 * 2);
    assert_eq!(sink.metas[0].sample_rate_hz, None, "no rate was invented");
}

/// A sink that timestamps every push (relative to its construction, which
/// tests place immediately before `stream`) and stops after a target count.
struct TimingSink {
    start: std::time::Instant,
    /// (elapsed at push, samples in push), in delivery order.
    pushes: Vec<(std::time::Duration, usize)>,
    metas: Vec<SourceMeta>,
    target: usize,
    received: usize,
}

impl TimingSink {
    fn until(target: usize) -> Self {
        TimingSink {
            start: std::time::Instant::now(),
            pushes: Vec::new(),
            metas: Vec::new(),
            target,
            received: 0,
        }
    }
}

impl SampleSink for TimingSink {
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
        self.pushes.push((self.start.elapsed(), samples.len()));
        self.received += samples.len();
        if self.received >= self.target {
            SinkFlow::Stop
        } else {
            SinkFlow::Continue
        }
    }

    fn meta_changed(&mut self, meta: &SourceMeta) {
        self.metas.push(meta.clone());
    }
}

/// Assert the pacing invariant on a recorded delivery schedule: at no push
/// had more samples arrived than `rate × elapsed` allows (equivalently, every
/// push happened no earlier than the stream time of its last sample). The
/// 2 ms tolerance covers timer/measurement granularity only — `thread::sleep`
/// never wakes early, so the invariant itself has no slack.
fn assert_never_ahead_of_rate(pushes: &[(std::time::Duration, usize)], rate: f64, what: &str) {
    let tol = std::time::Duration::from_millis(2);
    let mut cum = 0usize;
    for (i, (at, len)) in pushes.iter().enumerate() {
        cum += len;
        let due = std::time::Duration::from_secs_f64(cum as f64 / rate);
        assert!(
            *at + tol >= due,
            "{what}: push {i} brought delivery to {cum} samples at {at:?}, \
             but rate {rate} allows that count only from {due:?}"
        );
    }
}

/// Timing seal (R2): throttled replay is paced *within* blocks, not around
/// read sizes — delivery timestamps match the configured rate with the
/// stated burstiness bound ([`phosphene_sources::file::PACE_BLOCK_MAX_S`]),
/// and C5 still means no rate → no throttle and no invented metadata.
#[test]
fn throttled_delivery_matches_the_rate_within_bounded_blocks() {
    use phosphene_sources::file::PACE_BLOCK_MAX_S;

    let total = 8192usize;
    let rate = 200_000.0; // 8192 samples → 40.96 ms of stream time
    let samples = vec![Complex::new(0.1f32, 0.0f32); total];
    let mut bytes = Vec::new();
    encode_append(IqFormat::Cs16, &samples, &mut bytes);
    let file = TempFile::with_bytes("pace.cs16", &bytes);

    // Real-time (the default): the whole file fits in one read, so every
    // property below is about pacing inside the block, not around it.
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(rate);
    let mut source = FileSource::new(config).unwrap();
    let mut sink = TimingSink::until(usize::MAX);
    source.stream(&mut sink).unwrap();

    assert_eq!(sink.received, total);
    assert_eq!(sink.metas[0].sample_rate_hz, Some(rate));
    // Burstiness is bounded as stated: no push exceeds PACE_BLOCK_MAX_S of
    // stream time, so this replay cannot arrive as one (or two) bursts.
    let max_block = (rate * PACE_BLOCK_MAX_S) as usize;
    for (i, (_, len)) in sink.pushes.iter().enumerate() {
        assert!(
            *len <= max_block,
            "push {i} delivered {len} samples; the stated bound is {max_block}"
        );
    }
    assert!(
        sink.pushes.len() >= total / max_block,
        "only {} pushes for {total} samples — pacing is around the read, not in it",
        sink.pushes.len()
    );
    // Delivery never runs ahead of the rate, push by push...
    assert_never_ahead_of_rate(&sink.pushes, rate, "real-time replay");
    // ...and the full replay takes the stream duration, within a stated
    // tolerance. The upper bound is deliberately loose (scheduler load can
    // stretch sleeps); the lower bound is the guarantee.
    let expected = std::time::Duration::from_secs_f64(total as f64 / rate);
    let (finished, _) = *sink.pushes.last().unwrap();
    assert!(
        finished + std::time::Duration::from_millis(2) >= expected,
        "replay finished in {finished:?}, faster than {expected:?} allows"
    );
    assert!(
        finished <= expected + std::time::Duration::from_millis(500),
        "replay took {finished:?}, far beyond {expected:?} — throttle miscalibrated"
    );

    // Fast-forward at 4×: the same invariant against the scaled rate.
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(rate);
    config.pace = ReplayPace::Factor(4.0);
    let mut source = FileSource::new(config).unwrap();
    let mut sink = TimingSink::until(usize::MAX);
    source.stream(&mut sink).unwrap();
    assert_eq!(sink.received, total);
    assert_never_ahead_of_rate(&sink.pushes, rate * 4.0, "4× fast-forward");
    let (finished, _) = *sink.pushes.last().unwrap();
    let expected = std::time::Duration::from_secs_f64(total as f64 / (rate * 4.0));
    assert!(
        finished + std::time::Duration::from_millis(2) >= expected,
        "4× fast-forward finished in {finished:?}, faster than {expected:?}"
    );

    // No rate: C5 — unthrottled, rate honestly absent, nothing invented.
    let config = file_config(&file, IqFormat::Cs16);
    let mut source = FileSource::new(config).unwrap();
    let sink = drain_source(&mut source);
    assert_eq!(sink.samples.len(), total);
    assert_eq!(sink.metas[0].sample_rate_hz, None);
    assert_eq!(sink.metas[0].center_freq_hz, None);
}

/// Pacing-arithmetic seal (R2, second round): the arithmetic is TOTAL. A rate
/// and pace factor that are each individually valid but whose PRODUCT is
/// unusable (overflow to infinity — the probe that exited 101 —, underflow to
/// zero, or too slow for a representable per-sample schedule slot) return a
/// clean, actionable error naming the offending values; either side of the
/// boundary constructs, and the extreme-but-finite side also streams without
/// panic.
#[test]
fn pacing_product_extremes_error_cleanly_instead_of_panicking() {
    let samples: Vec<Complex<f32>> = (0..64)
        .map(|i| Complex::new(i as f32 / 64.0, 0.0))
        .collect();
    let mut bytes = Vec::new();
    encode_append(IqFormat::Cs16, &samples, &mut bytes);
    let file = TempFile::with_bytes("pace-extreme.cs16", &bytes);

    // The probe: finite positive rate × finite positive factor = infinity.
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(1e308);
    config.pace = ReplayPace::Factor(10.0);
    let err = FileSource::new(config).unwrap_err();
    let SourceError::InvalidConfig(msg) = &err else {
        panic!("expected InvalidConfig, got {err:?}");
    };
    assert!(
        msg.contains("1e308") && msg.contains("10"),
        "error must name the offending values: {msg}"
    );

    // Underflow: the product collapses to zero.
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(1e-308);
    config.pace = ReplayPace::Factor(1e-100);
    assert!(matches!(
        FileSource::new(config),
        Err(SourceError::InvalidConfig(_))
    ));

    // Positive but so slow a single sample's slot exceeds representable time
    // (real-time pace: the rate alone is the product).
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(1e-300);
    assert!(matches!(
        FileSource::new(config),
        Err(SourceError::InvalidConfig(_))
    ));

    // Either side of the boundary. Huge-but-finite product: constructs AND
    // streams to completion without panic (every schedule slot is already
    // past due, so this is effectively unthrottled)...
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(1e200);
    config.pace = ReplayPace::Factor(1e100);
    let mut source = FileSource::new(config).expect("finite product must construct");
    let sink = drain_source(&mut source);
    assert_eq!(sink.samples.len(), 64);

    // ...slow-but-schedulable constructs (1 Hz: a 1 s slot is fine)...
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(1.0);
    FileSource::new(config).expect("1 Hz must be a valid replay rate");

    // ...and an ordinary fast-forward is untouched.
    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(1e6);
    config.pace = ReplayPace::Factor(4.0);
    FileSource::new(config).expect("ordinary rates must be unaffected");
}

/// Timing seal (R2): a stop/resume cycle resumes PACING, not just delivery —
/// the resumed window never receives more samples than the rate allows.
#[test]
fn stop_resume_resumes_pacing_not_a_burst() {
    let total = 8192usize;
    let rate = 200_000.0;
    let samples = vec![Complex::new(0.1f32, 0.0f32); total];
    let mut bytes = Vec::new();
    encode_append(IqFormat::Cs16, &samples, &mut bytes);
    let file = TempFile::with_bytes("pace-resume.cs16", &bytes);

    let mut config = file_config(&file, IqFormat::Cs16);
    config.sample_rate_hz = Some(rate);
    let mut source = FileSource::new(config).unwrap();

    // Session 1: consumer pauses partway through.
    let mut first = TimingSink::until(2000);
    source.stream(&mut first).unwrap();
    assert_never_ahead_of_rate(&first.pushes, rate, "pre-stop session");
    let stopped_at = first.received;
    assert!(stopped_at >= 2000 && stopped_at < total);

    // Session 2: the resumed window is itself paced — no unpaced burst.
    let mut resumed = TimingSink::until(usize::MAX);
    source.stream(&mut resumed).unwrap();
    assert_eq!(stopped_at + resumed.received, total);
    assert_never_ahead_of_rate(&resumed.pushes, rate, "resumed session");
    let remaining = std::time::Duration::from_secs_f64((total - stopped_at) as f64 / rate);
    let (finished, _) = *resumed.pushes.last().unwrap();
    assert!(
        finished + std::time::Duration::from_millis(2) >= remaining,
        "resumed session delivered {} samples in {finished:?}; \
         the rate allows them only over {remaining:?}",
        resumed.received
    );
}

/// Pipe-like inputs stream through the same path; stop/resume continues
/// exactly where the stream left off, carried partial sample included.
#[test]
fn reader_input_streams_and_resumes_across_a_stop() {
    let pattern: Vec<Complex<f32>> = (0..1000)
        .map(|i| Complex::new(i as f32 / 1000.0, 0.5))
        .collect();
    let mut bytes = Vec::new();
    encode_append(IqFormat::Cs16, &pattern, &mut bytes);

    let config = FileSourceConfig::new(Input::Stdin, IqFormat::Cs16);
    let mut source =
        FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
    assert_eq!(source.meta().label, "stdin");

    // Stop after 300 samples...
    let mut sink = CollectingSink::until(300);
    source.stream(&mut sink).unwrap();
    let first = sink.samples.len();
    assert!(first >= 300);

    // ...and resume: the remainder follows with nothing lost or repeated.
    let mut rest = CollectingSink::drain();
    source.stream(&mut rest).unwrap();
    assert_eq!(first + rest.samples.len(), pattern.len());
    let tol = 0.5 / 32767.0 + f32::EPSILON;
    for (i, b) in sink.samples.iter().chain(&rest.samples).enumerate() {
        let a = &pattern[i];
        assert!(
            (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol,
            "sample {i} lost across stop/resume: {a} vs {b}"
        );
    }
}

/// A FIFO path behaves like the pipe it is (FR-S1: stdin / FIFO / file).
#[cfg(unix)]
#[test]
fn fifo_path_streams_like_a_pipe() {
    let mut path = std::env::temp_dir();
    path.push(format!("phosphene-m1e-{}-fifo.cs8", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let status = std::process::Command::new("mkfifo").arg(&path).status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("skipping FIFO test: mkfifo unavailable");
            return;
        }
    }

    // --loop over a FIFO is rejected up front (fs::metadata never blocks).
    let mut config = FileSourceConfig::new(Input::Path(path.clone()), IqFormat::Cs8);
    config.loop_replay = true;
    assert!(matches!(
        FileSource::new(config),
        Err(SourceError::InvalidConfig(_))
    ));

    let samples: Vec<Complex<f32>> = (0..100)
        .map(|i| Complex::new(i as f32 / 127.0, 0.0))
        .collect();
    let mut bytes = Vec::new();
    encode_append(IqFormat::Cs8, &samples, &mut bytes);

    let writer_path = path.clone();
    let writer = std::thread::spawn(move || {
        // Blocks until the reader opens the other end.
        let mut fifo = std::fs::OpenOptions::new()
            .write(true)
            .open(&writer_path)
            .expect("open fifo for write");
        fifo.write_all(&bytes).expect("write into fifo");
        // Dropping closes the write end → reader sees EOF.
    });

    let config = FileSourceConfig::new(Input::Path(path.clone()), IqFormat::Cs8);
    let mut source = FileSource::new(config).unwrap();
    let sink = drain_source(&mut source);
    writer.join().unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(sink.samples.len(), 100);
    let tol = 0.5 / 127.0 + f32::EPSILON;
    for (i, (a, b)) in samples.iter().zip(&sink.samples).enumerate() {
        assert!((a.re - b.re).abs() <= tol, "fifo sample {i}: {a} vs {b}");
    }
}

#[test]
fn invalid_configurations_are_rejected_with_specific_errors() {
    let file = TempFile::with_bytes("valid.cf32", &[0u8; 16]);

    // Loop over stdin and over an arbitrary reader: no rewind, no loop.
    let mut config = FileSourceConfig::new(Input::Stdin, IqFormat::Cf32);
    config.loop_replay = true;
    let err = FileSource::new(config.clone()).unwrap_err();
    assert!(matches!(err, SourceError::InvalidConfig(_)), "{err}");
    assert!(err.to_string().contains("stdin"), "{err}");
    let err = FileSource::from_reader(Box::new(std::io::empty()), config).unwrap_err();
    assert!(matches!(err, SourceError::InvalidConfig(_)), "{err}");

    // Bad rates, centers, and pace factors.
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut config = file_config(&file, IqFormat::Cf32);
        config.sample_rate_hz = Some(rate);
        assert!(matches!(
            FileSource::new(config),
            Err(SourceError::InvalidConfig(_))
        ));
    }
    let mut config = file_config(&file, IqFormat::Cf32);
    config.center_freq_hz = Some(f64::NAN);
    assert!(matches!(
        FileSource::new(config),
        Err(SourceError::InvalidConfig(_))
    ));
    for factor in [0.0, -2.0, f64::NAN, f64::INFINITY] {
        let mut config = file_config(&file, IqFormat::Cf32);
        config.sample_rate_hz = Some(1e6);
        config.pace = ReplayPace::Factor(factor);
        assert!(matches!(
            FileSource::new(config),
            Err(SourceError::InvalidConfig(_))
        ));
    }

    // A missing file is an I/O error naming the path.
    let config = FileSourceConfig::new(
        Input::Path(PathBuf::from("/nonexistent/phosphene-m1e.cf32")),
        IqFormat::Cf32,
    );
    let err = FileSource::new(config).unwrap_err();
    assert!(matches!(err, SourceError::Io(_)), "{err}");
    assert!(err.to_string().contains("phosphene-m1e.cf32"), "{err}");
}

#[test]
fn trait_contract_wrong_backend_controls_and_metadata() {
    let file = TempFile::with_bytes("meta.cu8", &[128u8; 20]);

    // open() routes by descriptor; each backend rejects the other's.
    assert!(matches!(
        FileSource::open(&SourceDesc::SigGen(SigGenConfig::new(1e6))),
        Err(SourceError::WrongBackend { backend: "file" })
    ));
    let mut config = file_config(&file, IqFormat::Cu8);
    config.sample_rate_hz = Some(2_400_000.0);
    config.center_freq_hz = Some(100_000_000.0);
    assert!(matches!(
        SigGen::open(&SourceDesc::File(config.clone())),
        Err(SourceError::WrongBackend { backend: "siggen" })
    ));

    let mut source = FileSource::open(&SourceDesc::File(config)).unwrap();
    assert_eq!(source.caps(), ControlCaps::NONE);
    assert!(matches!(
        source.set(Control::GainDb(10.0)),
        Err(SourceError::UnsupportedControl {
            backend: "file",
            ..
        })
    ));

    let meta = source.meta();
    assert!(
        meta.label.ends_with("meta.cu8") && !meta.label.contains('/'),
        "label should be the bare file name, got {:?}",
        meta.label
    );
    assert_eq!(meta.sample_rate_hz, Some(2_400_000.0));
    assert_eq!(meta.center_freq_hz, Some(100_000_000.0));
    assert!(meta.provenance.contains("cu8"), "{}", meta.provenance);
    assert!(meta.provenance.contains("meta.cu8"), "{}", meta.provenance);

    // And the equivalent through decode: cu8 midpoint decodes to (almost) DC
    // silence, well below full scale.
    let mut decoded = Vec::new();
    decode_append(IqFormat::Cu8, &[128u8; 2], &mut decoded);
    assert!(decoded[0].re.abs() < 0.01 && decoded[0].im.abs() < 0.01);
}

/// M3-C's round-trip seal, through the production `FileSource` path: the same
/// samples written as `cs32` and as `cf32` read back the same, to within the
/// `cs32` quantisation bound.
///
/// This is the test that catches a wrong scale. A decoder that divided by
/// 2³¹ − 1 read as `2^31` (or by 65536, or by anything else) would still
/// produce a smooth-looking spectrum; it would simply sit at the wrong level
/// forever. Comparing against `cf32` — the format that needs no scale at all
/// — is what makes the constant visible.
#[test]
fn cs32_and_cf32_replay_the_same_signal() {
    let rate = 1_024_000.0;
    let mut config = SigGenConfig::new(rate);
    config.tones = vec![ToneConfig {
        offset_hz: 137_000.0,
        level_dbfs: -3.0,
    }];
    let mut gen = SigGen::new(config).unwrap();
    let mut original = vec![Complex::new(0.0f32, 0.0f32); 4096];
    gen.fill(&mut original);

    let read_back = |format: IqFormat| {
        let mut bytes = Vec::new();
        encode_append(format, &original, &mut bytes);
        let file = TempFile::with_bytes(&format!("cs32-rt-{format}.iq"), &bytes);
        let mut source = FileSource::new(file_config(&file, format)).unwrap();
        drain_source(&mut source).samples
    };
    let via_cs32 = read_back(IqFormat::Cs32);
    let via_cf32 = read_back(IqFormat::Cf32);

    assert_eq!(via_cs32.len(), original.len());
    assert_eq!(via_cf32.len(), original.len());
    // Half a `cs32` quantisation step, which is far below `f32` resolution:
    // the two agree essentially bit for bit.
    let tol = 0.5 / 2_147_483_647.0;
    for (i, (a, b)) in via_cs32.iter().zip(&via_cf32).enumerate() {
        assert!(
            (a.re - b.re).abs() <= tol + f32::EPSILON && (a.im - b.im).abs() <= tol + f32::EPSILON,
            "sample {i}: cs32 read {a}, cf32 read {b}"
        );
    }
    // And 8 bytes a sample is what was actually consumed — a wrong stride
    // would have produced a different sample count from the same file.
    assert_eq!(IqFormat::Cs32.bytes_per_sample(), 8);
}

/// The D-006 seal for `cs32`, closed against the **raw codes** rather than
/// against our own encoder (D-062: "the seal that matters").
///
/// [`full_scale_tone_reads_0_dbfs_in_every_format`] writes its tone through
/// `encode_append` and reads it back through `decode_append`, so a scale that
/// was wrong in *both* directions would cancel and the tone would still read
/// 0.0 dBFS. This test never asks the encoder anything: it writes the i32
/// codes a full-scale `ci32_le` recorder writes — `round(cos·2147483647)` —
/// and asserts the production path reads 0.0 dBFS ±0.1 from them, invariant
/// across FFT size and window as D-006 requires.
#[test]
fn a_full_scale_cs32_tone_written_as_raw_codes_reads_0_dbfs() {
    let frames = 4usize;
    for (n, window) in [
        (1024usize, WindowKind::Hann),
        (2048, WindowKind::Hann),
        (1024, WindowKind::BlackmanHarris4),
        (2048, WindowKind::BlackmanHarris4),
    ] {
        let bin = n / 8;
        // Bytes exactly as a full-scale ci32_le capture carries them: no
        // `encode_append` anywhere in this construction.
        let mut bytes = Vec::with_capacity(n * frames * 8);
        for i in 0..n * frames {
            let phase = std::f64::consts::TAU * bin as f64 * i as f64 / n as f64;
            let code = |v: f64| ((v * 2_147_483_647.0).round() as i32).to_le_bytes();
            bytes.extend_from_slice(&code(phase.cos()));
            bytes.extend_from_slice(&code(phase.sin()));
        }
        let file = TempFile::with_bytes(&format!("raw-cs32-{n}-{window:?}.cs32"), &bytes);

        let mut source = FileSource::new(file_config(&file, IqFormat::Cs32)).unwrap();
        let sink = drain_source(&mut source);
        assert_eq!(sink.samples.len(), n * frames, "n={n}");

        let mut analyzer = SpectrumAnalyzer::new(n, window).unwrap();
        let mut dbfs = vec![0.0f32; n];
        analyzer.process(&sink.samples[..n], &mut dbfs);
        let peak = dbfs[fftshift_index(bin, n)];
        assert!(
            peak.abs() < 0.1,
            "raw full-scale cs32 codes read {peak} dBFS at n={n} {window:?}, expected 0.0 ±0.1"
        );
    }
}

// ---------------------------------------------------------------------------
// M3-C's corpus seal: bytes we did not write
// ---------------------------------------------------------------------------

/// The committed excerpts of the real SDRangel `ci32_le` recording. See
/// `tests/corpus/README.md` for their provenance and how to regenerate them.
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus");

/// The environment variable that points the opt-in test at the **whole** 8.8 MB
/// recording, which is deliberately not vendored into this public repository.
const CORPUS_ENV: &str = "PHOSPHENE_CORPUS_CI32";

/// A `.sigmf-data` fixture and its sidecar, copied into a scratch directory so
/// nothing ever writes near the committed originals.
struct CorpusFixture {
    dir: PathBuf,
    data: PathBuf,
}

impl CorpusFixture {
    /// Copy `<stem>.sigmf-data` (and `<stem>.sigmf-meta`, if it exists) out of
    /// `tests/corpus` under the name the real recording carries, so the
    /// sidecar lookup and the filename rungs see exactly what they would see
    /// on the owner's disk.
    fn stage(tag: &str, stem: &str, as_name: &str) -> CorpusFixture {
        let dir = std::env::temp_dir().join(format!(
            "phosphene-corpus-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let data = dir.join(format!("{as_name}.sigmf-data"));
        std::fs::copy(format!("{CORPUS}/{stem}.sigmf-data"), &data).expect("stage the capture");
        let meta = PathBuf::from(format!("{CORPUS}/{stem}.sigmf-meta"));
        if meta.is_file() {
            std::fs::copy(&meta, dir.join(format!("{as_name}.sigmf-meta"))).expect("stage sidecar");
        }
        CorpusFixture { dir, data }
    }
}

impl Drop for CorpusFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

/// Walk D-055's precedence chain over a capture and hand back the config the
/// app's `build_source` would have built from it — no flags, nothing typed.
fn config_from_the_capture(path: &std::path::Path) -> FileSourceConfig {
    let meta = metadata::resolve(Some(path), None, None, None)
        .expect("the sidecar must be believable and its datatype supported");
    FileSourceConfig {
        input: Input::Path(path.to_path_buf()),
        format: meta.format().expect("the sidecar declares the format"),
        sample_rate_hz: meta.rate(),
        center_freq_hz: meta.center(),
        pace: ReplayPace::RealTime,
        loop_replay: false,
    }
}

/// **The seal that could actually have failed**: our decode of 548 samples of
/// a real `ci32_le` recording is compared against *somebody else's* decode of
/// the very same samples.
///
/// Every other `cs32` test in this file writes its bytes with `encode_append`
/// and reads them with `decode_append`. That proves the arithmetic is
/// self-consistent and nothing more — a scale wrong in both directions
/// cancels, a swapped I/Q swaps back. What the format was *added* for is
/// reading a recorder that is not us, and only a file we did not write can
/// test that.
///
/// `slice70401.reference.cf32` is the openphy vector toolchain's `cf32` decode
/// of `slice70401.sigmf-data` — a NumPy pipeline that shares no code with
/// phosphene, produced as ground truth for an unrelated 5G NR test. Agreement
/// pins four separate byte-level decisions at once: little-endian words, I
/// before Q, an 8-byte stride, and the 2³¹ divisor. Break any one of them and
/// the comparison fails.
///
/// It says nothing about *level*, which is the point: `ci32_le` names the
/// container and not where full scale sits inside it, and **D-071** rules
/// that `cs32` normalises by the container maximum and that SDRangel's
/// 24-in-32 recordings therefore read ≈48 dB low, correctly. This test pins
/// the decode without pinning a level. See `tests/corpus/README.md`.
///
/// The assertion is **bit equality**, not a tolerance: our `i32 as f32 /
/// i32::MAX as f32` and their `raw / 2**31` are the same computation, because
/// `i32::MAX as f32` rounds to exactly 2³¹ (see `format.rs`).
#[test]
fn the_corpus_ci32_le_bytes_decode_as_an_independent_decoder_read_them() {
    let reference = std::fs::read(format!(
        "{CORPUS}/sdrangel-pluto-ci32le-slice70401.reference.cf32"
    ))
    .expect("the third-party reference decode must be committed beside the slice");
    let expected: Vec<Complex<f32>> = reference
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            Complex::new(
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
            )
        })
        .collect();
    assert_eq!(expected.len(), 548, "the reference slice is 548 samples");

    // Through the production path: a real file on disk, opened as cs32.
    let fx = CorpusFixture::stage("slice", "sdrangel-pluto-ci32le-slice70401", "slice");
    let mut source = FileSource::new(FileSourceConfig::new(
        Input::Path(fx.data.clone()),
        IqFormat::Cs32,
    ))
    .unwrap();
    let ours = drain_source(&mut source).samples;

    assert_eq!(
        ours.len(),
        expected.len(),
        "an 8-byte stride over {} bytes of real ci32_le",
        548 * 8
    );
    for (i, (a, b)) in ours.iter().zip(&expected).enumerate() {
        assert_eq!(
            (a.re, a.im),
            (b.re, b.im),
            "sample {i}: phosphene read {a}, the independent decoder read {b}"
        );
    }
    // The slice is real signal, not a run of zeros that would agree trivially.
    let peak = ours.iter().map(|s| s.norm()).fold(0.0f32, f32::max);
    assert!(
        peak > 1e-4,
        "the reference slice must carry signal, peak was {peak}"
    );
}

/// **The capture that motivated the lane, opened end to end.**
///
/// The first 16 384 samples of the real recording, under the real recording's
/// name, beside the real recording's sidecar, with **no flags typed**: the
/// D-055 chain reads `ci32_le` → [`IqFormat::Cs32`], 7.68 MS/s and
/// 1876.954 MHz off the sidecar; `FileSource` streams the bytes; and the
/// `SourceMeta` the sink receives — the value the chrome and the frequency
/// axis actually consume (D-031) — carries the declared rate and centre.
///
/// Before M3-C this file's honest answer was `UnsupportedDatatype`.
#[test]
fn the_corpus_ci32_le_capture_opens_from_its_own_sidecar() {
    let fx = CorpusFixture::stage(
        "head",
        "sdrangel-pluto-ci32le-head",
        "1876954_7680KSPS_srsRAN_Project_gnb_short",
    );
    let config = config_from_the_capture(&fx.data);
    assert_eq!(config.format, IqFormat::Cs32, "core:datatype: \"ci32_le\"");
    assert_eq!(config.sample_rate_hz, Some(7.68e6));
    assert_eq!(config.center_freq_hz, Some(1_876_954_000.0));

    let mut source = FileSource::new(config).unwrap();
    let sink = drain_source(&mut source);
    assert_eq!(
        sink.samples.len(),
        16_384,
        "8 bytes per sample, no leftover"
    );

    let meta = sink.metas.last().expect("the source must announce itself");
    assert_eq!(meta.sample_rate_hz, Some(7.68e6));
    assert_eq!(meta.center_freq_hz, Some(1_876_954_000.0));
    assert!(
        meta.provenance.contains("cs32"),
        "the provenance must name the native format: {:?}",
        meta.provenance
    );

    // It decoded real, varying samples rather than a run of zeros or a
    // constant — every component finite and inside full scale, which is a
    // decode-correctness property of the format and true of any `cs32` file.
    assert!(
        sink.samples
            .iter()
            .all(|s| s.re.is_finite() && s.im.is_finite()),
        "a real capture must not decode to NaN or infinity"
    );
    assert!(
        sink.samples
            .iter()
            .all(|s| s.re.abs() <= 1.0 && s.im.abs() <= 1.0),
        "every cs32 code decodes inside ±1.0 by construction"
    );
    assert!(
        sink.samples.iter().any(|s| *s != sink.samples[0]),
        "the capture decoded to a constant — nothing was really read"
    );

    // **No level is asserted here, deliberately (D-071 §4.)** `ci32_le` names
    // the container and not where full scale sits inside it, so the dBFS a
    // real recording reads depends on its recorder's convention — SDRangel
    // writes 24-bit values in this 32-bit container and therefore reads
    // ≈48 dB low, which D-071 rules is correct behaviour for an ambiguous
    // input. Asserting a number here would bake one producer's choice into
    // our seal. What is asserted instead is what the sidecar actually
    // declares: the format, the rate and the centre, all the way to the
    // `SourceMeta` the display consumes.
    //
    // The *decode* is pinned elsewhere and does not need a level to be:
    // [`the_corpus_ci32_le_bytes_decode_as_an_independent_decoder_read_them`]
    // compares our bytes-to-samples against a third party's, bit for bit.
    let n = 1024usize;
    let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
    let mut dbfs = vec![0.0f32; n];
    analyzer.process(&sink.samples[..n], &mut dbfs);
    assert!(
        dbfs.iter().all(|v| v.is_finite()),
        "the capture must produce a finite spectrum"
    );
}

/// The same proof over the **whole** 8.8 MB recording rather than a committed
/// prefix — opt-in, because the capture is not vendored into this public repo
/// (see `tests/corpus/README.md`):
///
/// ```sh
/// PHOSPHENE_CORPUS_CI32=…/1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data \
///   cargo test -p phosphene-sources --test file -- --ignored the_whole_corpus_recording
/// ```
///
/// It reads the file where it lives, unthrottled (1 100 272 samples at
/// 7.68 MS/s is 143 ms of capture, and pacing it would only make the test
/// slow), and asserts the same things the prefix does: the sidecar resolves
/// to `cs32` with its declared rate and centre, every whole sample in the
/// file decodes, and none of them is NaN or outside ±1.0. It **prints** the
/// levels it measured and asserts none of them, for D-071 §4's reason.
#[test]
#[ignore = "needs the 8.8 MB corpus capture; set PHOSPHENE_CORPUS_CI32 to its path"]
fn the_whole_corpus_recording_opens_end_to_end() {
    let Ok(path) = std::env::var(CORPUS_ENV) else {
        panic!("set {CORPUS_ENV} to the .sigmf-data path (see tests/corpus/README.md)");
    };
    let path = PathBuf::from(path);
    let mut config = config_from_the_capture(&path);
    assert_eq!(config.format, IqFormat::Cs32);
    assert_eq!(config.sample_rate_hz, Some(7.68e6));
    assert_eq!(config.center_freq_hz, Some(1_876_954_000.0));
    config.pace = ReplayPace::Unthrottled;

    let bytes = std::fs::metadata(&path)
        .expect("the capture must be readable")
        .len();
    let mut source = FileSource::new(config).unwrap();
    let sink = drain_source(&mut source);
    assert_eq!(
        sink.samples.len() as u64,
        bytes / 8,
        "every whole 8-byte sample in the file, and no partial one"
    );

    let peak = sink.samples.iter().map(|s| s.norm()).fold(0.0f32, f32::max);
    assert!(
        sink.samples
            .iter()
            .all(|s| s.re.is_finite() && s.im.is_finite()),
        "a real capture must not decode to NaN or infinity"
    );
    assert!(
        sink.samples
            .iter()
            .all(|s| s.re.abs() <= 1.0 && s.im.abs() <= 1.0),
        "every cs32 code decodes inside ±1.0 by construction"
    );

    let n = 2048usize;
    let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
    let mut dbfs = vec![0.0f32; n];
    analyzer.process(&sink.samples[..n], &mut dbfs);
    let top = dbfs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        dbfs.iter().all(|v| v.is_finite()),
        "the capture must produce a finite spectrum"
    );
    // Reported, never asserted — the level is the recorder's convention, not
    // ours to seal (D-071 §4).
    println!(
        "{} samples, peak |x| {peak:.6} ({:.1} dBFS), strongest bin {top:.1} dBFS",
        sink.samples.len(),
        20.0 * peak.log10()
    );
}
