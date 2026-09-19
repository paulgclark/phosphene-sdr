// SPDX-License-Identifier: MIT

//! Hardware-validation harness for the SoapySDR backend (lane S1-A).
//!
//! Captures through [`SoapySource`] exactly as the app would — same trait,
//! same conversion path — then reports what the seal wants as evidence: the
//! sample count, wall duration, overflow count, the device-readback metadata,
//! and an averaged spectrum with its top peaks in absolute frequency, so a
//! known emitter (an FM broadcast station) can be checked against its real
//! dial frequency.
//!
//! ```text
//! cargo run -p phosphene-sources --features soapy --example soapy_capture -- \
//!     [device-args] [rate_hz] [center_hz] [gain_db] [seconds]
//! ```
//!
//! Defaults: any device, 2.048 MS/s, 100 MHz, 40 dB, 1 s.

use std::time::Instant;

use phosphene_core::{SpectrumAnalyzer, WindowKind};
use phosphene_sources::source::{SampleSink, SampleSource, SinkFlow};
use phosphene_sources::{Complex, SoapySource, SoapySourceConfig};

const FFT_SIZE: usize = 4096;

struct Collect {
    samples: Vec<Complex<f32>>,
    want: usize,
}

impl SampleSink for Collect {
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
        self.samples.extend_from_slice(samples);
        if self.samples.len() >= self.want {
            SinkFlow::Stop
        } else {
            SinkFlow::Continue
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device_args = args.first().cloned().unwrap_or_default();
    let parse = |i: usize, default: f64| -> f64 {
        args.get(i)
            .map_or(default, |s| s.parse().expect("numeric argument"))
    };
    let rate = parse(1, 2_048_000.0);
    let center = parse(2, 100_000_000.0);
    let gain = parse(3, 40.0);
    let seconds = parse(4, 1.0);

    let mut config = SoapySourceConfig::new(device_args);
    config.sample_rate_hz = Some(rate);
    config.center_freq_hz = Some(center);
    config.gain_db = Some(gain);

    let mut source = match SoapySource::new(config) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("soapy_capture: {e}");
            std::process::exit(1);
        }
    };

    let meta = source.meta();
    println!("device   : {} — {}", meta.label, meta.provenance);
    println!(
        "readback : rate {} Hz, centre {} Hz (requested {rate} / {center})",
        meta.sample_rate_hz.expect("hardware knows its rate"),
        meta.center_freq_hz.expect("hardware knows its centre"),
    );

    let want = (rate * seconds) as usize;
    let mut sink = Collect {
        samples: Vec::with_capacity(want),
        want,
    };
    let started = Instant::now();
    if let Err(e) = source.stream(&mut sink) {
        eprintln!("soapy_capture: {e}");
        std::process::exit(1);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let stats = source.stats();
    let n = sink.samples.len();
    let mean_power: f64 = sink
        .samples
        .iter()
        .map(|s| f64::from(s.re * s.re + s.im * s.im))
        .sum::<f64>()
        / n as f64;
    println!(
        "capture  : {n} samples in {elapsed:.3} s ({:.3} MS/s effective), \
         {} overflow event(s), mean power {mean_power:.3e} ({:.1} dBFS)",
        n as f64 / elapsed / 1e6,
        stats.overflows,
        10.0 * mean_power.log10(),
    );

    // Averaged periodogram over all whole frames.
    let mut analyzer = SpectrumAnalyzer::new(FFT_SIZE, WindowKind::Hann).expect("valid FFT size");
    let mut dbfs = vec![0.0f32; FFT_SIZE];
    let mut avg = vec![0.0f64; FFT_SIZE];
    let frames = n / FFT_SIZE;
    let (whole_frames, _) = sink.samples.as_chunks::<FFT_SIZE>();
    for frame in whole_frames {
        analyzer.process(frame, &mut dbfs);
        for (a, &d) in avg.iter_mut().zip(dbfs.iter()) {
            *a += f64::from(d);
        }
    }
    for a in &mut avg {
        *a /= frames as f64;
    }

    // Top peaks with a little local-maximum spacing so one carrier is one row.
    let rb = rate / FFT_SIZE as f64;
    let mut peaks: Vec<(usize, f64)> = avg
        .iter()
        .copied()
        .enumerate()
        .filter(|&(i, p)| {
            let lo = i.saturating_sub(8);
            let hi = (i + 8).min(FFT_SIZE - 1);
            avg[lo..=hi].iter().all(|&q| q <= p)
        })
        .collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let median = {
        let mut sorted = avg.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[FFT_SIZE / 2]
    };
    println!(
        "spectrum : {frames} × {FFT_SIZE}-point Hann frames, median floor {median:.1} dBFS/bin"
    );
    for (rank, (bin, power)) in peaks.iter().take(8).enumerate() {
        let freq = center + (*bin as f64 - FFT_SIZE as f64 / 2.0) * rb;
        println!(
            "  peak {rank}: {:.4} MHz  {power:.1} dBFS/bin  ({:+.1} dB over median floor)",
            freq / 1e6,
            power - median
        );
    }
}
