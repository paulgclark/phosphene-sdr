// SPDX-License-Identifier: MIT

//! The device half of the [`soapy`](super) backend — everything that touches
//! libSoapySDR, compiled only under the `soapy` feature. See the module docs
//! in `mod.rs` for the Pattern C licensing audit trail and the behaviour
//! contract.

use std::time::{Duration, Instant};

use phosphene_core::Complex;
use soapysdr::{Args, Device, Direction, ErrorCode};

use super::{
    chosen_message, format_device_args, no_devices_message, only_audio_message, parse_device_args,
    select_device, source_meta, Selection, SoapyDeviceInfo, SoapySourceConfig,
};
use crate::source::{
    apply_and_verify, control_channel, settle_control, Control, ControlCaps, ControlHandle,
    ControlInbox, ControlOutcome, SampleSink, SampleSource, SinkFlow, SourceDesc, SourceError,
    SourceMeta, TuneRange,
};

/// RX channel the v1 backend streams (multi-channel devices use channel 0;
/// channel selection is FR-S7 territory, M3).
const CHANNEL: usize = 0;

/// Per-read timeout handed to Soapy, µs. Short enough that a silent device is
/// noticed promptly; long enough that a healthy stream almost never times out.
const READ_TIMEOUT_US: i64 = 200_000;

/// How long a device may deliver nothing at all before the stream is declared
/// dead. USB unplug usually fails a read outright; this grace period catches
/// drivers that degrade to endless timeouts instead.
const SILENT_DEVICE_GRACE: Duration = Duration::from_secs(5);

/// Minimum interval between overflow diagnostics on stderr — every event is
/// counted, but a sustained overflow storm must not flood the terminal.
const OVERFLOW_NOTE_INTERVAL: Duration = Duration::from_secs(5);

/// Read-buffer size in samples when the driver reports no usable MTU.
const FALLBACK_READ_SAMPLES: usize = 8192;

/// Upper bound on the read buffer, samples — guards against a driver
/// advertising an absurd MTU.
const MAX_READ_SAMPLES: usize = 1 << 18;

/// Honest accounting of what a [`SoapySource`] has done so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SoapySourceStats {
    /// Samples delivered to sinks, across all streaming sessions.
    pub samples_delivered: u64,
    /// Hardware overflow events (`SOAPY_SDR_OVERFLOW`): the driver discarded
    /// an unknown amount of data before delivery. Events only — no sample
    /// count is ever inferred (D-050).
    pub overflows: u64,
}

/// A SoapySDR device as a [`SampleSource`] (FR-S5, Pattern C). See the
/// [module docs](super) for the behaviour contract.
pub struct SoapySource {
    device: Device,
    config: SoapySourceConfig,
    info: SoapyDeviceInfo,
    /// Device readback (clarification C5) — refreshed after every set.
    sample_rate_hz: f64,
    /// Device readback — refreshed after every set.
    center_freq_hz: f64,
    /// The device's own centre-frequency range, read once at open (D-058) —
    /// `None` when the device reports none. Never a guess.
    tune_range: Option<TuneRange>,
    /// Live-control inbox (D-056/D-058), drained inside [`SoapySource::stream`]
    /// where this source holds `&mut self` and can touch the device.
    control: Option<ControlInbox>,
    stats: SoapySourceStats,
    last_overflow_note: Option<Instant>,
}

impl std::fmt::Debug for SoapySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoapySource")
            .field("config", &self.config)
            .field("info", &self.info)
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("center_freq_hz", &self.center_freq_hz)
            .field("tune_range", &self.tune_range)
            .field("stats", &self.stats)
            .finish()
    }
}

impl SoapySource {
    /// Enumerate, select, open, and configure a device.
    ///
    /// Zero matching devices is a [`SourceError::NoDevice`] naming what was
    /// searched — never a panic and never a hang. When several devices match,
    /// the first radio wins and stderr names the choice (FR-S8). Requested
    /// rate / centre / gain are applied through Soapy's API, then read back
    /// so the reported metadata is the device's own truth (C5).
    pub fn new(config: SoapySourceConfig) -> Result<SoapySource, SourceError> {
        config.validate()?;
        let requested =
            parse_device_args(&config.device_args).map_err(SourceError::InvalidConfig)?;
        let searched = format_device_args(&requested);
        let found = soapysdr::enumerate(searched.as_str()).map_err(|e| {
            SourceError::Io(format!(
                "SoapySDR device enumeration for \"{searched}\" failed: {e}"
            ))
        })?;
        let candidates: Vec<Vec<(String, String)>> = found
            .iter()
            .map(|args| {
                args.iter()
                    .map(|(k, v)| (k.to_owned(), v.to_owned()))
                    .collect()
            })
            .collect();
        let index = match select_device(&requested, &candidates) {
            Selection::Chosen { index } => index,
            Selection::OnlyAudio { count } => {
                return Err(SourceError::NoDevice(only_audio_message(&searched, count)))
            }
            Selection::None => return Err(SourceError::NoDevice(no_devices_message(&searched))),
        };
        if candidates.len() > 1 {
            eprintln!(
                "phosphene: {}",
                chosen_message(candidates.len(), &candidates[index])
            );
        }
        let info = SoapyDeviceInfo::from_pairs(&candidates[index]);
        let device = Device::new(found[index].iter().collect::<Args>()).map_err(|e| {
            SourceError::Io(format!(
                "cannot open SoapySDR device {}: {e}",
                info.describe()
            ))
        })?;

        if let Some(rate) = config.sample_rate_hz {
            device
                .set_sample_rate(Direction::Rx, CHANNEL, rate)
                .map_err(|e| {
                    SourceError::InvalidConfig(format!(
                        "device {} rejected sample rate {rate} Hz: {e}",
                        info.describe()
                    ))
                })?;
        }
        if let Some(freq) = config.center_freq_hz {
            device
                .set_frequency(Direction::Rx, CHANNEL, freq, ())
                .map_err(|e| {
                    SourceError::InvalidConfig(format!(
                        "device {} rejected center frequency {freq} Hz: {e}",
                        info.describe()
                    ))
                })?;
        }
        if let Some(gain) = config.gain_db {
            device.set_gain(Direction::Rx, CHANNEL, gain).map_err(|e| {
                SourceError::InvalidConfig(format!(
                    "device {} rejected gain {gain} dB: {e}",
                    info.describe()
                ))
            })?;
        }

        // D-058: the tunable range is the DEVICE'S answer or nothing at all.
        // A driver that reports no range leaves this `None` and the UI then
        // offers no bounded control — inventing limits would put a slider in
        // front of the user promising frequencies the radio cannot reach.
        // A failed query is treated the same way as "reports none": the
        // absence of a range is not a reason to refuse to stream.
        let tune_range = device
            .frequency_range(Direction::Rx, CHANNEL)
            .ok()
            .and_then(|ranges| TuneRange::hull(ranges.iter().map(|r| (r.minimum, r.maximum))));

        let mut source = SoapySource {
            device,
            config,
            info,
            sample_rate_hz: 0.0,
            center_freq_hz: 0.0,
            tune_range,
            control: None,
            stats: SoapySourceStats::default(),
            last_overflow_note: None,
        };
        source.refresh_readback()?;
        Ok(source)
    }

    /// The configuration this source was opened with.
    pub fn config(&self) -> &SoapySourceConfig {
        &self.config
    }

    /// What has happened so far — delivery and overflow accounting.
    pub fn stats(&self) -> SoapySourceStats {
        self.stats
    }

    /// Re-read rate and centre from the device — metadata is always the
    /// hardware's own answer, never an echo of the request (C5).
    fn refresh_readback(&mut self) -> Result<(), SourceError> {
        self.sample_rate_hz = self
            .device
            .sample_rate(Direction::Rx, CHANNEL)
            .map_err(|e| {
                SourceError::Io(format!(
                    "cannot read the sample rate back from {}: {e}",
                    self.info.describe()
                ))
            })?;
        self.center_freq_hz = self.device.frequency(Direction::Rx, CHANNEL).map_err(|e| {
            SourceError::Io(format!(
                "cannot read the center frequency back from {}: {e}",
                self.info.describe()
            ))
        })?;
        Ok(())
    }

    /// Apply one control to the device, then **verify it by reading the
    /// device back** (C5) — and say which of D-060's three states that left
    /// the radio in.
    ///
    /// **The single place a control is applied.** [`SampleSource::set`] (used
    /// between streaming sessions) and the live [`ControlHandle`] path (used
    /// while streaming) both land here — D-058's "two entry points, one
    /// mechanism": a second application path would be a second source of
    /// truth about where the radio is, and would drift.
    ///
    /// The ordering discipline itself lives in [`apply_and_verify`], not
    /// here: the bug D-060 records was an *ordering*, not a missing feature,
    /// and an ordering that each backend re-implements is one each backend
    /// can get wrong on its own. The readback is written into `self` only
    /// through the closure below, so a partial readback can never leave a
    /// field describing the device from before the tune.
    fn apply_control(&mut self, ctl: Control) -> ControlOutcome {
        let device = &self.device;
        let describe = self.info.describe();
        let mut sample_rate_hz = self.sample_rate_hz;
        let mut center_freq_hz = self.center_freq_hz;
        let outcome = apply_and_verify(
            &describe,
            &ctl,
            || match ctl {
                Control::CenterFreqHz(freq) => device
                    .set_frequency(Direction::Rx, CHANNEL, freq, ())
                    .map_err(|e| {
                        SourceError::InvalidConfig(format!(
                            "device {describe} rejected center frequency {freq} Hz: {e}"
                        ))
                    }),
                Control::SampleRateHz(rate) => device
                    .set_sample_rate(Direction::Rx, CHANNEL, rate)
                    .map_err(|e| {
                        SourceError::InvalidConfig(format!(
                            "device {describe} rejected sample rate {rate} Hz: {e}"
                        ))
                    }),
                Control::GainDb(gain) => {
                    device.set_gain(Direction::Rx, CHANNEL, gain).map_err(|e| {
                        SourceError::InvalidConfig(format!(
                            "device {describe} rejected gain {gain} dB: {e}"
                        ))
                    })
                }
                ref other => Err(SourceError::UnsupportedControl {
                    backend: "soapy",
                    control: other.name(),
                }),
            },
            || {
                sample_rate_hz = device.sample_rate(Direction::Rx, CHANNEL).map_err(|e| {
                    SourceError::Io(format!(
                        "cannot read the sample rate back from {describe}: {e}"
                    ))
                })?;
                center_freq_hz = device.frequency(Direction::Rx, CHANNEL).map_err(|e| {
                    SourceError::Io(format!(
                        "cannot read the center frequency back from {describe}: {e}"
                    ))
                })?;
                Ok(())
            },
        );
        self.sample_rate_hz = sample_rate_hz;
        self.center_freq_hz = center_freq_hz;
        outcome
    }

    /// Service every pending live control (D-056/D-058), from inside the
    /// stream loop where this source holds `&mut self`.
    ///
    /// Three rules, all honesty rules:
    ///
    /// * **The sample rate is fixed at start** (D-056, the owner's own
    ///   constraint): the ring size (NFR-P3), the FFT cadence and every
    ///   accumulator interval derive from it, so a live rate change is not a
    ///   retune but a pipeline rebuild — a different, much larger lane. The
    ///   live path refuses it in as many words. `set` between sessions is
    ///   untouched.
    /// * **The readback is announced in band.** When the applied control
    ///   moved the device's own answer, [`SampleSink::meta_changed`] carries
    ///   the new [`SourceMeta`] into the stream at exactly the sample where
    ///   it starts being true — the segment boundary spec §5.3 already
    ///   defines, and the point at which D-056's accumulator reset belongs.
    /// * **A refusal is answered as a refusal.** The caller gets the
    ///   device's own error and the readback (announced too, if the failed
    ///   attempt moved the device at all), never a silent `Ok`.
    /// * **A tune that cannot be read back ends the stream** (D-060). Once
    ///   the tuner has been mutated the old centre is no longer a claim this
    ///   program is entitled to make, so [`settle_control`] tells the sink
    ///   its accumulators are stale and hands back the error this returns —
    ///   the stream stops rather than keep drawing under a centre nothing
    ///   can substantiate.
    fn drain_controls(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
        while let Some(request) = self.control.as_ref().and_then(ControlInbox::try_recv) {
            let control = request.control().clone();
            let before = self.meta();
            let outcome = match control {
                Control::SampleRateHz(rate) => {
                    ControlOutcome::Rejected(SourceError::UnsupportedControl {
                        backend: "soapy (streaming)",
                        control: Control::SampleRateHz(rate).name(),
                    })
                }
                other => self.apply_control(other),
            };
            let after = self.meta();
            settle_control(outcome, request, sink, &before, &after)?;
        }
        Ok(())
    }

    /// Counted, rate-limited overflow diagnostic — every event increments
    /// [`SoapySourceStats::overflows`]; stderr sees at most one line per
    /// [`OVERFLOW_NOTE_INTERVAL`], with the running total.
    fn note_overflow(&mut self) {
        self.stats.overflows += 1;
        let due = match self.last_overflow_note {
            None => true,
            Some(at) => at.elapsed() >= OVERFLOW_NOTE_INTERVAL,
        };
        if due {
            eprintln!(
                "phosphene: {}: hardware overflow — the driver dropped samples \
                 before delivery ({} event(s) so far)",
                self.info.describe(),
                self.stats.overflows
            );
            self.last_overflow_note = Some(Instant::now());
        }
    }
}

impl SampleSource for SoapySource {
    fn open(desc: &SourceDesc) -> Result<Self, SourceError> {
        match desc {
            SourceDesc::Soapy(config) => SoapySource::new(config.clone()),
            _ => Err(SourceError::WrongBackend { backend: "soapy" }),
        }
    }

    /// Stream from the device into `sink` until the sink stops the session,
    /// the device disconnects, or the stream fails.
    ///
    /// This loop never stalls on the sink: the app's ring producer sheds and
    /// counts whole batches when full (D-013/NFR-P3), so sustained overload
    /// lands in the FR-D11 accounting at the ring, and the driver's own
    /// buffers stay serviced.
    ///
    /// Hardware overflow (`SOAPY_SDR_OVERFLOW`) means the device lost an
    /// **unknown** amount of data before delivery. Per **D-050** exactly
    /// that is reported: the *event* is counted in
    /// [`SoapySourceStats::overflows`], handed to
    /// [`SampleSink::device_overflow`], and noted on stderr (rate-limited).
    /// No sample count is ever inferred — SoapySDR's read path surfaces no
    /// timestamp-validity flag, so a quantity derived from `time_ns` could
    /// fabricate drops that never happened, the mirror image of the
    /// dishonesty D-048 exists to prevent. A device that fails or falls
    /// silent beyond [`SILENT_DEVICE_GRACE`] ends the stream with an error
    /// naming it.
    fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
        sink.meta_changed(&self.meta());
        let mut rx = self
            .device
            .rx_stream::<Complex<f32>>(&[CHANNEL])
            .map_err(|e| {
                SourceError::Io(format!(
                    "cannot open an RX stream on {}: {e}",
                    self.info.describe()
                ))
            })?;
        let read_samples = rx
            .mtu()
            .ok()
            .filter(|&mtu| mtu > 0)
            .unwrap_or(FALLBACK_READ_SAMPLES)
            .min(MAX_READ_SAMPLES);
        let mut buf = vec![Complex::new(0.0_f32, 0.0); read_samples];
        rx.activate(None).map_err(|e| {
            SourceError::Io(format!(
                "cannot activate the RX stream on {}: {e}",
                self.info.describe()
            ))
        })?;

        let mut last_delivery = Instant::now();
        let result = loop {
            // Live control, serviced once per read (D-056/D-058): the read
            // timeout bounds how long a request can sit here even on a
            // silent device, so a retune is answered promptly whether or not
            // samples are flowing.
            // D-060: a tune the device took but would not confirm ends the
            // stream here, before another sample is accepted under a centre
            // this program can no longer name.
            if let Err(e) = self.drain_controls(sink) {
                break Err(e);
            }
            match rx.read(&mut [&mut buf[..]], READ_TIMEOUT_US) {
                Ok(0) => {}
                Ok(n) => {
                    last_delivery = Instant::now();
                    self.stats.samples_delivered += n as u64;
                    if sink.push(&buf[..n]) == SinkFlow::Stop {
                        break Ok(());
                    }
                }
                Err(e) if e.code == ErrorCode::Overflow => {
                    // Samples are flowing (too fast, in fact). How many the
                    // device lost is unknowable through this binding, so
                    // only the event is reported (D-050).
                    last_delivery = Instant::now();
                    self.note_overflow();
                    sink.device_overflow();
                }
                Err(e) if e.code == ErrorCode::Timeout => {
                    if last_delivery.elapsed() >= SILENT_DEVICE_GRACE {
                        break Err(SourceError::Io(format!(
                            "{} delivered no samples for {} s — device unplugged \
                             or stream stalled?",
                            self.info.describe(),
                            SILENT_DEVICE_GRACE.as_secs()
                        )));
                    }
                }
                Err(e) => {
                    break Err(SourceError::Io(format!(
                        "streaming from {} failed: {e}",
                        self.info.describe()
                    )))
                }
            }
        };
        let _ = rx.deactivate(None);
        result
    }

    fn caps(&self) -> ControlCaps {
        ControlCaps {
            tune: true,
            rate: true,
            gain: true,
            antenna: false,
            // D-058: the device's own reported range, or `None`.
            tune_range: self.tune_range,
        }
    }

    fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
        match self.apply_control(ctl) {
            ControlOutcome::Applied => Ok(()),
            // Between streaming sessions there is no sink holding stale
            // history and no stream to end, but the error is the same one
            // and it still says the readback failed after the device moved.
            ControlOutcome::Rejected(e) | ControlOutcome::Unverified(e) => Err(e),
        }
    }

    fn control_handle(&mut self) -> Option<ControlHandle> {
        let (handle, inbox) = control_channel();
        self.control = Some(inbox);
        Some(handle)
    }

    fn meta(&self) -> SourceMeta {
        source_meta(&self.info, self.sample_rate_hz, self.center_freq_hz)
    }
}
