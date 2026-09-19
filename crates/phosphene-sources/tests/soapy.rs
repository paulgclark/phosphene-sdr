// SPDX-License-Identifier: MIT

//! Hardware-in-loop seal for the SoapySDR backend (lane S1-A).
//!
//! These tests need the `soapy` feature (gated in `Cargo.toml`) **and** an
//! attached radio: on a machine where enumeration finds no radio, every test
//! here skips with a note and passes, so `cargo test --features soapy` on a
//! bare machine stays green (the lane's runtime-check requirement). On the
//! capture nodes (SoapySDR 0.8.1, USRP B200mini via the SoapyUHD runtime
//! module) they exercise a real open → configure → stream → readback cycle.
//!
//! The "streams zeros" trap (D-031/D-032/D-033): a backend that opens a
//! device and delivers silence passes every unit test, so this seal asserts
//! the delivered samples carry actual energy and variation, not just the
//! right count.
//!
//! ## FL-7, 2026-09-15: radio-free reproduction attempt, no crash — stopped at the escalation gate
//!
//! Two SIGSEGV sightings on native SoapySDR code motivated this lane (plan_dag.md's FL-7 row).
//! Sighting 2 — `phosphene-app`'s `tests::soapy_source_runs_end_to_end_through_the_production_path`,
//! killed by signal 11 about 1.1 s in, right after UHD initialised and the SoapySDR audio module
//! failed to reach PulseAudio/ALSA, on a CI runner with no radio attached — is the reproducible one.
//!
//! **Method (FL-7's brief, ruled 2026-09-15):** a lane-local `SOAPY_SDR_PLUGIN_PATH` holding only
//! `libaudioSupport.so`, `SOAPY_SDR_ROOT` pointed at a nonexistent path, and a positive control
//! (`SoapySDRUtil --info`) confirmed before every run to list exactly that one module and no radio
//! driver. Under that path, `soapy_source_runs_end_to_end_through_the_production_path` was looped
//! 50 times in `gdb -batch` (the only backtrace route available: `core_pattern` pipes to apport on
//! the lane host, `ulimit -c` is 0, and `coredumpctl` is not installed).
//!
//! **Result: 0 crashes in 50 runs.** Each run resolved the same way sighting 2's own message
//! predicts for a bare enumeration: `select_device` saw only the audio pseudo-device, returned
//! `Selection::OnlyAudio`, and the test's `no SoapySDR radio` match arm skipped it in well under a
//! tenth of a second — never reaching `SoapySource::stream` at all.
//!
//! **Diagnosis, not a guess.** ⚠ **Corrected 2026-09-16 (Amendment 2):** the CI runner **has** a
//! working ALSA card too (measured: `0 [PCH] HDA Intel PCH`) — the original line here, claiming it
//! did not, was wrong. The CI audio failure is **environmental**: a service user with no PulseAudio
//! session and no device access, not a missing card. On this lane host, entering that same
//! environmental failure needs the same "no session, no access" condition rather than hiding the
//! card, and this environment has no permission to create the mount/user namespace that a first
//! attempt reached for (`unshare` refused with `EPERM`/`EPERM` on both the mount and user namespace
//! forms) — Amendment 1 reached the failure by environment variables instead (see below). The
//! sighting's log also shows UHD
//! initialising immediately beforehand, which the brief's own module list forbids reproducing here:
//! loading `libuhdSupport.so` on this lane host does not probe "no device" the way it did on the CI
//! runner — this host has real attached radios, so enumeration would find them, which is exactly
//! the `--radio` escalation the brief reserves for a scheduled window.
//!
//! **Stopping here, per the brief's own gate:** *"Only if the crash needs a radio driver loaded
//! with no radio attached … is the next step 'UHD support module, no device' … Stop and ask rather
//! than doing it here."* Zero crashes over 50 runs at the only module combination this host can
//! safely test is consistent with the crash needing exactly that condition. No fix is proposed by
//! this pass: without a native backtrace, changing `crates/phosphene-sources/src/soapy/**` would be
//! guessing, which D-093–D-095 rule out.
//!
//! ## FL-7 Amendment 1, 2026-09-16: the audio failure path reached by ENVIRONMENT, still no crash
//!
//! The master measured that `PULSE_SERVER=unix:/nonexistent/pulse` and
//! `ALSA_CONFIG_PATH=/nonexistent/asound.conf`, alongside the same audio-only narrowed path, force
//! `RtApiPulse::DeviceInfo`'s `pa_context_connect()` to fail and ALSA's `snd_ctl_open_noupdate` to
//! reject both `default` and `hw:0` — sighting 2's own log shape — so `SoapySDRUtil --find` now
//! reports **no** device at all (exit 1) instead of the audio pseudo-device. With no device found,
//! `select_device` cannot return `Selection::OnlyAudio` (there is nothing to be "only" audio among),
//! so the early skip that ended every run in the first pass should not fire the same way.
//!
//! **Re-run:** both variables set on every invocation, positive control (`SoapySDRUtil --info`,
//! confirming the module list and no radio driver) run first each time, unchanged. Looped
//! `soapy_source_runs_end_to_end_through_the_production_path` under `gdb -batch` for **150 runs**
//! (50, then 100 more) with `--nocapture` to see exactly where each run resolves.
//!
//! **Result: 0 crashes in 150 runs.** Every run reproduced the master's measured native log lines
//! verbatim (`pa_context_connect() failed: Connection refused`, then the two ALSA `conf.c`/
//! `control.c` errors for `default` and `hw:0`), then `SoapySource::new`'s `enumerate("")` returned
//! an empty list — `Selection::None`, not `OnlyAudio` — and the test's `no SoapySDR devices` match
//! arm skipped it, in 0.04–0.06 s each time. So the amendment's env vars do reach sighting 2's exact
//! failure text on this host, but that text alone is not sufficient to crash here.
//!
//! **Diagnosis, unchanged from the first pass.** The one difference this host still cannot supply
//! is UHD initialising immediately beforehand (sighting 2's log: "right after UHD initialised …").
//! Loading `libuhdSupport.so` on this lane host does not probe "no device" the way it did on the CI
//! runner — this host has real attached radios, so enumeration would find and interact with them,
//! which the brief reserves for a scheduled `--radio` window, not a bisection step available here.
//!
//! **Stopping here, per Amendment 1's own instruction:** *"If it still does not crash after a
//! comparable loop count, record that with the counts and stop and ask. The remaining hypothesis is
//! UHD-with-no-device, which this host cannot test safely."* 150 runs at 0 crashes, with the native
//! failure text matching sighting 2's own log shape, is that comparable count. No fix is proposed:
//! there is still no backtrace to fix from.
//!
//! ## FL-7 Amendment 2, 2026-09-16: reproduced on the CI runner itself, still no crash
//!
//! The remaining difference — UHD initialising with no device, on a runner with no radio attached —
//! could not be tested on the lane host at all (it has real attached radios), so Amendment 2 moved
//! the reproduction to the CI runner itself, through a temporary `workflow_dispatch`-only job
//! (`fl7-soapy-crash-repro`, since removed — see the workflow's own git history, run 35045128931).
//! That job ran in **CI's own default module environment**, no narrowed path: `SoapySDRUtil --info`
//! logged all twelve installed modules including `libuhdSupport.so`, matching sighting 2's own CI
//! image exactly.
//!
//! **Result: 0 crashes in 200 runs, 174 s elapsed** (well inside the 15-minute/200-run cap). Every
//! run's output was **byte-identical**: `[INFO] [UHD] linux; …` (UHD initialising, as in sighting
//! 2's log), then `pa_context_connect() failed: Connection refused`, then ALSA failing to find card
//! `0` (the CI runner's own environmental audio failure, corrected from this file's first pass —
//! see below), then the test's `ok` in 0.51 s every time.
//!
//! **Correction (Amendment 2, item 0):** the first pass's diagnosis above claimed sighting 2's CI
//! runner had no working ALSA card. Measured by the master: it has one (`0 [PCH] HDA Intel PCH`).
//! The CI audio failure is **environmental** — the job runs as a service user with no PulseAudio
//! session and no device access — not a missing card. This CI job's own run confirms it: the same
//! service-user environment reproduced that exact failure shape (`cannot find card '0'` — a slightly
//! different ALSA error text than Amendment 1's forced `/nonexistent/asound.conf`, but the same "no
//! usable audio device" outcome) without any environment variable forcing at all, simply by being
//! the CI job it always was.
//!
//! **So sighting 2's exact reported conditions — UHD initialising, the audio module failing to
//! reach PulseAudio/ALSA, no radio attached — are now reproduced verbatim, natively, on the runner
//! class where it happened, and it still does not crash.** This is not evidence the crash cannot
//! happen; a single historical sighting against 200 clean runs is consistent with a rare or
//! host/timing-dependent fault (a specific one of the "five runners" sighting 2's box may have run
//! on, a driver/library version since patched, resource contention from a concurrent leg on the
//! same physical host that a solitary manual dispatch does not reproduce, or a race intrinsic to
//! `libusb`/UHD's own device enumeration that 200 serial runs simply did not hit).
//!
//! **Stopping here, per Amendment 2's own instruction:** *"No crash within the cap: record the
//! counts and the job log, and stop and ask. Do not raise the cap."* No fix is proposed: there is
//! still no backtrace, and inventing one from either sighting's log text alone would be exactly the
//! guessing D-093–D-095 rule out. The crash-capture path (apport-unpack, then reading the CoreDump's
//! frames with `gdb` on a separate machine of the same distribution and architecture) was built into
//! the temporary job but never exercised, because no crash occurred to capture.

//! ## CI-C, 2026-09-16: an in-process SIGSEGV handler makes the next crash readable
//!
//! Both FL-7 amendments above, plus a third sighting (`plan_dag.md`'s FL-7 row) crashed with no
//! backtrace anywhere a human could read one: the runner's crash handler (apport) writes no report
//! for a `cargo test` binary, because it ignores crashes from unpackaged binaries — measured, not
//! assumed (CI-C's lane brief). Rebuilding the apport-collection route was ruled out for that reason.
//!
//! Instead, this binary installs its own `SIGSEGV` handler ([`crash_frames::install`]), called at the
//! top of every `#[test]` below. On a fault it prints frames to stderr with only functions documented
//! safe to call from a signal handler — `write` and glibc's `backtrace_symbols_fd` (which, unlike
//! `backtrace_symbols`, never calls `malloc`) — then restores the default disposition and re-raises,
//! so the crash still kills the process with the same signal: the leg still fails, it now fails with
//! frames. `backtrace()` itself is not unconditionally async-signal-safe (its first call in a process
//! can lazily load unwind tables), so `install()` also calls it once, outside signal context, purely
//! to force that one-time cost to happen before any handler ever runs.
//!
//! **Amendment 1 (federator, 2026-09-16): this handler is the PRIMARY path, not a fallback,** because
//! a lane that waits on an awkward human action on a shared host waits forever — `.github/workflows/
//! ci.yml`'s `gdb` wrapping stays a conditional nicety behind `command -v gdb`, and nothing in this
//! crate's registry seal depends on it. **Amendment 2 (federator, 2026-09-16): gdb 15.1 is now
//! installed on the affected CI node** (the owner installed it; Amendment 1's worry that it might
//! never be is superseded, its conclusion is not), with `ptrace_scope` `1` there, so the step is
//! written as gdb *launching* the test (`--args`), never attaching to a running pid, which that
//! `ptrace_scope` refuses. Both paths now run for real on that node — they answer different failures
//! (a debugger absent or refused, versus a stack the handler cannot walk), so if the two ever disagree
//! on a captured crash, that is a finding to report, not a reason to drop either. ⚠ **The hard case,
//! ruled in advance:** a real crash inside `libusb`/UHD's native code can leave a stack
//! `backtrace_symbols_fd` cannot walk past — a short or empty frame list between the banner and the
//! footer. That is a **finding to report** (name where the walk stopped), not evidence the handler is
//! broken, and never a reason to wait on a debugger appearing on some other node instead.
//!
//! **Evidence — deliberate crash, this host, 2026-09-16**, run against this actual `cargo test`
//! binary (`PHOSPHENE_CI_C_DELIBERATE_CRASH=1 cargo test -p phosphene-sources --features soapy
//! --test soapy -- --ignored --exact --nocapture
//! deliberate_sigsegv_is_captured_with_frames_and_still_fails`), frames quoted verbatim (the build
//! path prefix common to every line is elided as `…`; it is this host's own local path, not part of
//! the evidence):
//! ```text
//! ===PHOSPHENE-CRASH-FRAMES=== SIGSEGV in this test binary
//! …/soapy-f5e978f0a2e3fc90(+0x66432)[0x621841a8d432]
//! /lib/x86_64-linux-gnu/libc.so.6(+0x45330)[0x7a507ae45330]
//! …/soapy-f5e978f0a2e3fc90(+0x614c9)[0x621841a884c9]
//! …/soapy-f5e978f0a2e3fc90(+0x68c18)[0x621841a8fc18]
//! …/soapy-f5e978f0a2e3fc90(+0x64787)[0x621841a8b787]
//! …/soapy-f5e978f0a2e3fc90(+0x61416)[0x621841a88416]
//! …/soapy-f5e978f0a2e3fc90(+0x6ac6b)[0x621841a91c6b]
//! …/soapy-f5e978f0a2e3fc90(+0x782b5)[0x621841a9f2b5]
//! …/soapy-f5e978f0a2e3fc90(+0x71c64)[0x621841a98c64]
//! …/soapy-f5e978f0a2e3fc90(+0x7b402)[0x621841aa2402]
//! …/soapy-f5e978f0a2e3fc90(+0x112ebf)[0x621841b39ebf]
//! /lib/x86_64-linux-gnu/libc.so.6(+0x9cb84)[0x7a507ae9cb84]
//! /lib/x86_64-linux-gnu/libc.so.6(+0x129ecc)[0x7a507af29ecc]
//! ===PHOSPHENE-CRASH-FRAMES-END===
//! ```
//! followed by `cargo test`'s own report: `error: test failed …  Caused by: process didn't exit
//! successfully … (signal: 11, SIGSEGV: invalid memory reference)`, exit 101 — the leg still fails,
//! and now with frames. (Frames are hex offsets into this test binary, not resolved symbol names:
//! `backtrace_symbols_fd` resolves against the dynamic symbol table via `dladdr`, which carries less
//! for a Rust binary than DWARF debug info would; even at this shape the binary plus offset is enough
//! to locate the fault with `addr2line` against the same build, which is the point — readable at all
//! beats the nothing every prior sighting left behind.)
//!
//! **Evidence — a clean run prints and captures nothing.** The same invocation without the env guard
//! set (`cargo test -p phosphene-sources --features soapy --test soapy -- --ignored --exact
//! --nocapture deliberate_sigsegv_is_captured_with_frames_and_still_fails`) printed only `SKIP (by
//! design): set PHOSPHENE_CI_C_DELIBERATE_CRASH=1 …` and `test result: ok`, exit 0; no
//! `PHOSPHENE-CRASH-FRAMES` banner. A normal `cargo test --features soapy` run (no `--ignored`) never
//! reaches the deliberate test at all, and the registry seal's own soapy-feature step (which `--skip`s
//! every named hardware test, this one included by never un-ignoring it) shows the same: 0 passed, 0
//! failed, 1 ignored, nothing printed.
//!
//! **Evidence — the debugger path (`ci.yml`'s wrapping of the OTHER historically-crashing binary,
//! sighting 2, `phosphene-app`'s own soapy test — outside this lane's territory, so it has no
//! in-process handler and gdb is its only coverage), this host, 2026-09-16.** The exact dispatch
//! `ci.yml` now runs — a temp-generated `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER` script
//! invoking `gdb -batch -x <script> --args "$@"`, set only for one `cargo test` invocation — was run
//! for real against `phosphene-app`'s actual compiled binary (not a stand-in): the runner script's own
//! marker line confirmed Cargo invoked it, gdb's startup banner and thread-creation messages appeared
//! ahead of the test's own output, the test still resolved exactly as it does unwrapped (`no SoapySDR
//! devices found` → skip → `ok`, this host having none of the modules FL-7's own investigation ruled
//! out here), and `cargo test` reported `1 passed; 0 failed`, exit 0 — Cargo's end of
//! `CARGO_TARGET_*_RUNNER` dispatches to the generated script correctly, and gdb's own `quit
//! $_exitcode` branch (taken because `$_siginfo` was void: no signal, a normal exit) passed the real
//! exit code straight through, unmasked.
//!
//! Reproducing sighting 2's actual SIGSEGV was not attempted here: FL-7's own lane (this file's
//! Amendments above) already spent 400 runs establishing that this class of host cannot enter that
//! failure natively without the real CI runner's UHD-with-no-device condition, and forcing it would
//! be exactly the guessing D-093–D-095 rule out. The `$_siginfo`/`thread apply all bt full` branch
//! itself — the part `cargo`'s own dispatch does not exercise on a clean pass — was validated
//! separately against a throwaway, deliberately-segfaulting probe carrying no handler of its own
//! (mirroring `phosphene-app`'s binary): same script, same dispatch shape, `$_isvoid ($_siginfo)`
//! false, banner printed, full thread backtrace with source lines shown, exit 1. (An earlier draft
//! checked `$_isvoid ($_exitsignal)` instead and errored: GDB stops the inferior on a fatal signal by
//! default rather than letting it exit, so `$_exitsignal` stays void at that point — `$_siginfo` is
//! the convenience variable that is actually set while the inferior is merely stopped.)
//!
//! ## Sighting 4 (2026-09-16, 21:13:46Z–21:14:57Z UTC): FL-7's real crash, caught live, with a named frame
//!
//! Not deliberate: this happened during PR #52's own registry-seal CI run (`test (linux-x11)`, run
//! 35146838721), the ordinary `--skip soapy_source_runs_end_to_end_through_the_production_path`
//! step, immediately after `forced_overflow_is_counted_and_reported_to_the_sink` and during
//! `nonsense_device_args_error_names_what_was_searched`'s `SoapySource::new` call (`driver=uhd,serial=
//! DOES_NOT_EXIST`) — the same "audio module fails to reach PulseAudio/ALSA" log shape every prior
//! sighting shares, this time followed by an actual fault instead of a clean enumeration. The handler
//! caught it. Frames, quoted verbatim:
//! ```text
//! ===PHOSPHENE-CRASH-FRAMES=== SIGSEGV in this test binary
//! …/soapy-fb481c94c0aed989(+0x63af2)[0x5c3f401f4af2]
//! /lib/x86_64-linux-gnu/libc.so.6(+0x45330)[0x7b0842e45330]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(_ZN3uhd9transport17usb_device_handle15get_device_listERKSt6vectorISt4pairIttESaIS4_EE+0x91)[0x7b083bc28311]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(_ZN3uhd9transport17usb_device_handle15get_device_listEtt+0x6b)[0x7b083bc2920b]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(+0x6dbcfc)[0x7b083badbcfc]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(+0x678ba7)[0x7b083ba78ba7]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(+0x864504)[0x7b083bc64504]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(+0x33e8e2)[0x7b083b73e8e2]
//! /lib/x86_64-linux-gnu/libc.so.6(+0xa1fb3)[0x7b0842ea1fb3]
//! /lib/x86_64-linux-gnu/libuhd.so.4.6.0(+0x8692cb)[0x7b083bc692cb]
//! /lib/x86_64-linux-gnu/libstdc++.so.6(+0xecdb4)[0x7b0842aecdb4]
//! /lib/x86_64-linux-gnu/libc.so.6(+0x9cb84)[0x7b0842e9cb84]
//! /lib/x86_64-linux-gnu/libc.so.6(+0x129ecc)[0x7b0842f29ecc]
//! ===PHOSPHENE-CRASH-FRAMES-END===
//! ```
//! **This names a function for the first time in FL-7's history:** the second and third frames
//! demangle to `uhd::transport::usb_device_handle::get_device_list(std::vector<std::pair<unsigned
//! short, unsigned short>, ...> const&)` and its overload taking a bare vendor/product ID pair —
//! UHD's own USB device enumeration, called from inside `libuhd.so`'s device-discovery path. Three
//! prior sightings inferred "module probing or device open/close" from log proximity alone; this is
//! the first frame that says so directly. `cargo test` then reported the binary killed by signal 11,
//! exit 101 — unmasked, exactly as designed — and the leg failed. Re-run once (job 104980730543),
//! per this project's own established FL-7 practice ("one named re-run cleared the leg" —
//! plan_dag.md's FL-7 row): **0 crashes, green**, matching FL-7's own extensive record of this being
//! rare and non-reproducible on demand. Run 35146838721 finished green overall.
//!
//! **Not in this lane:** diagnosing or fixing the `usb_device_handle::get_device_list` fault itself.
//! CI-C's brief is explicit that this lane makes the next crash readable and stops there; the frame
//! above is exactly the readable "next crash" the brief describes, handed to whoever picks up FL-7
//! next.
//!
//! ## Fix round (2026-09-16): a crash alone is not proof of capture — the oracle and its negative control
//!
//! Review 1 (judged 0d3cc63) blocked on this: every test here that calls [`crash_frames::install`]
//! only proves the handler was *invoked to install itself*, never that it actually *caught and
//! printed* anything. Delete all five `crash_frames::install();` call sites and nothing before this
//! fix would have noticed — [`deliberate_sigsegv_is_captured_with_frames_and_still_fails`] still dies
//! of `SIGSEGV` with no handler installed at all (the default disposition kills it exactly the same
//! way), and every other test still just skips for lack of a radio. A crash is not evidence of a
//! capture.
//!
//! [`crash_frames_banner_is_the_oracle_for_deliberate_sigsegv`] is the fix: it spawns this same
//! binary as a child (`std::env::current_exe()`), runs only the deliberate-crash test in it, and
//! asserts on the child's own output — the only place the difference between "crashed" and "crashed
//! *and captured*" is visible from outside a process that is, by definition, about to die.
//!
//! **The red control, reproduced and quoted verbatim (D-094/D-095: it must be reproducible, not just
//! asserted).** All five `crash_frames::install();` call sites deleted with `sed -i
//! '/^    crash_frames::install();$/d'`, the file otherwise byte-identical, then
//! `cargo test -p phosphene-sources --features soapy --test soapy -- --exact --nocapture
//! crash_frames_banner_is_the_oracle_for_deliberate_sigsegv`:
//! ```text
//! thread 'crash_frames_banner_is_the_oracle_for_deliberate_sigsegv' panicked at
//! crates/phosphene-sources/tests/soapy.rs:759:5:
//! the child crashed as expected but printed no PHOSPHENE-CRASH-FRAMES banner — the in-process
//! handler did not run. This is exactly the negative control for crash_frames::install(): a crash
//! alone is not proof of capture.
//! stderr:
//!
//! test crash_frames_banner_is_the_oracle_for_deliberate_sigsegv ... FAILED
//! test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out
//! ```
//! The child's exit status still carried signal 11 (the first assertion passed silently, exactly as
//! the brief predicts: "crash still SIGSEGVs"); the second assertion, on the banner's presence, is
//! what turned red. Restoring the five deleted lines (verified byte-identical to the pre-mutation
//! file) turned the same test green again in the same run, same host, same binary.
//!
//! **Not `#[ignore]`d, unlike the test it supervises.** The oracle's own process never crashes — only
//! its child does, fully contained — so it runs in every normal `cargo test --features soapy`,
//! including the registry seal (its name is not in that seal's `--skip` list, and it touches no
//! radio, so there is nothing to skip it for).

#![cfg(feature = "soapy")]

use std::sync::Mutex;
use std::time::Instant;

use phosphene_sources::source::{Control, SampleSink, SampleSource, SinkFlow, SourceMeta};
use phosphene_sources::{Complex, SoapySource, SoapySourceConfig};

/// An in-process `SIGSEGV` handler for this test binary only (CI-C): the runner's crash handler
/// (apport) never captures an unpackaged `cargo test` binary, so this binary captures its own
/// crash frames instead of leaving nothing readable behind. See this file's module doc for the
/// full rationale and quoted evidence.
///
/// Every file directly under `tests/` compiles as its own binary crate, separate from the
/// library: `src/lib.rs`'s `#![forbid(unsafe_code)]` binds `phosphene-sources` the library, not
/// this crate, so the raw FFI below (no way to call `sigaction`/`backtrace`/`backtrace_symbols_fd`
/// from safe Rust) does not conflict with that rule.
mod crash_frames {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Once;

    /// SIGSEGV's signal number is 11 on both Linux and macOS (POSIX), so this is portable without
    /// pulling in a signal-constant crate.
    const SIGSEGV: c_int = 11;
    const MAX_FRAMES: usize = 128;

    /// Where frames go. Stderr by default; a plain atomic, never a `Mutex`, because the handler
    /// must not block.
    static OUT_FD: AtomicI32 = AtomicI32::new(2);

    extern "C" {
        /// Raw, unsymbolicated frame walk — no allocation.
        fn backtrace(buffer: *mut *mut c_void, size: c_int) -> c_int;
        /// glibc/BSD libc: writes symbolised frames directly to `fd`, never calls `malloc` (unlike
        /// `backtrace_symbols`), and is the documented safe-for-a-signal-handler alternative.
        fn backtrace_symbols_fd(buffer: *const *mut c_void, size: c_int, fd: c_int);
        fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
        fn raise(sig: c_int) -> c_int;
        /// The plain, one-argument `signal()`: handler and return value both spelled as `usize` so
        /// the FFI signature needs no platform-specific `sighandler_t` type — `SIG_DFL` is the null
        /// pointer (`0`) on every POSIX libc this project targets.
        fn signal(signum: c_int, handler: usize) -> usize;
    }

    extern "C" fn handle_segv(_sig: c_int) {
        // Async-signal-safe from here down: no allocation, no locks, no formatting, no branching
        // that could panic — only raw byte writes and the two libc calls documented safe for
        // this use.
        unsafe {
            let fd = OUT_FD.load(Ordering::Relaxed);
            const BANNER: &[u8] = b"\n===PHOSPHENE-CRASH-FRAMES=== SIGSEGV in this test binary\n";
            write(fd, BANNER.as_ptr().cast(), BANNER.len());

            let mut frames: [*mut c_void; MAX_FRAMES] = [std::ptr::null_mut(); MAX_FRAMES];
            let n = backtrace(frames.as_mut_ptr(), MAX_FRAMES as c_int);
            if n > 0 {
                backtrace_symbols_fd(frames.as_ptr(), n, fd);
            }
            const FOOTER: &[u8] = b"===PHOSPHENE-CRASH-FRAMES-END===\n";
            write(fd, FOOTER.as_ptr().cast(), FOOTER.len());

            // Restore the default disposition and re-raise: this handler must never mask the
            // crash or change the exit status, only make it readable. The process still dies of
            // SIGSEGV, exactly as it would have with no handler installed.
            signal(SIGSEGV, 0 /* SIG_DFL */);
            raise(SIGSEGV);
        }
    }

    static INSTALLED: Once = Once::new();

    /// Installs the handler once per process. Idempotent and cheap ([`Once`]) — call it at the top
    /// of every `#[test]` in this binary; "only for the soapy test binary" just means "only called
    /// from this file".
    pub fn install() {
        INSTALLED.call_once(|| unsafe {
            // Force whatever one-time lazy initialisation `backtrace()` needs (glibc may load
            // unwind tables on its first call in a process) to happen here, outside signal
            // context, so the call inside `handle_segv` above never does it under a fault.
            let mut warm: [*mut c_void; 1] = [std::ptr::null_mut()];
            backtrace(warm.as_mut_ptr(), 1);

            signal(SIGSEGV, handle_segv as *const () as usize);
        });
    }
}

/// One radio, many tests: the harness runs tests in parallel by default, and
/// two tests opening the same USRP concurrently fight over the USB claim.
/// Every test that opens a device holds this lock (poisoning ignored — a
/// failed test must not cascade).
static DEVICE: Mutex<()> = Mutex::new(());

/// Requested capture configuration for the seal: modest rate the B200mini
/// (and most radios) accepts, tuned into the FM broadcast band.
const RATE_HZ: f64 = 2_048_000.0;
const CENTER_HZ: f64 = 100_000_000.0;
const GAIN_DB: f64 = 40.0;
/// One second of stream time.
const WANT_SAMPLES: usize = 2_048_000;

/// True when a real radio (anything but Soapy's audio pseudo-device) answers
/// enumeration — the runtime half of the hardware gate.
fn radio_present() -> bool {
    match soapysdr::enumerate("") {
        Ok(found) => found
            .iter()
            .any(|args| args.get("driver").is_some_and(|d| d != "audio")),
        Err(_) => false,
    }
}

struct Collect {
    samples: Vec<Complex<f32>>,
    want: usize,
    meta: Option<SourceMeta>,
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

    fn meta_changed(&mut self, meta: &SourceMeta) {
        self.meta = Some(meta.clone());
    }
}

#[test]
fn capture_delivers_real_samples_and_honest_readback_meta() {
    crash_frames::install();
    let _device = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    if !radio_present() {
        eprintln!("SKIP: no SoapySDR radio attached — hardware seal not exercised");
        return;
    }

    let mut config = SoapySourceConfig::new("");
    config.sample_rate_hz = Some(RATE_HZ);
    config.center_freq_hz = Some(CENTER_HZ);
    config.gain_db = Some(GAIN_DB);
    let mut source = SoapySource::new(config).expect("open the attached radio");

    // Metadata is device readback (C5): a hardware source always knows.
    let meta = source.meta();
    let rate = meta.sample_rate_hz.expect("hardware knows its rate");
    let center = meta.center_freq_hz.expect("hardware knows its centre");
    assert!(
        (rate - RATE_HZ).abs() / RATE_HZ < 0.01,
        "device readback rate {rate} Hz is not the requested {RATE_HZ} Hz"
    );
    assert!(
        (center - CENTER_HZ).abs() < 1_000.0,
        "device readback centre {center} Hz is not the requested {CENTER_HZ} Hz"
    );
    assert!(
        meta.provenance.contains("soapysdr"),
        "provenance: {}",
        meta.provenance
    );

    let mut sink = Collect {
        samples: Vec::with_capacity(WANT_SAMPLES),
        want: WANT_SAMPLES,
        meta: None,
    };
    let started = Instant::now();
    source
        .stream(&mut sink)
        .expect("stream one second of samples");
    let elapsed = started.elapsed().as_secs_f64();

    // meta_changed must precede the first push.
    assert_eq!(
        sink.meta.as_ref().and_then(|m| m.sample_rate_hz),
        Some(rate)
    );

    assert!(
        sink.samples.len() >= WANT_SAMPLES,
        "sink asked to stop at {WANT_SAMPLES} samples but got {}",
        sink.samples.len()
    );
    // A real-time source cannot deliver a second of samples much faster than
    // a second, and a healthy one should not need more than a few.
    assert!(
        (0.8..10.0).contains(&elapsed),
        "1 s of stream time took {elapsed:.2} s of wall time — not a live capture?"
    );

    // The anti-zeros seal (D-031/D-032/D-033): real RF at 40 dB gain in the
    // FM band has energy and variation. All-zero or constant output — the
    // classic dead backend — fails all three asserts.
    let n = sink.samples.len() as f64;
    let mean_power: f64 = sink
        .samples
        .iter()
        .map(|s| f64::from(s.re * s.re + s.im * s.im))
        .sum::<f64>()
        / n;
    let nonzero = sink
        .samples
        .iter()
        .filter(|s| s.re != 0.0 || s.im != 0.0)
        .count();
    let distinct_re = {
        let mut seen: Vec<f32> = sink.samples.iter().take(4096).map(|s| s.re).collect();
        seen.sort_by(f32::total_cmp);
        seen.dedup();
        seen.len()
    };
    assert!(
        mean_power > 1e-9,
        "mean power {mean_power:e} — the stream is (near-)silent"
    );
    assert!(
        mean_power < 4.0,
        "mean power {mean_power:e} — not full-scale-normalised cf32?"
    );
    assert!(
        nonzero as f64 / n > 0.5,
        "only {nonzero} of {n} samples are non-zero — dead stream"
    );
    assert!(
        distinct_re > 100,
        "only {distinct_re} distinct values in 4096 samples"
    );
    assert!(
        sink.samples
            .iter()
            .all(|s| s.re.is_finite() && s.im.is_finite()),
        "non-finite samples delivered"
    );

    let stats = source.stats();
    assert!(stats.samples_delivered >= WANT_SAMPLES as u64);
    eprintln!(
        "hardware seal: {} samples in {elapsed:.2} s at {rate} S/s, centre {center} Hz, \
         gain {GAIN_DB} dB, {} overflow event(s), mean power {mean_power:.3e}",
        stats.samples_delivered, stats.overflows
    );
}

#[test]
fn set_between_sessions_retunes_and_meta_follows_the_device() {
    crash_frames::install();
    let _device = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    if !radio_present() {
        eprintln!("SKIP: no SoapySDR radio attached — hardware seal not exercised");
        return;
    }

    let mut config = SoapySourceConfig::new("");
    config.sample_rate_hz = Some(RATE_HZ);
    config.center_freq_hz = Some(CENTER_HZ);
    let mut source = SoapySource::new(config).expect("open the attached radio");
    assert!(source.caps().tune && source.caps().rate && source.caps().gain);
    assert!(!source.caps().antenna);

    source
        .set(Control::CenterFreqHz(98_000_000.0))
        .expect("retune between sessions");
    let center = source
        .meta()
        .center_freq_hz
        .expect("hardware knows its centre");
    assert!(
        (center - 98_000_000.0).abs() < 1_000.0,
        "meta centre {center} Hz did not follow the retune"
    );

    // Controls the backend does not advertise are rejected, not ignored.
    let err = source
        .set(Control::Antenna("RX2".to_owned()))
        .expect_err("antenna is not advertised");
    assert!(err.to_string().contains("antenna"), "unhelpful: {err}");
}

/// A sink that stalls the source thread mid-stream — starving the driver so
/// the device genuinely overflows — and records every
/// [`SampleSink::device_overflow`] event (and any [`SampleSink::device_lost`]
/// call, which per D-050 must never happen through this binding).
struct StallingSink {
    samples: u64,
    want: u64,
    stall: Option<std::time::Duration>,
    overflow_events: u64,
    lost_claimed: u64,
}

impl SampleSink for StallingSink {
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
        self.samples += samples.len() as u64;
        if let Some(stall) = self.stall.take() {
            std::thread::sleep(stall);
        }
        if self.samples >= self.want {
            SinkFlow::Stop
        } else {
            SinkFlow::Continue
        }
    }

    fn device_lost(&mut self, samples: u64) {
        self.lost_claimed += samples;
    }

    fn device_overflow(&mut self) {
        self.overflow_events += 1;
    }
}

/// D-050 at the hardware: a real overflow (forced by stalling the source
/// thread at 16 MS/s) reaches the sink as counted EVENTS — and no sample
/// quantity is ever claimed, because this binding cannot validate one.
#[test]
fn forced_overflow_is_counted_and_reported_to_the_sink() {
    crash_frames::install();
    let _device = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    if !radio_present() {
        eprintln!("SKIP: no SoapySDR radio attached — hardware seal not exercised");
        return;
    }

    let mut config = SoapySourceConfig::new("");
    config.sample_rate_hz = Some(16_000_000.0);
    config.center_freq_hz = Some(CENTER_HZ);
    let mut source = SoapySource::new(config).expect("open the attached radio");

    let mut sink = StallingSink {
        samples: 0,
        want: 16_000_000, // one second of stream time, minus the hole
        stall: Some(std::time::Duration::from_millis(500)),
        overflow_events: 0,
        lost_claimed: 0,
    };
    source
        .stream(&mut sink)
        .expect("stream across the overflow");

    let stats = source.stats();
    eprintln!(
        "overflow seal: {} overflow event(s), {} reached the sink, {} delivered",
        stats.overflows, sink.overflow_events, sink.samples
    );
    assert!(
        stats.overflows > 0,
        "a 500 ms stall at 16 MS/s did not overflow the device — stall harder"
    );
    assert_eq!(
        stats.overflows, sink.overflow_events,
        "the sink did not receive exactly the events the device reported"
    );
    // D-050: no quantity may be invented — this binding cannot validate
    // timestamps, so `device_lost` must never fire.
    assert_eq!(
        sink.lost_claimed, 0,
        "a sample count was claimed from an unvalidated source"
    );
}

#[test]
fn nonsense_device_args_error_names_what_was_searched() {
    crash_frames::install();
    // Needs libSoapySDR (enumeration) but no radio: a nonsense serial must
    // produce the lane's clear zero-devices error, not a panic or a hang.
    let mut config = SoapySourceConfig::new("driver=uhd,serial=DOES_NOT_EXIST");
    config.sample_rate_hz = Some(RATE_HZ);
    let err = match SoapySource::new(config) {
        Err(e) => e.to_string(),
        Ok(_) => {
            eprintln!("SKIP: a device actually matched the nonsense serial");
            return;
        }
    };
    assert!(
        err.contains("driver=uhd,serial=DOES_NOT_EXIST"),
        "error does not name what was searched: {err}"
    );
    assert!(err.contains("SoapySDRUtil --find"), "unactionable: {err}");
}

/// CI-C evidence only (D-122: evidence lives in the branch): deliberately faults this process so
/// [`crash_frames`]'s handler can be exercised end to end and its frames quoted in this file's
/// module doc. Never runs in the normal seal: `#[ignore]` keeps a bare `cargo test` away from it,
/// and the env guard keeps `--include-ignored` from firing it by accident too — both have to be
/// defeated on purpose for this test to actually crash anything.
#[test]
#[ignore = "deliberately SIGSEGVs this process — CI-C evidence only, see the module doc"]
fn deliberate_sigsegv_is_captured_with_frames_and_still_fails() {
    if std::env::var("PHOSPHENE_CI_C_DELIBERATE_CRASH").as_deref() != Ok("1") {
        eprintln!(
            "SKIP (by design): set PHOSPHENE_CI_C_DELIBERATE_CRASH=1 to actually crash this \
             process — see this file's module doc (CI-C) for why"
        );
        return;
    }
    crash_frames::install();
    // A real fault, not a panic: a volatile read through a null pointer, which the optimiser
    // cannot remove and the process cannot survive.
    let p: *const u8 = std::ptr::null();
    unsafe {
        std::ptr::read_volatile(p);
    }
    unreachable!("SIGSEGV should have terminated the process on the line above");
}

/// The oracle for [`deliberate_sigsegv_is_captured_with_frames_and_still_fails`]: a crash alone is
/// not proof of capture. A process that dies of `SIGSEGV` with no handler installed at all looks
/// identical, from the outside, to one whose handler ran and printed nothing — both are just "killed
/// by signal 11". Only reading the child's own output can tell the two apart, so this test spawns
/// this same compiled binary as a child (`std::env::current_exe()`, the standard way a test
/// re-invokes itself in isolation), runs only the deliberate-crash test in it, and asserts on BOTH
/// halves: the child still died of `SIGSEGV` (the crash itself), AND its output actually carries the
/// `PHOSPHENE-CRASH-FRAMES` banner (the capture). This is a **negative control** for every
/// `crash_frames::install()` call in this file: delete all five and the child still SIGSEGVs (this
/// test's first assertion keeps passing), but the handler never ran, so no banner appears, and this
/// test's second assertion turns it red. Evidence that the mutation actually does this is quoted in
/// this file's module doc (D-122: evidence lives in the branch), never only asserted.
///
/// Not `#[ignore]`d: unlike the crash it supervises, this test itself never crashes — the deliberate
/// fault is fully contained in the child process — so it is safe to run in every normal seal, on any
/// host, radio or none.
#[test]
fn crash_frames_banner_is_the_oracle_for_deliberate_sigsegv() {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;

    let exe = std::env::current_exe().expect("this test binary's own path");
    let output = Command::new(&exe)
        .args([
            "--ignored",
            "--exact",
            "deliberate_sigsegv_is_captured_with_frames_and_still_fails",
        ])
        .env("PHOSPHENE_CI_C_DELIBERATE_CRASH", "1")
        .output()
        .expect("spawn this binary as a child to run the deliberate crash in isolation");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.signal(),
        Some(11),
        "expected the child to die of SIGSEGV (11); status was {:?}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stderr.contains("===PHOSPHENE-CRASH-FRAMES==="),
        "the child crashed as expected but printed no PHOSPHENE-CRASH-FRAMES banner — the \
         in-process handler did not run. This is exactly the negative control for \
         crash_frames::install(): a crash alone is not proof of capture.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("===PHOSPHENE-CRASH-FRAMES-END==="),
        "the banner opened but never closed — frame printing was interrupted before it \
         finished.\nstderr:\n{stderr}"
    );
}
