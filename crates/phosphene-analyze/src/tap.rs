// SPDX-License-Identifier: MIT

//! The two read-only taps into v1 (nextgen spec §2.2) and their supporting
//! types — re-exported from their canonical, frozen home in
//! [`phosphene_core::tap`] (D-017).
//!
//! This module once defined the tap contract locally while `phosphene-core`
//! was held by in-flight v1 lanes; per the D-017 swap plan it is now a pure
//! re-export, so every `crate::tap::…` path in this crate keeps working
//! against the single canonical definition. See `phosphene_core::tap` for the
//! contract's documentation.
//!
//! Both taps are lock-free, read-only, single-producer/single-consumer views
//! over data v1's compute path already holds: `SpectrumTap` exposes the most
//! recent dB power frames the display computes (feeds detection and every
//! FR-AM measurement), and `AnalysisTap` exposes the raw pre-FFT IQ ring
//! (feeds classification only, phase A1+; ring depth per NFR-A2).

pub use phosphene_core::tap::{
    AnalysisTap, BandMeta, CaptureMeta, FrameView, IqBuf, SpectrumTap, TapCaps, TapError, TimeSpan,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_meta_maps_bins_to_offsets_dc_centered() {
        let meta = BandMeta {
            center_freq_hz: Some(100_000_000.0),
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        };
        assert_eq!(meta.bin_offset_hz(512.0), 0.0); // DC at bins/2
        assert_eq!(meta.bin_offset_hz(812.0), 300_000.0);
        assert_eq!(meta.bin_offset_hz(0.0), -512_000.0);
        assert_eq!(meta.bin_absolute_hz(812.0), Some(100_300_000.0));
    }

    #[test]
    fn frame_view_holds_consecutive_frames_oldest_first() {
        let mut view = FrameView::new(4, 3);
        assert!(view.is_empty());
        view.push(&[0.0; 4], 10);
        view.push(&[1.0; 4], 11);
        assert_eq!(view.len(), 2);
        assert_eq!(view.frame(0), &[0.0; 4]);
        assert_eq!(view.frame(1), &[1.0; 4]);
        assert_eq!(view.seq(0), 10);
        assert_eq!(view.seq(1), 11);
        view.clear();
        assert!(view.is_empty());
        view.push(&[2.0; 4], 42);
        assert_eq!(view.seq(0), 42);
    }

    #[test]
    #[should_panic(expected = "consecutive")]
    fn frame_view_rejects_sequence_gaps() {
        let mut view = FrameView::new(2, 2);
        view.push(&[0.0; 2], 5);
        view.push(&[0.0; 2], 7);
    }

    #[test]
    fn tap_error_messages_carry_the_numbers() {
        let msg = TapError::InsufficientHistory {
            requested_s: 1.0,
            available_s: 0.25,
        }
        .to_string();
        assert!(msg.contains('1'), "unhelpful: {msg}");
        assert!(msg.contains("0.25"), "unhelpful: {msg}");
    }
}
