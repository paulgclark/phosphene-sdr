<!-- SPDX-License-Identifier: MIT -->

# Vendored fonts

| File | Source | Version | License |
|---|---|---|---|
| `JetBrainsMono-Regular.ttf` | https://github.com/JetBrains/JetBrainsMono | v2.304 | OFL-1.1 (`OFL.txt`, verbatim from the same tag) |

JetBrains Mono is the chrome face for the D-010 visual identity. It is
embedded via `include_bytes!` in `src/theme.rs`, replacing egui's bundled
default fonts, which include a face under Ubuntu-font-1.0 — a license outside
the LC-2 permitted set. OFL fonts are explicitly permitted by LC-2.
