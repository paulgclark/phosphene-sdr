#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# install.sh — build phosphene and install the `phosphene` binary for the current user.
#
# The Rust equivalent of `sudo make install` is `cargo install`, and it is deliberately
# NOT sudo: it installs to ~/.cargo/bin, per user, no root (D-067). This script never
# invokes sudo and never asks for a password.
#
#   bash scripts/install.sh                 # detect SoapySDR, build with SDR support if present
#   PHOSPHENE_SOAPY=off bash scripts/install.sh   # force a plain build, no SDR support
#   PHOSPHENE_SOAPY=on  bash scripts/install.sh   # require SDR support; fail if SoapySDR is missing
#
# It is idempotent: re-running it re-installs over the previous copy.
set -euo pipefail

# ── where we are ────────────────────────────────────────────────────────────────
# Resolve the repo root from this script's own location so the installer works
# from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MIN_RUSTC_MAJOR=1
MIN_RUSTC_MINOR=95
SOAPY_MODE="${PHOSPHENE_SOAPY:-auto}"

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }
die()  { printf '\nxx %s\n' "$*" >&2; exit 1; }

case "$SOAPY_MODE" in
  auto | on | off) ;;
  *) die "PHOSPHENE_SOAPY must be auto, on or off (got '$SOAPY_MODE')." ;;
esac

say "phosphene installer — building from $ROOT"

# ── 1. toolchain ────────────────────────────────────────────────────────────────
# Fail with the fix, not with a compiler error page four minutes into a build.
step "Rust toolchain"

if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
  die "no Rust toolchain on PATH. Install one with:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  then open a new shell (or 'source \$HOME/.cargo/env') and re-run this script."
fi

RUSTC_PATH="$(command -v rustc)"
RUSTC_VERSION="$(rustc --version | awk '{print $2}')"
RUSTC_MAJOR="${RUSTC_VERSION%%.*}"
RUSTC_REST="${RUSTC_VERSION#*.}"
RUSTC_MINOR="${RUSTC_REST%%.*}"

version_ok=1
if [ "$RUSTC_MAJOR" -lt "$MIN_RUSTC_MAJOR" ]; then
  version_ok=0
elif [ "$RUSTC_MAJOR" -eq "$MIN_RUSTC_MAJOR" ] && [ "$RUSTC_MINOR" -lt "$MIN_RUSTC_MINOR" ]; then
  version_ok=0
fi

if [ "$version_ok" -eq 0 ]; then
  msg="rustc $RUSTC_VERSION at $RUSTC_PATH is too old — phosphene needs \
rustc >= $MIN_RUSTC_MAJOR.$MIN_RUSTC_MINOR (its egui/epaint dependency requires it)."
  # A distro-packaged rustc shadowing a newer rustup one on PATH is a real
  # failure seen on this fleet: the toolchain is installed and still unused.
  RUSTUP_RUSTC="${CARGO_HOME:-$HOME/.cargo}/bin/rustc"
  if [ "$RUSTC_PATH" != "$RUSTUP_RUSTC" ] && [ -x "$RUSTUP_RUSTC" ]; then
    msg="$msg

  A rustup toolchain already exists at $RUSTUP_RUSTC ($("$RUSTUP_RUSTC" --version | awk '{print $2}')),
  but PATH prefers $RUSTC_PATH. Put rustup's bin directory first:

    export PATH=\"${CARGO_HOME:-\$HOME/.cargo}/bin:\$PATH\""
  else
    msg="$msg

  Ubuntu's packaged rustc is older than this; use rustup instead:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    rustup update stable"
  fi
  die "$msg"
fi

say "rustc $RUSTC_VERSION ($RUSTC_PATH) — ok, needs >= $MIN_RUSTC_MAJOR.$MIN_RUSTC_MINOR"
say "cargo $(cargo --version | awk '{print $2}')"

# ── 2. SoapySDR probe ───────────────────────────────────────────────────────────
# Both outcomes get stated. A silent fallback to a build without SDR support is
# the defect this project keeps ruling against: the user would find out only when
# `--source soapy` refused, long after install (D-067).
step "SDR support (SoapySDR)"

SOAPY_VERSION=""
if [ "$SOAPY_MODE" = "off" ]; then
  say "PHOSPHENE_SOAPY=off — skipping the SoapySDR probe by request."
elif command -v pkg-config >/dev/null 2>&1 && SOAPY_VERSION="$(pkg-config --modversion SoapySDR 2>/dev/null)"; then
  say "found SoapySDR $SOAPY_VERSION (pkg-config)."
else
  SOAPY_VERSION=""
  if ! command -v pkg-config >/dev/null 2>&1; then
    say "pkg-config is not installed, so SoapySDR cannot be detected."
  else
    say "pkg-config cannot see SoapySDR."
  fi
fi

WITH_SOAPY=0
[ -n "$SOAPY_VERSION" ] && WITH_SOAPY=1

if [ "$SOAPY_MODE" = "on" ] && [ "$WITH_SOAPY" -eq 0 ]; then
  die "PHOSPHENE_SOAPY=on was requested but SoapySDR was not found.
  Ubuntu:  sudo apt install libsoapysdr-dev soapysdr-module-uhd soapysdr-tools
  macOS:   brew install soapysdr"
fi

CARGO_ARGS=(install --path "$ROOT/crates/phosphene-app" --locked --force)
if [ "$WITH_SOAPY" -eq 1 ]; then
  CARGO_ARGS+=(--features phosphene-sources/soapy)
  say ""
  say "-> building WITH SDR support: '--source soapy:driver=uhd' and friends will work."
  say "   Live radios are reached through SoapySDR's own runtime modules; phosphene"
  say "   links nothing but libSoapySDR itself."
else
  say ""
  say "-> building WITHOUT SDR support: file, stdin and the demo signal generator work;"
  say "   '--source soapy' will refuse at startup."
  say "   To get SDR support, install SoapySDR and re-run this script:"
  say "     Ubuntu:  sudo apt install libsoapysdr-dev soapysdr-module-uhd soapysdr-tools"
  say "     macOS:   brew install soapysdr"
fi

# ── 3. build + install ──────────────────────────────────────────────────────────
step "cargo install (per user, no sudo — installs to ${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin)"

# Point cargo at the workspace target directory so a re-run reuses the build
# cache instead of recompiling the world from a fresh temp dir every time.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
say "build cache: $CARGO_TARGET_DIR"
say "this can take a few minutes on a cold cache."
cargo "${CARGO_ARGS[@]}"

# ── 4. where it went, and can the shell see it ──────────────────────────────────
# An installer that leaves the binary invisible has not finished the job: rustup
# installed with --no-modify-path has already cost time on this fleet (D-067).
step "Installed binary"

BIN_DIR="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin"
BIN="$BIN_DIR/phosphene"

[ -x "$BIN" ] || die "cargo reported success but $BIN is missing or not executable."
say "binary: $BIN"
say "version: $("$BIN" --version)"

RESOLVED="$(command -v phosphene 2>/dev/null || true)"
if [ -z "$RESOLVED" ]; then
  warn "$BIN_DIR is NOT on your PATH, so typing 'phosphene' will not find it."
  say  "Fix it for this shell and every future one:"
  say  ""
  say  "    echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc && export PATH=\"$BIN_DIR:\$PATH\""
  say  ""
  say  "(zsh users: ~/.zshrc. Until then, run it by full path: $BIN)"
elif [ "$RESOLVED" != "$BIN" ]; then
  warn "'phosphene' on PATH resolves to $RESOLVED, not the copy just installed at $BIN."
  say  "Put $BIN_DIR earlier on PATH, or remove the other copy."
else
  say "'phosphene' resolves from PATH: $RESOLVED"
fi

# ── 5. what is actually attached ────────────────────────────────────────────────
# SoapySDR being installed is not a usable radio. Report devices seen, never
# devices assumed (D-067).
if [ "$WITH_SOAPY" -eq 1 ]; then
  step "Visible SDR devices"
  if command -v SoapySDRUtil >/dev/null 2>&1; then
    FIND_OUT="$(SoapySDRUtil --find 2>/dev/null || true)"
    # Every enumerated device prints one 'driver = <name>' line.
    DRIVERS="$(printf '%s\n' "$FIND_OUT" | sed -n 's/^[[:space:]]*driver[[:space:]]*=[[:space:]]*//p' | sort)"
    TOTAL=0
    [ -n "$DRIVERS" ] && TOTAL="$(printf '%s\n' "$DRIVERS" | grep -c .)"
    # 'audio' is SoapySDR's sound-card backend and is present on almost every
    # machine; counting it as a radio would be exactly the cheerful guess we
    # are trying not to make.
    RADIOS=0
    RADIO_DRIVERS="$(printf '%s\n' "$DRIVERS" | grep -v '^audio$' || true)"
    [ -n "$RADIO_DRIVERS" ] && RADIOS="$(printf '%s\n' "$RADIO_DRIVERS" | grep -c .)"

    say "built with SDR support; $RADIOS radio(s) currently visible to SoapySDR ($TOTAL device(s) enumerated in total)."
    if [ "$RADIOS" -gt 0 ]; then
      printf '%s\n' "$RADIO_DRIVERS" | sort | uniq -c | while read -r n drv; do
        say "  $n x driver=$drv   ->  phosphene --source soapy:driver=$drv"
      done
    else
      say "  Nothing is attached right now (or the driver module for it is not installed)."
      say "  The library being present does not make a radio appear; plug one in and re-run"
      say "  'SoapySDRUtil --find' to check."
    fi
    if [ -n "$DRIVERS" ] && printf '%s\n' "$DRIVERS" | grep -q '^audio$'; then
      say "  (the 'audio' entries above are SoapySDR's sound-card backend, not radios.)"
    fi
    if ! SoapySDRUtil --info 2>/dev/null | grep -qi 'uhdSupport'; then
      say "  No UHD module is loaded, so USRPs will not enumerate. Ubuntu: 'sudo apt install soapysdr-module-uhd'."
      say "  macOS: build SoapyUHD from upstream master (see README) — the tap formula does not build."
    fi
  else
    # Do not guess. Not knowing is a reportable state.
    say "built with SDR support, but SoapySDRUtil is not installed, so the number of"
    say "visible radios cannot be reported here."
    say "  Ubuntu: sudo apt install soapysdr-tools    macOS: it ships with 'brew install soapysdr'"
  fi
fi

step "Done"
if [ -n "$RESOLVED" ] && [ "$RESOLVED" = "$BIN" ]; then
  say "Try it:  phosphene"
else
  say "Try it:  $BIN"
fi
say "Press ? in the window for the key map. See README.md for more invocations."
