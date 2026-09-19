# phosphene

A single-binary, real-time spectrum analyzer for Linux and macOS: the phosphor-persistence
"RTSA" display — histogram-graded spectrum, waterfall, live and max-hold traces — fed from an
SDR, a raw IQ file, or a pipe.

No GNU Radio, no OpenCL, no Qt. It is Rust on `wgpu`, so it runs on Vulkan, Metal and native
Wayland without an interop layer, and it is MIT licensed.

Measured smooth at **50 MS/s** on an Intel i9 desktop, on an Ubuntu field laptop, and on a
MacBook Pro — the last of those live from a USRP. The design target (NFR-P1) is 20 MS/s.

---

## Install

Everything below installs **per user, into `~/.cargo/bin`** — the Rust equivalent of
`make install` is `cargo install`, and it deliberately does not use `sudo`. The only commands
here that need root are the optional system packages that give you SDR support.

### Download a prebuilt binary

Each tagged release publishes two archives — `phosphene-<tag>-linux-x86_64.tar.gz` and
`phosphene-<tag>-macos-arm64.tar.gz` (`<tag>` is the release tag, e.g. `v0.1.0`) — each with a
same-named `.sha256` file beside it. Unlike a from-source build, which follows whatever
`scripts/install.sh` detects, **the release binary always includes SoapySDR support**: it still
needs libSoapySDR installed on the machine to launch, radio attached or not. Install the SDR
package for your platform from the sections below before running a downloaded binary.

```bash
# verify before running an unfamiliar binary
sha256sum -c phosphene-<tag>-linux-x86_64.tar.gz.sha256      # Linux
shasum -a 256 -c phosphene-<tag>-macos-arm64.tar.gz.sha256   # macOS

tar xzf phosphene-<tag>-linux-x86_64.tar.gz    # or the macos-arm64 archive
./phosphene
```

### Device support at a glance

Six device families exist as SoapySDR modules. "Visible" means `SoapySDRUtil --info` lists the
module and its driver factory, the same bar used everywhere else on this page; it is never a
claim that a specific radio enumerates or streams.

| Family | Ubuntu 24.04 | macOS | Hardware-tested |
|---|---|---|---|
| UHD / USRP | `soapysdr-module-uhd` (apt) | build SoapyUHD from source (below) | **yes** — 50 MS/s live, three machines (see top of page) |
| RTL-SDR | `soapysdr-module-rtlsdr` (apt) | `soapyrtlsdr` (brew) | not yet — a unit is still being sourced |
| HackRF | `soapysdr-module-hackrf` (apt) | `soapyhackrf` (brew) | **yes** — 751 MHz LTE carrier, Linux and macOS |
| LimeSDR | `soapysdr-module-lms7` (apt) — installed and visible | `limesuite` (brew) — formula confirmed only | **yes** — 751 MHz LTE carrier, Linux and macOS |
| PlutoSDR | no apt module — build from source (below) | no clean formula — build from source (below) | **yes** — 751 MHz LTE carrier, Linux and macOS (macOS needs one extra step — see below) |
| BladeRF | `soapysdr-module-bladerf` (apt) — installed and visible | tap formula is stale — build from source (below) | **yes** — 751 MHz LTE carrier, Linux and macOS (macOS needs one extra step — see below) |

HackRF, LimeSDR, PlutoSDR and BladeRF have all since been hardware-verified: each tuned to a live
751 MHz LTE carrier and correctly displayed by phosphene, with `SoapySDRUtil --find` confirming
each device enumerated unguarded first. HackRF and LimeSDR needed nothing beyond the recipes
below, on either OS. BladeRF and PlutoSDR each need one additional step on macOS not covered by
the recipes as written — see their macOS sections below. The Ubuntu build for PlutoSDR is
upstream's own recipe, hardware-verified the same way (see below).

### Ubuntu 24.04

```bash
# 1. Rust. The distro's rustc is too old — phosphene needs 1.95 or newer.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 2. Optional — SDR support. libsoapysdr-dev is what phosphene builds against;
#    the *module* packages are what let SoapySDR actually see a radio. LimeSDR's
#    module is soapysdr-module-lms7 (named for its LMS7002 chip, not "limesdr").
#    PlutoSDR has no module package here — see below.
sudo apt install libsoapysdr-dev soapysdr-tools \
                 soapysdr-module-uhd \
                 soapysdr-module-lms7 \
                 soapysdr-module-bladerf     # -rtlsdr, -hackrf also exist

# 3. Build and install.
git clone https://github.com/paulgclark/phosphene.git
cd phosphene
bash scripts/install.sh
```

Skip step 2 and you get a working phosphene without SDR support — file replay, `stdin` and the
demo signal generator. The installer says which build it made, in both cases; it never falls
back silently.

> **`libsoapysdr-dev` on its own is not enough for a USRP.** The library is the API; the
> `soapysdr-module-*` package is the driver that makes a device enumerate. A machine here had
> the library, no module, and saw nothing.

> **LimeSDR's module package is `soapysdr-module-lms7`, not `soapysdr-module-limesdr`.** It is
> named for LimeSDR's LMS7002 transceiver chip. Installed and confirmed on this project's own
> Ubuntu build machine: `SoapySDRUtil --info` lists `libLMS7Support.so` and the `lime` factory
> (visibility only when this was written; since hardware-verified — see the table above).
> `soapysdr-module-bladerf` was confirmed the same way, listing `libbladeRFSupport.so` and the
> `bladerf` factory, also since hardware-verified.

#### PlutoSDR on Ubuntu: build SoapyPlutoSDR from source

Ubuntu 24.04 has no `soapysdr-module-plutosdr`, or any Pluto-named SoapySDR module — checked
directly against the apt package database, not assumed. It does ship the AD9361/IIO runtime
stack PlutoSDR needs (`libiio0`, `libad9361-0`), so the module itself is the only gap:

```bash
sudo apt install libiio-dev libad9361-dev cmake build-essential
git clone https://github.com/pothosware/SoapyPlutoSDR.git
cd SoapyPlutoSDR && mkdir build && cd build
cmake ..
make -j"$(nproc)"
sudo make install
```

This is upstream's own recipe (`pothosware/SoapyPlutoSDR`'s README). It wasn't build-verified
when this section was first written — the machine used to write it had `libiio`/`libad9361`'s
runtime libraries already installed but not their `-dev` headers, and installing them needed
`sudo` access that pass didn't have — but it has since been used to bring up a working PlutoSDR
on Linux, hardware-tested per the table above.

### macOS

```bash
# 1. Rust.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 2. Optional — SDR support. `soapysdr` is the library; the `soapy*` formulae
#    beside it are the device modules, and you need the one for your radio.
#    LimeSDR is the one exception: its module ships inside `limesuite`, not a `soapy*` name.
brew install soapysdr
brew install soapyrtlsdr soapyhackrf limesuite   # pick the ones you actually have

# 3. Build and install.
git clone https://github.com/paulgclark/phosphene.git
cd phosphene
bash scripts/install.sh
```

RTL-SDR and HackRF are the easy cases: `soapyrtlsdr` and `soapyhackrf` are both in
homebrew-core and pull in `librtlsdr` / `hackrf` themselves. LimeSDR is nearly as easy:
`limesuite` is also homebrew-core, not a third-party tap, and its own formula test already checks
that `SoapySDRUtil` sees driver `lime` once it has built.

> **LimeSDR, PlutoSDR and BladeRF have since been hardware-verified on macOS** (see the table
> above). None of that material was run against real macOS hardware when this section was first
> written — package and formula names below were only checked against `formulae.brew.sh` and the
> cited tap's own file listing. LimeSDR needed nothing beyond what's below; BladeRF and PlutoSDR
> each needed one extra step, noted in their own sections.

#### USRPs on macOS: build SoapyUHD from master

There is no working `soapyuhd` in Homebrew, but the bridge module that lets SoapySDR see a USRP
builds cleanly from upstream master in about a minute. Verified on two Macs — a USRP runs at
50 MS/s on a MacBook Pro this way:

```bash
brew install uhd soapysdr
git clone --depth 1 https://github.com/pothosware/SoapyUHD.git
cd SoapyUHD && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$(brew --prefix)"
make -j8 && make install
```

> **Use `$(brew --prefix)`, not a literal path.** Homebrew lives in `/opt/homebrew` on Apple
> Silicon and `/usr/local` on Intel. Install the module under the wrong one and it builds, it
> installs, and SoapySDR never looks there — `SoapySDRUtil --find` just reports nothing, with no
> error to tell you why.

`SoapySDRUtil --info` should now list `libuhdSupport.so`, and `SoapySDRUtil --find` should see
the radio. Note that the version it prints reads `0.4.1-<commit>` even when built from master —
upstream has never bumped the number, so **the commit hash is the only thing that tells you which
source you built**; `0.4.1` on its own does not mean you got the broken tap formula.

> **Do not `brew install soapyuhd`.** The formula in the `pothosware/pothos` tap pins SoapyUHD
> 0.4.1 (2019) and does not build: its `CMakeLists.txt` asks for a CMake compatibility level
> modern CMake removed, it forces `CXX_STANDARD 11` against UHD 4.10 headers that need C++17,
> and it omits a `boost::lexical_cast` include — upstream
> [pothosware/homebrew-pothos#55](https://github.com/pothosware/homebrew-pothos/issues/55).
> Current Homebrew also refuses untrusted taps, so you meet two failures before the real one.
> Upstream **master** has fixed all three defects, which is why the clone above is unpinned:
> that is the whole trick.

If the radio can live on a Linux box instead, `soapyremote` is the lower-effort path — it already
ships inside Homebrew's `soapysdr` (`soapysdr-module-remote` on the Linux side) and needs none of
the above: stream from there, display here.

Either way, the installer tells you what SoapySDR can actually see once it has built — it reports
radios it has enumerated, not radios a library implies.

#### BladeRF on macOS: build SoapyBladeRF from master, not the tap formula

`libbladerf` itself is homebrew-core and installs cleanly:

```bash
brew install libbladerf soapysdr cmake
```

The SoapySDR module doesn't have a homebrew-core formula. It exists only in the
`pothosware/pothos` tap — the same untrusted tap the USRP section above already has to work
around for SoapyUHD — pinned to release `soapy-bladerf-0.4.1`. That release fails to build
against libbladerf 2.x (a `bladerf_frequency` type mismatch), fixed on `SoapyBladeRF`'s master
branch in [pothosware/SoapyBladeRF#19](https://github.com/pothosware/SoapyBladeRF/pull/19) but
never repinned in the tap formula. Building master directly sidesteps it, the same trick as the
USRP recipe above:

```bash
git clone https://github.com/pothosware/SoapyBladeRF.git
cd SoapyBladeRF && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$(brew --prefix)"
make -j8 && make install
```

The failure and its fix above come from upstream's own issue tracker, not from a local build; the
recipe itself is now hardware-verified on macOS, with one thing to know first:

> **BladeRF enumerates before it can stream.** A bladerf2 board answers `SoapySDRUtil --find`
> as soon as libusb can see it, but phosphene's own open fails until the board's FPGA bitstream
> is loaded — it reports `Board state insufficient... requires Initialized`. Load a matching
> `.rbf` first: `bladeRF-cli -d "*:serial=<yours>" -l /path/to/board.rbf`. With that done, this
> recipe brings up a working BladeRF on macOS, hardware-tested per the table above.

#### PlutoSDR on macOS: build from source, three layers deep

There is no homebrew-core formula for PlutoSDR's SoapySDR module, and the only tap that has one
(`pothosware/pothos`, `soapyplutosdr`) is pinned to release `0.2.1` against a `libiio` pinned to
`v0.15` — both untrusted, both old. Building from source means building the AD9361/IIO stack
too, since neither `libiio` nor `libad9361` has a homebrew-core formula either:

```bash
brew install cmake soapysdr libusb

git clone --branch v0.25 https://github.com/analogdevicesinc/libiio.git
cd libiio && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$(brew --prefix)" -DOSX_FRAMEWORK=OFF
make -j8 && make install
cd ../..

git clone --branch libad9361-iio-v0 https://github.com/analogdevicesinc/libad9361-iio.git
cd libad9361-iio && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$(brew --prefix)" -DCMAKE_PREFIX_PATH="$(brew --prefix)" \
         -DOSX_FRAMEWORK=OFF
make -j8 && make install
cd ../..

git clone https://github.com/pothosware/SoapyPlutoSDR.git
cd SoapyPlutoSDR && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$(brew --prefix)" -DCMAKE_PREFIX_PATH="$(brew --prefix)"
make -j8 && make install
```

> **`libiio`'s CMake defaults to `WITH_USB_BACKEND=ON`, and hard-fails configure if `libusb-1.0`
> isn't found** (`message(SEND_ERROR "Unable to find libusb-1.0 dependency.")` in its
> `CMakeLists.txt`) — hence `libusb` above, on top of `cmake` and `soapysdr`.

> **`libiio` and `libad9361-iio` both default to `OSX_FRAMEWORK=ON` on macOS** — the same
> option block, carried between the two projects. With it on, `make install` ignores
> `CMAKE_INSTALL_PREFIX` altogether and shells out to `/usr/sbin/installer -pkg … -target /`,
> which needs root and leaves a `/Library/Frameworks` bundle instead of the plain dylib,
> headers and `.pc` file under `$(brew --prefix)` that the next link in the chain looks for.
> `-DOSX_FRAMEWORK=OFF` on **both** builds is what makes the chain able to find them at all.
> One flag per project is enough: each `CMakeLists.txt` declares `OSX_PACKAGE` — and the
> `SKIP_INSTALL_ALL` that suppresses the normal `install()` rules — only *inside* the
> `if(Darwin AND OSX_FRAMEWORK)` block, so turning the framework off skips both with it.
> `libad9361-iio`'s own `CMakeLists.txt` locates `libiio` with a bare, hint-free
> `find_library`/`find_path` call. `SoapyPlutoSDR`'s own `FindLibIIO.cmake`/`FindLibAD9361.cmake`
> do check pkg-config and Homebrew paths — but the paths they hardcode are the
> `Cellar/<formula>/<version>` layout `brew install` itself produces, not the plain
> `$(brew --prefix)/include` + `$(brew --prefix)/lib` layout this recipe's manual `make install`
> leaves behind (neither `libiio` nor `libad9361` is installed via `brew` here — see above).
> `-DCMAKE_PREFIX_PATH="$(brew --prefix)"` on both `cmake` invocations covers that gap regardless
> of whether pkg-config or a bare default search would otherwise have found it.

> **Use the `libad9361-iio-v0` branch, not `main`.** `libad9361-iio`'s `main` branch targets
> `libiio`'s newer v1.0 API; `libad9361-iio-v0` targets the v0.x API that both this recipe's
> `libiio` clone and Ubuntu's packaged `libiio` are on.

This chain is now hardware-verified on macOS, with one fix needed beyond the steps above:

> **`libad9361-iio`'s source expects a header path `libiio` doesn't provide on this branch.**
> Its code has `#include <iio/iio.h>`, but the v0.x `libiio` this recipe targets installs its
> header flat, as `iio.h` — a real upstream `libad9361-iio` defect (MacPorts' own port for it
> carries a patch removing exactly those includes). Fix it with a symlink before building
> `libad9361-iio`:
> ```bash
> mkdir -p "$(brew --prefix)/include/iio"
> ln -s ../iio.h "$(brew --prefix)/include/iio/iio.h"
> ```
> With that in place, this chain brings up a working PlutoSDR on macOS, hardware-tested per the
> table above.

### What the installer does

`scripts/install.sh` checks your toolchain, probes for SoapySDR with `pkg-config`, runs
`cargo install --path crates/phosphene-app` with or without `--features phosphene-sources/soapy`
accordingly, then tells you where the binary landed and whether your shell can actually see it.
If SDR support went in, it also reports **how many radios are visible right now** — a library
being installed is not a radio being attached.

It is safe to re-run, and it never calls `sudo`. Two knobs:

```bash
PHOSPHENE_SOAPY=off bash scripts/install.sh   # force a plain build, no SDR support
PHOSPHENE_SOAPY=on  bash scripts/install.sh   # require SDR support; fail if SoapySDR is missing
```

If you would rather not run a script, the whole of it is one command, from the clone:

```bash
cargo install --locked --path crates/phosphene-app --features phosphene-sources/soapy
```

Drop `--features` for a build without SDR support. What the script adds is the checking: your
rustc version, whether SoapySDR is really there, where the binary went, whether your shell can
see it, and what is actually plugged in.

> **If your shell then says `phosphene: command not found`,** `~/.cargo/bin` is not on your
> `PATH` — which is what happens when rustup was installed with `--no-modify-path`. Fix it once:
>
> ```bash
> echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && export PATH="$HOME/.cargo/bin:$PATH"
> ```

---

## Run

The binary is on your `PATH` after installing, so none of these needs a path to an executable:

```bash
phosphene                                   # demo signal generator
phosphene --source soapy:driver=uhd --rate 10e6 --center 751e6 --gain 50
phosphene --source file:cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16
phosphene --headless --source file:capture.cf32 --out shot.png
```

`phosphene --help` lists every flag.

### Captures that name themselves

Look at the third command again: no `--rate`, no `--center`, no `--format`, and the axes are
still labelled correctly. phosphene reads the capture's own metadata, in this order:

**explicit flag** → **SigMF sidecar** (`.sigmf-meta`) → **file name** → nothing.

The file-name convention is `cap_<freq><unit>_<rate><unit>sps[_anything].<ext>`, where `p` is
the decimal point that a file name cannot carry:

```
cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16   ->  1.985 GHz, 30.72 Msps, interleaved int16
cap_632MHz_30p72Msps.sc16                          ->  632 MHz, 30.72 Msps
```

Everything after the rate is ignored, so your own tags and timestamps can ride along. Units are
read, never assumed. A name that does not match is not an error — it just falls through to the
next thing in the chain, and an inferred value is always shown as inferred rather than dressed
up as a measurement.

### Keys

Press **`?`** in the window for the key map. It is generated from the bindings themselves, so it
cannot drift; a copy in this file could, which is why there isn't one.

### Over SSH

phosphene draws a real window, so a bare `ssh` session has nothing to draw on. Either run it at
the machine, or point it at that machine's display:

```bash
DISPLAY=:0 phosphene
```

For a scripted or truly headless box, `--headless` renders to a PNG instead and needs no
display server at all.

---

## Licensing

phosphene is **MIT**, and it links no GPL code.

UHD, SoapyUHD and librtlsdr *are* GPL, and a reader who followed the macOS recipe above will
rightly ask. There is no conflict, because none of them is part of this program: they are
**runtime modules that you install**, which libSoapySDR (BSL-1.0) loads on your machine at run
time. phosphene links libSoapySDR and nothing else — it never links, vendors or ships UHD. That
is the identical posture on both platforms: the distro's `soapysdr-module-uhd` on Linux is as GPL
as the SoapyUHD you build on macOS, and neither becomes part of this binary. The arrangement is
deliberate, and it is enforced in CI by `cargo deny`.
