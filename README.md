# PerceptoMap

[![Downloads](https://img.shields.io/github/downloads/hqrrr/PerceptoMap/total?label=Downloads)](https://github.com/hqrrr/PerceptoMap/releases)&nbsp;
[![Release](https://img.shields.io/github/v/release/hqrrr/PerceptoMap?label=Release&color=4c8eda)](https://github.com/hqrrr/PerceptoMap/releases)
&nbsp;
[![Platforms](https://img.shields.io/badge/Platforms-Windows%20|%20macOS%20|%20Linux-4c8eda)](https://github.com/hqrrr/PerceptoMap/releases)&nbsp;
[![License](https://img.shields.io/github/license/hqrrr/PerceptoMap?color=888)](https://github.com/hqrrr/PerceptoMap/blob/main/LICENSE)&nbsp;

### Visualizing how we hear — from spectrograms to perception

PerceptoMap is an open-source audio plugin (VST3) that visualizes psychoacoustic features of audio signals in real time. Built with [JUCE](https://juce.com/), it's designed to help you **see** how we **perceive** sound — not just how it looks on a frequency plot.

Unlike typical spectrum or spectrogram analyzers, it supports perceptual visualizations such as **Mel spectrograms**, **Mel-frequency cepstral coefficients (MFCCs)**, **Chromagram**, and soon also **Tempogram** etc., offering insight into how humans perceive sound.

> 🎧 If you're the kind of creator who trusts your ears above all — you might not need this.  
> But if you're curious about how your audio *measures up* to what humans actually hear… welcome aboard.

## Quick Navigation

- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Roadmap](#roadmap)
- [Why develop this plugin?](#why-develop-this-plugin)
- [How to install?](#how-to-install)
  - [Windows](#windows)
  - [macOS (Intel / Apple Silicon)](#macos-intel--apple-silicon)
  - [Linux](#linux)
- [Build Instructions for Developers](#build-instructions-for-developers)
- [License & Cost](#license--cost)

## Key Features

- Real-time **Mel Spectrogram** display with perceptual frequency scaling
- Real-time **Mel-frequency cepstral coefficients (MFCCs)** representing timbral texture and spectral envelope
- Real-time **Spectral Centroid** tracking to visualize spectral brightness (center of mass of STFT spectrum)
- Real-time **Chromagram** showing the energy distribution across the 12 pitch classes (C to B), regardless of octave. <span style="color: gray;">[added in v0.5]</span>
- **Time–Frequency Reassignment** mode for enhanced STFT resolution, based on the paper [[hal-00414583: Time-Frequency reassignment: from principles to algorithms](https://hal.science/hal-00414583/document)]. This mode sharpens the localization of spectral peaks by reassigning energy to more accurate time–frequency coordinates, making harmonic structures and transient details clearer compared to the standard STFT. <span style="color: gray;">[added in v0.6]</span>
- Visual analysis of **Tempogram** and other psychoacoustic features *(planned)* 
- Configurable color maps
- Adjustable brightness gain and enhanced colormap modes to improve visibility of fine details in the spectrogram <span style="color: gray;">[added in v0.4]</span>
- Optional dB scaling, log or linear frequency axis for classic linear STFT spectrogram
- Freeze frame mode and interactive mouse hover to inspect frequency and loudness at any point
- Configurable FFT size for balancing time resolution and frequency resolution.<span style="color: gray;">[added in v0.6]</span>
- Independent scroll speed control, allowing smooth visualization at different FFT sizes and overlap settings without distorting the spectral data. <span style="color: gray;">[added in v0.6]</span>


## Screenshots

<!-- <p align="center">
  <img src="_pics/gui_ableton_live_12.png" width="100%" alt="GUI" />
  <sub>Plugin User Interface in Ableton Live 12</sub>
</p>

--- -->

<table style="table-layout: fixed; width: 100%;">
  <tr>
    <td align="center" valign="top">
      <img src="_pics/gui_resize_window_2.png" width="100%" alt="Resizable GUI Small" />
      <br/>
      <sub><strong>Resizable GUI</strong><br/>Window can shrink to fit minimal layout</sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_resize_window_1.png" width="100%" alt="Resizable GUI Large" />
      <br/>
      <sub><strong>Resizable GUI</strong><br/>Window expands for detailed viewing</sub>
    </td>
  </tr>
</table>

---

<table>
  <tr>
    <td align="center" valign="top">
      <img src="_pics/gui_color_classic.png" width="100%" alt="Classic Colormap" />
      <br/>
      <sub><strong>Classic</strong><br/>Default high-contrast mapping</sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_color_gray.png" width="100%" alt="Grayscale Colormap" />
      <br/>
      <sub><strong>Grayscale</strong><br/>Neutral luminance-based display</sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_color_magma.png" width="100%" alt="Grayscale Colormap" />
      <br/>
      <sub><strong>Magma</strong><br/>Perceptually uniform, dark background</sub>
    </td>
  </tr>
</table>

---

<table style="table-layout: fixed; width: 100%;">
  <tr>
    <td align="center" valign="top">
      <img src="_pics/gui_mouse_info.png" width="100%" alt="Hover Frequency Readout" />
      <sub><strong>Hover Readout</strong><br/>Displays precise frequency, dB level, and corresponding MIDI note (C4 = Middle C) <span style="color: gray;">[added in v0.3]</span> under the mouse pointer.</sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_color_scheme_floor_value.png" width="100%" alt="Hover Frequency Readout" />
      <sub><strong>Adjustable dB Floor Slider</strong><br/>Controls the minimum dB threshold for color brightness, helping visualize low-level signals. <span style="color: gray;">[added in v0.3]</span></sub>
    </td>
  </tr>
</table>


---

<table style="table-layout: fixed; width: 100%;">
  <tr>
    <td align="center" valign="top">
      <img src="_pics/gui_enhanced_resolution.png" width="100%" alt="Hover Frequency Readout" />
      <sub><strong>Adjustable Brightness Gain</strong><br/>The Norm Factor slider allows manual control over spectrogram brightness, helping to adapt the display to signals with different loudness levels. <span style="color: gray;">[added in v0.4]</span></sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_enhanced_color.png" width="100%" alt="Hover Frequency Readout" />
      <sub><strong>Fine Detail with Enhanced Colormap Modes</strong><br/>With non-linear color legend to enhance contrast, making subtle details more visible: <i>Magma+ & Grayscale+</i>. <span style="color: gray;">[added in v0.4]</span></sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_enhanced_STFT_time_frequency_reassignment.png" width="100%" alt="Time–Frequency Reassignment Mode" />
      <sub><strong>Time–Frequency Reassignment Mode</strong><br/>
      Sharpens the time–frequency localization of spectral peaks by reassigning energy to more accurate coordinates. Harmonic structures and transients become more clearly defined, compared to standard STFT. <span style="color: gray;">[added in v0.6]</span></sub>
    </td>
  </tr>
</table>

---

<table style="table-layout: fixed; width: 100%;">
  <tr>
    <td align="center" valign="top">
      <img src="_pics/gui_freq_axis_linear_scale.png" width="100%" alt="Linear STFT spectrogram with linear frequency axis" />
      <br/>
      <sub>
        <strong>Linear STFT spectrogram with linear frequency axis:</strong>
        <i>
          Displays physical frequency content directly.
        </i>
      </sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_color_magma.png" width="100%" alt="Linear STFT spectrogram with log frequency axis" />
      <br/>
      <sub>
        <strong>Linear STFT spectrogram with log frequency axis:</strong>
        <i>
          Approximates human pitch perception. Emphasizes low-frequency resolution and compresses high-frequency bands.
        </i>
      </sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_melspectrogram.png" width="100%" alt="Mel-scaled STFT spectrogram" />
      <br/>
      <sub>
        <strong>Mel-scaled STFT spectrogram:</strong>
        <i>
          Reflects the nonlinear frequency resolution of human hearing. Provides a more perceptually accurate representation than simple log-scaling.
        </i>
      </sub>
    </td>
  </tr>
</table>

<table style="table-layout: fixed; width: 100%;">
  <tr>
    <td align="center" valign="top">
      <img src="_pics/gui_mfcc.png" width="100%" alt="Mel-frequency cepstral coefficient" />
      <br/>
      <sub>
        <strong>Mel-frequency cepstral coefficients (MFCCs):</strong>
        <i>
          Captures the spectral envelope using Discrete Cosine Transform (DCT) over Mel energies. Represents timbral texture and vocal tract shape. <span style="color: gray;">[added in v0.2]</span>
        </i>
      </sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_spectral_centroid.png" width="100%" alt="blank" />
      <br/>
      <sub>
        <strong>Spectral Centroid:</strong>
        <i>
          Indicates the "center of mass" of the spectrum. Tracks brightness and perceptual sharpness by showing where the dominant frequencies are concentrated over time. <span style="color: gray;">[added in v0.3]</span>
        </i>
      </sub>
    </td>
    <td align="center" valign="top">
      <img src="_pics/gui_chroma.png" width="100%" alt="blank" />
      <br/>
      <sub>
        <strong>Chromagram:</strong>
        <i>
          Projects spectral energy onto the 12 pitch classes (C, C#, D, …), regardless of octave. Useful for analyzing harmony, key, and chord structures. <span style="color: gray;">[added in v0.5]</span>
        </i>
      </sub>
    </td>
  </tr>
</table>

[Back to top ↥](#perceptomap)

## Roadmap

| Feature | Status | Description | Implementation Details |
|-|-|-|-|
| Linear STFT Spectrogram | ✅ Done (v0.1) | Classic time–frequency analysis | FFT size = 2048, Hann window, with log/linear frequency axis display|
| Mel-Spectrogram | ✅ Done (v0.1) | Nonlinear frequency scaling approximating human pitch perception | 128 bands, Slaney-style: `2595 * log10(1 + f / 700)`|
| MFCC | ✅ Done (v0.2) | Mel frequency cepstral coefficients, compact representation of timbre based on perceptual log-mel spectrum | DCT-II on log-mel spectrum, 20 coefficients, no liftering, values clipped to [−100, 100] and normalized to [0, 1] for display |
| Spectral Centroid (STFT-based) | ✅ Done (v0.3) | Tracks the "center of mass" of the spectrum; correlates with brightness and sharpness | Computed from linear STFT magnitude spectrum with smoothing (Exponential Moving Average), overlaid as a curve on the STFT spectrogram |
| Adjustable brightness gain and enhanced colormap modes | ✅ Done (v0.4) | To improve visibility of fine details in the spectrogram | Brightness remapped using non-linear scaling; norm factor slider controls global dB scaling, color maps applied after brightness normalization |
| Chroma | ✅ Done (v0.5) | Pitch class mapping, projection of spectral energy onto 12 pitch classes (C, C#, D…) | Triangular chroma filter bank built from STFT bins, 12 overlapping filters per octave; energy mapped to pitch classes regardless of octave; supports smooth pitch transitions and partial overlaps |
| Enhanced STFT with Time–Frequency Reassignment (Linear+) | ✅ Done (v0.6) | Sharper time–frequency localization by reassigning each STFT bin’s energy to more accurate time/frequency coordinates | Based on [[hal-00414583: Time-Frequency reassignment: from principles to algorithms](https://hal.science/hal-00414583/document)], implemented with Gaussian-window STFT and its time & frequency derivatives. Instantaneous frequency and group delay estimates are used to re-map spectral energy, improving localization of transients and harmonics compared to standard STFT. Supports same FFT size and log/linear axis options as Linear mode |
| Tempogram / Rhythm Map | ⏳ Planned | Visualizes perceived tempo and rhythmic periodicities over time | - |
| Spectral Flatness / Contrast | ⏳ Planned | Measures of timbral characteristics | - |

[Back to top ↥](#perceptomap)

## Why develop this plugin?

In the fields of psychoacoustics, and machine learning, perceptually inspired representations such as **Mel spectrograms** and **MFCCs** are widely used — for example, in music genre classification, emotion recognition, or detecting AI-generated audio.  

As a frequent user of tools like [librosa](https://librosa.org/) in Python, while learning about DAWs, I was surprised to find that most DAWs seem to lack real-time, perceptually grounded visualization tools.  

So I decided to build one — a lightweight, JUCE-based plugin that brings these powerful analysis tools directly into the DAW environment, where musicians, sound designers, and researchers can explore them interactively.

[Back to top ↥](#perceptomap)

## How to install?

You can download the latest version of **PerceptoMap** from the [Releases](https://github.com/hqrrr/PerceptoMap/releases) page.

### Available Format

- VST3 (`.vst3`)

### Plugin installation paths

#### Windows
1. Download the plugin `.zip` file from the [Releases](https://github.com/hqrrr/PerceptoMap/releases)
2. Unzip and copy `.vst3` **plugin folder** to the default system VST3 directory: `C:\Program Files\Common Files\VST3\`

> Note: If you use a custom VST3 plugin path, copy it there instead.

3. Launch your DAW and run a plugin rescan if necessary
4. Then, you should be able to find the plugin under `hqrrr - PerceptoMap`

#### macOS (Intel / Apple Silicon)

1. Download the plugin `.zip` file from the [Releases](https://github.com/hqrrr/PerceptoMap/releases):
   - `macOS_x64` for Intel
   - `macOS_arm` for Apple Silicon
2. Unzip and copy the `.vst3` **plugin folder** to the default system VST3 directory:  
   `/Library/Audio/Plug-ins/VST3`

> Note: If you use a custom VST3 plugin path, copy it there instead.

3. Launch your DAW and rescan plugins if needed
4. Then, you should be able to find the plugin under `hqrrr - PerceptoMap`

#### Linux
1. Download the plugin `.zip` file from the [Releases](https://github.com/hqrrr/PerceptoMap/releases)
2. Unzip and copy `.vst3` **plugin folder** to your VST3 directory
3. Launch your DAW and rescan plugins if needed
4. Then, you should be able to find the plugin under `hqrrr - PerceptoMap`

[Back to top ↥](#perceptomap)

## Build Instructions for Developers

### Prerequisites

- JUCE 8.x (automatically fetched via CMake)
- C++17 compatible compiler, e.g. Visual Studio 2022 (Windows)
- CMake 3.22+

### Build with CMake

Based on [JUCE CMake Plugin Template](https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template).

#### On Windows (Visual Studio 2022)

1. Open the project root in Visual Studio (choose "Open a local folder").
2. Visual Studio will automatically detect the CMakeLists.txt.
3. Select a CMake target configuration (e.g. x64-Release).
4. In the CMake Targets View: `PerceptoMap/PerceptoMap Project/Targets`, right-click `PerceptoMap_VST3` and click Build.
4. The plugin binary will be placed in the build output directory: `out/build/x64-Release/VST3/PerceptoMap.vst3`

#### On Windows (Terminal)

```
cd path\to\PerceptoMap
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If using Ninja

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Folder Structure

```text
PerceptoMap/
├── _pics/               -> Screenshots and images for documentation
├── Source/              -> Main plugin source code
├── CMakeLists.txt       -> Main build configuration (CMake-based)
├── CMakeSettings.json   -> (Optional) Visual Studio CMake config
├── README.md            -> Project documentation
└── LICENSE              -> AGPLv3 license file (required for JUCE open-source usage)
```

[Back to top ↥](#perceptomap)

## License & Cost

**PerceptoMap** is proudly open-source and completely free to use, modify, and redistribute under the terms of the [GNU AGPLv3 License](LICENSE).

There are no hidden fees, paid versions, or limitations — the plugin is intended to be a community-driven tool for perceptual audio analysis and creative exploration.

I do **not** ask for donations — what matters more is your **feedback**, **feature ideas**, or even better: your **involvement in development**.

Ways you can contribute:

- 🐞 Report bugs or issues you encounter
- 💡 Suggest improvements or new perceptual features
- 🔧 Submit pull requests to improve code or documentation
- 📢 Share the plugin with others who may find it useful

---

**Feel free to leave a comment** — bug reports, feature ideas, or just thoughts are always welcome.  

[Back to top ↥](#perceptomap)