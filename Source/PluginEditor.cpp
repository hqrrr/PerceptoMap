/*
  ==============================================================================

    PluginEditor.cpp
    Implements the GUI layout, including spectrogram view, controls, and legend.

    Author: hqrrr
    GitHub: https://github.com/hqrrr/PerceptoMap

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "SpectrogramComponent.h"  // Include the spectrogram visualization

//==============================================================================
SpectrogramAudioProcessorEditor::SpectrogramAudioProcessorEditor(SpectrogramAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    // Set initial editor window size
    setSize(800, 600);
    setResizable(true, true);
    setResizeLimits(200, 200, 3000, 3000);

    // Add and show the spectrogram component
    addAndMakeVisible(spectrogram);
    // Configure spectrogram settings
    spectrogram.setUseLogFrequency(true);
    // Set sample rate
    spectrogram.setSampleRate(audioProcessor.getSampleRate());

    // Fixed display of fold/expand button (always visible)
    addAndMakeVisible(toggleUiButton);
    toggleUiButton.setTooltip("Show/Hide control panels");
    toggleUiButton.setAlwaysOnTop(true);
    toggleUiButton.onClick = [this]()
    {
        controlsVisible = !controlsVisible;
        updateControlsVisibility();
        // switch arrow
        toggleUiButton.setButtonText(controlsVisible ? HideMenuText : ShowMenuText);
        resized();
        repaint();
    };
    // init visibility
    updateControlsVisibility();

    // ==== 1st row ====

    // Add and configure freeze button
    addAndMakeVisible(freezeButton);
    freezeButton.setTooltip("Freeze or resume spectrogram scrolling");
    freezeButton.onClick = [this]()
    {
        isFrozen = !isFrozen;
        freezeButton.setButtonText(isFrozen ? "Resume" : "Freeze");
        spectrogram.setFrozen(isFrozen);
    };

    // Presets
    addAndMakeVisible(presetBox);
    static constexpr int kPlaceholderId = 999;
    presetBox.setTooltip("Select global preset slot");
    presetBox.addItem("---", kPlaceholderId);
    presetBox.addItem("Preset1", 1);
    presetBox.addItem("Preset2", 2);
    presetBox.addItem("Preset3", 3);
    presetBox.setSelectedId(kPlaceholderId, juce::dontSendNotification); // Default

    presetBox.onChange = [this]()
    {
        if (isApplyingPreset) return;

        const int id = presetBox.getSelectedId();

        // id == kPlaceholderId -> place holder, no actions
        if (id == kPlaceholderId)
        {
            savePresetButton.setEnabled(false);
            return;
        }

        // only Preset 1/2/3 allowed
        if (id < 1 || id > 3)
            return;

        savePresetButton.setEnabled(true);

        const int presetIdx = id - 1; // 1->0, 2->1, 3->2
        auto p = audioProcessor.loadPreset(presetIdx);

        // When switching presets, directly apply them while synchronizing currentSettings
        applyDataToUI(p, true);
        pushCurrentToProcessor();
    };

    addAndMakeVisible(savePresetButton);
    savePresetButton.setTooltip("Overwrite selected preset with current settings");
    savePresetButton.setButtonText("Save");
    savePresetButton.setEnabled(false);
    savePresetButton.onClick = [this]()
    {
        const int id = presetBox.getSelectedId();
        // id == kPlaceholderId -> place holder, no actions
        if (id == kPlaceholderId)
            return;

        // only Preset 1/2/3 allowed
        if (id < 1 || id > 3)
            return;

        const int presetIdx = id - 1;
        auto d = captureUIToData();
        audioProcessor.savePreset(presetIdx, d);

        savePresetButton.setButtonText("Saved");
        // delay 1000 ms
        juce::Timer::callAfterDelay(1000, [btn = juce::Component::SafePointer<juce::TextButton>(&savePresetButton)]
        {
            if (btn != nullptr)
                btn->setButtonText("Save");
        });
    };

    // ==== 2nd row ====

    // Label
    addAndMakeVisible(rowGeneralLabel);
    rowGeneralLabel.setText("General:", juce::dontSendNotification);
    rowGeneralLabel.setJustificationType(juce::Justification::centredRight);
    rowGeneralLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(labelAlpha));
    rowGeneralLabel.setInterceptsMouseClicks(false, false);

    // Add and configure FPS dropdown
    addAndMakeVisible(fpsBox);
    fpsBox.setTooltip("UI frame rate (render FPS)");
    fpsBox.addItem("30 FPS", 30);
    fpsBox.addItem("60 FPS", 60);
    fpsBox.addItem("90 FPS", 90);
    fpsBox.addItem("120 FPS", 120);
    fpsBox.addItem("240 FPS", 240);

    fpsBox.setSelectedId(60);   // default: 60 FPS
    fpsBox.onChange = [this]()
    {
        spectrogram.setUiFps(fpsBox.getSelectedId());
        pushCurrentToProcessor();
    };

    spectrogram.setUiFps(fpsBox.getSelectedId());

    // Add and configure FFT size dropdown
    addAndMakeVisible(fftSizeBox);
    fftSizeBox.setTooltip("Select FFT window size. Larger = better frequency, worse time resolution.");
    fftSizeBox.addItem("512", 9);   // 2^9
    fftSizeBox.addItem("1024", 10);  // 2^10
    fftSizeBox.addItem("2048", 11);  // 2^11
    fftSizeBox.addItem("4096", 12);  // 2^12
    fftSizeBox.addItem("8192", 13);  // 2^13

    fftSizeBox.setSelectedId(11); // default: 2048
    fftSizeBox.onChange = [this]()
    {
        const int newOrder = fftSizeBox.getSelectedId();
        spectrogram.setFFTOrder(newOrder);
        updateLegendImage();
        repaint();
        pushCurrentToProcessor();
    };

    // Add and configure overlap dropdown
    addAndMakeVisible(overlapBox);
    overlapBox.setTooltip("FFT overlap (1/2/4/8/16/32/64), corresponding to overlap ratio 0/50/75/87.5/93.75/96.88/98.44 %. Affects time resolution. The larger the overlap, the more FFTs are performed per second, which means increased CPU overhead.");
    overlapBox.addItem("0 %", 1);
    overlapBox.addItem("50 %", 2);
    overlapBox.addItem("75 %", 4);
    overlapBox.addItem("87.5 %", 8);
    overlapBox.addItem("93.75 %", 16);
    overlapBox.addItem("96.88 %", 32);
    overlapBox.addItem("98.44 %", 64);
    overlapBox.setSelectedId(2); // default = 2 (50%)
    overlapBox.onChange = [this]()
    {
        spectrogram.setOverlap(overlapBox.getSelectedId());
        pushCurrentToProcessor();
    };

    // ==== 3rd row ====
    
    // Label
    addAndMakeVisible(rowSpectroLabel);
    rowSpectroLabel.setText("Mode:", juce::dontSendNotification);
    rowSpectroLabel.setJustificationType(juce::Justification::centredRight);
    rowSpectroLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(labelAlpha));
    rowSpectroLabel.setInterceptsMouseClicks(false, false);

    // Add and configure colour scheme combo box
    addAndMakeVisible(colourSchemeBox);
    colourSchemeBox.setTooltip("Select the color map used for the spectrogram display.");
    colourSchemeBox.addItem("Classic", static_cast<int>(SpectrogramComponent::ColourScheme::Classic));
    colourSchemeBox.addItem("Magma", static_cast<int>(SpectrogramComponent::ColourScheme::Magma));
    colourSchemeBox.addItem("Magma+", static_cast<int>(SpectrogramComponent::ColourScheme::MagmaEnhanced));
    colourSchemeBox.addItem("Grayscale", static_cast<int>(SpectrogramComponent::ColourScheme::Grayscale));
    colourSchemeBox.addItem("Grayscale+", static_cast<int>(SpectrogramComponent::ColourScheme::GrayscaleEnhanced));

    colourSchemeBox.setSelectedId(static_cast<int>(SpectrogramComponent::ColourScheme::Classic));
    // Create color legend (horizontal bar)
    updateLegendImage();
    colourSchemeBox.onChange = [this]()
    {
        auto selectedId = colourSchemeBox.getSelectedId();
        spectrogram.setColourScheme(static_cast<SpectrogramComponent::ColourScheme>(selectedId));
        updateLegendImage();
        repaint();
        pushCurrentToProcessor();
    };
    
    // Add and configure spectrogram mode combo box
    addAndMakeVisible(spectrogramModeBox);
    spectrogramModeBox.setTooltip(
        "Select the type of spectrogram to display.\n"
        "- Linear: Standard STFT spectrogram with linear or log frequency axis.\n"
        "- Linear+: Enhanced STFT spectrogram after time-frequency reassignment with linear or log frequency axis.\n"
        "- Mel: Mel-scaled spectrogram that spaces frequencies according to nonlinear human pitch perception.\n"
        "- Mel+: Mel-scaled spectrogram after time-frequency reassignment.\n"
        "- MFCC: Mel-frequency cepstral coefficient, representing timbral texture. Typically used in audio classification and speech recognition.\n"
        "- Spectral Centroid: STFT spectrogram with added curves showing where the energy is centered and how widely it is spread across frequencies.\n"
        "- Chroma: Chromagram showing the energy distribution across the 12 pitch classes (C to B), regardless of octave. Useful for analyzing harmonic content and key.\n"
        "- Fourier Tempogram: Tempo (BPM) energy vs. time computed from the onset envelope via STFT, overlays a dynamic tempo line (peak per frame with log-normal prior). Tip: Use a higher FFT size like 4096.\n"
        "- Autocorr Tempogram: Tempo (BPM) vs. time via autocorrelation of the onset envelope. More robust to local phase than Fourier Tempogram.\n"
        "- Spectral Contrast: Octave-band spectral contrast showing the ratio of spectral peaks to valleys in each frequency band. Bright bands indicate strong harmonic content; dark bands indicate noise-like spectra.\n"
        "- Spectral Flatness: Per-frame Wiener entropy (tonality coefficient). Measures how noise-like (close to 1.0) vs tone-like (close to 0.0) a sound is. Rendered as a height-proportional filled area chart.\n"
        "- Spectral Rolloff: Cumulative energy threshold curves (R25/R50/R85/R95) overlaid on STFT spectrogram.\n"
    );
    spectrogramModeBox.addItem("Linear", static_cast<int>(SpectrogramComponent::SpectrogramMode::Linear));
    spectrogramModeBox.addItem("Linear+", static_cast<int>(SpectrogramComponent::SpectrogramMode::LinearPlus));
    spectrogramModeBox.addItem("Mel", static_cast<int>(SpectrogramComponent::SpectrogramMode::Mel));
    spectrogramModeBox.addItem("Mel+", static_cast<int>(SpectrogramComponent::SpectrogramMode::MelPlus));
    spectrogramModeBox.addItem("MFCC", static_cast<int>(SpectrogramComponent::SpectrogramMode::MFCC));
    spectrogramModeBox.addItem("Spectral Centroid", static_cast<int>(SpectrogramComponent::SpectrogramMode::LinearWithCentroid));
    spectrogramModeBox.addItem("Spectral Contrast", static_cast<int>(SpectrogramComponent::SpectrogramMode::SpectralContrast));
    spectrogramModeBox.addItem("Spectral Flatness", static_cast<int>(SpectrogramComponent::SpectrogramMode::SpectralFlatness));
    spectrogramModeBox.addItem("Spectral Rolloff", static_cast<int>(SpectrogramComponent::SpectrogramMode::LinearWithRolloff));
    spectrogramModeBox.addItem("Chroma", static_cast<int>(SpectrogramComponent::SpectrogramMode::Chroma));
    spectrogramModeBox.addItem("Fourier Tempogram", static_cast<int>(SpectrogramComponent::SpectrogramMode::FourierTempogram));
    spectrogramModeBox.addItem("Autocorr Tempogram", static_cast<int>(SpectrogramComponent::SpectrogramMode::AutoTempogram));


    spectrogramModeBox.setSelectedId(static_cast<int>(SpectrogramComponent::SpectrogramMode::Linear));  // default: linear

    spectrogramModeBox.onChange = [this]()
    {
        auto selectedId = spectrogramModeBox.getSelectedId();
        auto mode = static_cast<SpectrogramComponent::SpectrogramMode>(selectedId);
        spectrogram.setSpectrogramMode(mode);
        updateLegendImage();
        // Enable/disable controls by mode
        MenuDisableControl(mode);
        repaint();
        pushCurrentToProcessor();
    };

    // Label
    addAndMakeVisible(menuDisplayLabel);
    menuDisplayLabel.setText("Display:", juce::dontSendNotification);
    menuDisplayLabel.setJustificationType(juce::Justification::centredRight);
    menuDisplayLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(labelAlpha));
    menuDisplayLabel.setInterceptsMouseClicks(false, false);

    // Add and configure dB floor slider (lower limit)
    addAndMakeVisible(floorDbSlider);
    floorDbSlider.setTooltip(
        "Set the dB floor (minimum brightness threshold) for spectrogram display.\n"
        "Note: Not applicable in MFCC and Chroma mode"
    );
    floorDbSlider.setRange(-200.0, -1.0, 1.0);  // floor dB from -400 to -20
    floorDbSlider.setValue(-100.0); // default: -100 dB
    floorDbSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);

    floorDbSlider.onValueChange = [this]()
    {
        float db = (float)floorDbSlider.getValue();
        spectrogram.setFloorDb(db);
        updateLegendImage();
        repaint();
        pushCurrentToProcessor();
    };
    // double click to reset
    floorDbSlider.setDoubleClickReturnValue(true, -100.0);

    // Add and configure norm factor slider (scale/brightness gain of dB values)
    addAndMakeVisible(normFactorSlider);
    normFactorSlider.setTooltip(
        "Set brightness scale factor (norm factor) for spectrogram display.\n"
        "Useful for adjusting overall dB level display."
    );
    normFactorSlider.setRange(0.001, 10.0, 0.001); // allow finer range
    normFactorSlider.setSkewFactorFromMidPoint(1.0); // nonlinear feel
    normFactorSlider.setValue(1.0);  // default
    normFactorSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);

    normFactorSlider.onValueChange = [this]()
    {
        float scale = (float)normFactorSlider.getValue();
        spectrogram.setNormFactor(scale);
        updateLegendImage();
        repaint();
        pushCurrentToProcessor();
    };
    // double click to reset
    normFactorSlider.setDoubleClickReturnValue(true, 1.0);

    // ==== 4th row ====

    // Label
    addAndMakeVisible(rowAxisLabel);
    rowAxisLabel.setText("Axes:", juce::dontSendNotification);
    rowAxisLabel.setJustificationType(juce::Justification::centredRight);
    rowAxisLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(labelAlpha));
    rowAxisLabel.setInterceptsMouseClicks(false, false);

    // Add and configure scroll speed dropdown
    addAndMakeVisible(scrollSpeedBox);
    scrollSpeedBox.setTooltip(
        "Horizontal scroll speed (x axis)"
    );
    scrollSpeedBox.addItem("x1", 1);
    scrollSpeedBox.addItem("x2", 2);
    scrollSpeedBox.addItem("x4", 4);
    scrollSpeedBox.addItem("x8", 8);
    scrollSpeedBox.addItem("x12", 12);
    scrollSpeedBox.addItem("x16", 16);

    scrollSpeedBox.setSelectedId(2); // default = 2
    scrollSpeedBox.onChange = [this]()
    {
        const int sp = scrollSpeedBox.getSelectedId();
        spectrogram.setScrollSpeedMultiplier(sp);
        pushCurrentToProcessor();
    };

    // Add and configure log scale (y axis) combo box
    addAndMakeVisible(logScaleBox);
    logScaleBox.setTooltip(
        "Select frequency y axis scale.\n"
        "Note: Not applicable in Mel-spectrogram / MFCC / Chroma mode"
    );
    logScaleBox.addItem("Linear Y", 1);
    logScaleBox.addItem("Log Y", 2);

    logScaleBox.setSelectedId(2);   // default: log
    logScaleBox.onChange = [this]()
    {
        bool useLog = logScaleBox.getSelectedId() == 2;
        spectrogram.setUseLogFrequency(useLog);
        pushCurrentToProcessor();
    };

    // y axis frequency range controls
    addAndMakeVisible(yRangeSlider);
    yRangeSlider.setTooltip(
        "Y-axis lower/upper bound in Hz (min/max frequency)\n"
        "Note: Not applicable in MFCC / Chroma mode"
    );
    yRangeSlider.setSliderStyle(juce::Slider::TwoValueHorizontal);
    yRangeSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    yRangeSlider.setPopupDisplayEnabled(false, false, nullptr);
    yRangeSlider.setNumDecimalPlacesToDisplay(0);
    // init maxHz
    const double maxHz = audioProcessor.getSampleRate() * 0.5;
    // range
    yRangeSlider.setRange(30.0, maxHz, 1.0);
    auto [fmin0, fmax0] = spectrogram.getFrequencyRangeHz();
    if (fmin0 <= 0) fmin0 = 30.0f;
    if (fmax0 <= 0) fmax0 = (float)maxHz;
    // slider position
    yRangeSlider.setMinAndMaxValues(fmin0, fmax0, juce::dontSendNotification);
    lastYMinHz = fmin0;
    lastYMaxHz = fmax0;
    // text box
    auto initHzEditor = [](juce::TextEditor& ed, const juce::String& tooltip)
    {
        ed.setTooltip(tooltip);
        ed.setInputRestrictions(7, "0123456789");
        ed.setJustification(juce::Justification::centredRight);
        ed.setSelectAllWhenFocused(true);
        ed.setTextToShowWhenEmpty("-", juce::Colours::grey);
        //ed.setColour(juce::TextEditor::backgroundColourId, juce::Colours::black.withAlpha(0.12f));
    };
    addAndMakeVisible(yMinHzEdit);
    addAndMakeVisible(yMaxHzEdit);
    initHzEditor(yMinHzEdit, "Y-axis lower bound (Hz)");
    initHzEditor(yMaxHzEdit, "Y-axis upper bound (Hz)");
    yMinHzEdit.setJustification(juce::Justification::centred);
    yMaxHzEdit.setJustification(juce::Justification::centred);
    yMinHzEdit.setText(juce::String((int)fmin0), juce::dontSendNotification);
    yMaxHzEdit.setText(juce::String((int)fmax0), juce::dontSendNotification);
    enum class WhichEnd { Min, Max, Unknown };
    auto applyYRange = [this](double a, double b, WhichEnd end)
    {
        if (updatingYControls) return;
        juce::ScopedValueSetter<bool> guard(updatingYControls, true);

        const double maxHzNow = audioProcessor.getSampleRate() * 0.5;

        a = juce::jlimit(30.0, maxHzNow - kMinBandWidthHz, a);
        b = juce::jlimit(30.0 + kMinBandWidthHz, maxHzNow, b);

        // ensure min range
        if (b - a < kMinBandWidthHz)
        {
            if (end == WhichEnd::Min)          b = std::min(maxHzNow, a + kMinBandWidthHz);
            else if (end == WhichEnd::Max)     a = std::max(30.0, b - kMinBandWidthHz);
            else {
                b = std::max(b, a + kMinBandWidthHz);
                b = std::min(b, maxHzNow);
            }
        }
        // set values
        yRangeSlider.setMinAndMaxValues(a, b, juce::dontSendNotification);

        // set text box
        yMinHzEdit.setText(juce::String((int)a), juce::dontSendNotification);
        yMaxHzEdit.setText(juce::String((int)b), juce::dontSendNotification);

        // apply to spectrogram
        spectrogram.setFrequencyRangeHz((float)a, (float)b);

        lastYMinHz = a;
        lastYMaxHz = b;
    };
    // connect slider
    yRangeSlider.onValueChange = [this, applyYRange]()
    {
        double a = yRangeSlider.getMinValue();
        double b = yRangeSlider.getMaxValue();

        WhichEnd end = std::abs(a - lastYMinHz) >= std::abs(b - lastYMaxHz) ? WhichEnd::Min : WhichEnd::Max;
        applyYRange(a, b, end);
        pushCurrentToProcessor();
    };
    // connect text box
    auto commitMin = [this, applyYRange]()
    {
        const double a = yMinHzEdit.getText().getDoubleValue();
        const double b = yRangeSlider.getMaxValue();
        applyYRange(a, b, WhichEnd::Min);
    };
    auto commitMax = [this, applyYRange]()
    {
        const double a = yRangeSlider.getMinValue();
        const double b = yMaxHzEdit.getText().getDoubleValue();
        applyYRange(a, b, WhichEnd::Max);
    };

    yMinHzEdit.onReturnKey = commitMin;
    yMinHzEdit.onFocusLost = commitMin;
    yMaxHzEdit.onReturnKey = commitMax;
    yMaxHzEdit.onFocusLost = commitMax;
    // double click to reset
    yRangeSlider.setMinAndMaxValues(30.0, audioProcessor.getSampleRate() * 0.5, juce::sendNotificationSync);
    // refresh y range slider
    const double srNow = audioProcessor.getSampleRate();
    if (srNow > 0.0)
        refreshYRangeSliderForSampleRate(srNow);

    // toggle button for y axis (note)
    addAndMakeVisible(noteAxisToggle);
    noteAxisToggle.setButtonText("Note Axis");
    noteAxisToggle.setTooltip("Show note axis. Note: Not applicable in MFCC / Chroma mode");
    noteAxisToggle.onClick = [this]()
    {
        spectrogram.setShowNoteCAxis(noteAxisToggle.getToggleState());
        repaint();
        pushCurrentToProcessor();
    };

    spectrogram.onRolloffVisibilityChanged = [this]()
    {
        pushCurrentToProcessor();
    };

    // reset tempo avg button in Tempogram
    addAndMakeVisible(tempoAvgResetBtn);
    tempoAvgResetBtn.setTooltip("Reset average BPM");
    tempoAvgResetBtn.setWantsKeyboardFocus(false);
    tempoAvgResetBtn.onClick = [this] 
    {
        spectrogram.resetTempoStats();
        //repaint();
        pushCurrentToProcessor();
    };
    tempoAvgResetBtn.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    tempoAvgResetBtn.setColour(juce::TextButton::textColourOnId, juce::Colours::lightgrey);
    tempoAvgResetBtn.setColour(juce::TextButton::textColourOffId, juce::Colours::lightgrey);
    tempoAvgResetBtn.setVisible(false);


    applyDataToUI(audioProcessor.getCurrentSettings(), true);

}

void SpectrogramAudioProcessorEditor::MenuDisableControl(SpectrogramComponent::SpectrogramMode mode)
{
    switch (mode)
    {
        // Mel
        case SpectrogramComponent::SpectrogramMode::Mel:
        case SpectrogramComponent::SpectrogramMode::MelPlus:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(true);
            yMinHzEdit.setEnabled(true);
            yMaxHzEdit.setEnabled(true);
            floorDbSlider.setEnabled(true);
            noteAxisToggle.setEnabled(true);
            tempoAvgResetBtn.setVisible(false);
            break;
        }
        // MFCC
        case SpectrogramComponent::SpectrogramMode::MFCC:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(false);
            yMinHzEdit.setEnabled(false);
            yMaxHzEdit.setEnabled(false);
            floorDbSlider.setEnabled(false);
            noteAxisToggle.setEnabled(false);
            tempoAvgResetBtn.setVisible(false);
            break;
        }
        // Chroma
        case SpectrogramComponent::SpectrogramMode::Chroma:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(false);
            yMinHzEdit.setEnabled(false);
            yMaxHzEdit.setEnabled(false);
            floorDbSlider.setEnabled(false);
            noteAxisToggle.setEnabled(false);
            tempoAvgResetBtn.setVisible(false);
            break;
        }
        // Spectral Contrast
        case SpectrogramComponent::SpectrogramMode::SpectralContrast:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(false);
            yMinHzEdit.setEnabled(false);
            yMaxHzEdit.setEnabled(false);
            floorDbSlider.setEnabled(true);
            noteAxisToggle.setEnabled(false);
            tempoAvgResetBtn.setVisible(false);
            break;
        }
        // Spectral Flatness
        case SpectrogramComponent::SpectrogramMode::SpectralFlatness:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(false);
            yMinHzEdit.setEnabled(false);
            yMaxHzEdit.setEnabled(false);
            floorDbSlider.setEnabled(true);
            noteAxisToggle.setEnabled(false);
            tempoAvgResetBtn.setVisible(false);
            break;
        }
        // Fourier Tempogram
        case SpectrogramComponent::SpectrogramMode::FourierTempogram:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(false);
            yMinHzEdit.setEnabled(false);
            yMaxHzEdit.setEnabled(false);
            floorDbSlider.setEnabled(true);
            noteAxisToggle.setEnabled(false);
            tempoAvgResetBtn.setVisible(true);
            // auto switch FFT size to >=4096 for better tempogram results
            const int curOrder = fftSizeBox.getSelectedId();
            if (curOrder < 12)  // 2^12=4096
                fftSizeBox.setSelectedId(12, juce::sendNotificationSync);
            break;
        }
        // Autocorr Tempogram
        case SpectrogramComponent::SpectrogramMode::AutoTempogram:
        {
            logScaleBox.setEnabled(false);
            yRangeSlider.setEnabled(false);
            yMinHzEdit.setEnabled(false);
            yMaxHzEdit.setEnabled(false);
            floorDbSlider.setEnabled(true);
            noteAxisToggle.setEnabled(false);
            tempoAvgResetBtn.setVisible(true);
            // auto switch FFT size to =2048 for better tempogram results
            const int curOrder = fftSizeBox.getSelectedId();
            if (curOrder != 11)  // 2^11=2048
                fftSizeBox.setSelectedId(11, juce::sendNotificationSync);

            break;
        }
        // Spectral Rolloff (falls through to default — same controls as Linear/Linear+)
        case SpectrogramComponent::SpectrogramMode::LinearWithRolloff:
        // Linear STFT
        default:
            logScaleBox.setEnabled(true);
            yRangeSlider.setEnabled(true);
            yMinHzEdit.setEnabled(true);
            yMaxHzEdit.setEnabled(true);
            floorDbSlider.setEnabled(true);
            noteAxisToggle.setEnabled(true);
            tempoAvgResetBtn.setVisible(false);
            break;
    }
}

SpectrogramAudioProcessorEditor::~SpectrogramAudioProcessorEditor() noexcept
{
}

//==============================================================================
void SpectrogramAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Fill background with the default look-and-feel color
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));

    // Legend aligned to topBar (same vertical height), placed at top-right corner
    const int topBarHeight = 30;
    const int baseMargin = 40;
    const int rightReserve = baseMargin + toggleW + gap;

    if (controlsVisible)
    {
        const int legendX = getWidth() - legendImage.getWidth() - rightReserve;
        const int legendY = (topBarHeight - legendImage.getHeight()) / 2;  // vertical center inside topBar

        // Draw legend color bar
        g.drawImage(legendImage, legendX, legendY, legendImage.getWidth(), legendImage.getHeight(),
            0, 0, legendImage.getWidth(), legendImage.getHeight());

        // dB labels
        g.setColour(juce::Colours::white);
        g.setFont(12.0f);

        if (spectrogram.getCurrentMode() == SpectrogramComponent::SpectrogramMode::MFCC ||
            spectrogram.getCurrentMode() == SpectrogramComponent::SpectrogramMode::Chroma ||
            spectrogram.getCurrentMode() == SpectrogramComponent::SpectrogramMode::SpectralContrast ||
            spectrogram.getCurrentMode() == SpectrogramComponent::SpectrogramMode::SpectralFlatness)
        {
            // normalized legend label for MFCC and Chromagram [0, 1]
            g.drawText("0.0", legendX - 50, legendY, 45, legendImage.getHeight(), juce::Justification::right);
            g.drawText("1.0", legendX + legendImage.getWidth() + 5, legendY, 40, legendImage.getHeight(), juce::Justification::left);
        }
        else
        {
            g.drawText(juce::String((int)spectrogram.getFloorDb()) + " dB", legendX - 50, legendY, 45, legendImage.getHeight(), juce::Justification::right);
            g.drawText("0 dB", legendX + legendImage.getWidth() + 5, legendY, 40, legendImage.getHeight(), juce::Justification::left);
        }
    }
}

void SpectrogramAudioProcessorEditor::updateLegendImage()
{
    const int legendWidth = 180;
    const int legendHeight = 20;

    legendImage = juce::Image(juce::Image::RGB, legendWidth, legendHeight, false);

    for (int x = 0; x < legendWidth; ++x)
    {
        float value = static_cast<float>(x) / (legendWidth - 1);
        juce::Colour colour = spectrogram.getColourForValue(value);

        for (int y = 0; y < legendHeight; ++y)
            legendImage.setPixelAt(x, y, colour);
    }
}

void SpectrogramAudioProcessorEditor::refreshYRangeSliderForSampleRate(double sr)
{
    const double maxHz = sr * 0.5;

    yRangeSlider.setRange(30.0, maxHz, 1.0);

    auto [a0, b0] = spectrogram.getFrequencyRangeHz();
    double a = juce::jlimit(1.0, maxHz - 1.0, (double)a0);
    double b = juce::jlimit(a + kMinBandWidthHz, maxHz, (double)b0);

    updatingYControls = true;
    yRangeSlider.setMinAndMaxValues(a, b, juce::dontSendNotification);
    yMinHzEdit.setText(juce::String((int)a), juce::dontSendNotification);
    yMaxHzEdit.setText(juce::String((int)b), juce::dontSendNotification);
    updatingYControls = false;

    lastYMinHz = a;
    lastYMaxHz = b;

    spectrogram.setFrequencyRangeHz((float)a, (float)b);
}

SpectrogramAudioProcessor::PresetData SpectrogramAudioProcessorEditor::captureUIToData() const
{
    SpectrogramAudioProcessor::PresetData d;
    d.fpsId = fpsBox.getSelectedId();
    d.fftOrderId = fftSizeBox.getSelectedId();
    d.overlapId = overlapBox.getSelectedId();

    d.modeId = spectrogramModeBox.getSelectedId();
    d.colourId = colourSchemeBox.getSelectedId();

    d.floorDb = floorDbSlider.getValue();
    d.normFactor = normFactorSlider.getValue();

    d.scrollSpeedId = scrollSpeedBox.getSelectedId();
    d.logScaleId = logScaleBox.getSelectedId();

    d.yMinHz = yRangeSlider.getMinValue();
    d.yMaxHz = yRangeSlider.getMaxValue();

    d.noteAxis = noteAxisToggle.getToggleState();

    d.rolloffR25Visible = spectrogram.getRolloffCurveVisible(0);
    d.rolloffR50Visible = spectrogram.getRolloffCurveVisible(1);
    d.rolloffR85Visible = spectrogram.getRolloffCurveVisible(2);
    d.rolloffR95Visible = spectrogram.getRolloffCurveVisible(3);

    return d;
}

void SpectrogramAudioProcessorEditor::pushCurrentToProcessor()
{
    audioProcessor.setCurrentSettings(captureUIToData());
}

void SpectrogramAudioProcessorEditor::applyDataToUI(
    const SpectrogramAudioProcessor::PresetData& d,
    bool alsoApplyToSpectrogram)
{
    juce::ScopedValueSetter<bool> guard(isApplyingPreset, true);

    fpsBox.setSelectedId(d.fpsId, juce::dontSendNotification);
    fftSizeBox.setSelectedId(d.fftOrderId, juce::dontSendNotification);
    overlapBox.setSelectedId(d.overlapId, juce::dontSendNotification);

    spectrogramModeBox.setSelectedId(d.modeId, juce::dontSendNotification);
    colourSchemeBox.setSelectedId(d.colourId, juce::dontSendNotification);

    floorDbSlider.setValue(d.floorDb, juce::dontSendNotification);
    normFactorSlider.setValue(d.normFactor, juce::dontSendNotification);

    scrollSpeedBox.setSelectedId(d.scrollSpeedId, juce::dontSendNotification);
    logScaleBox.setSelectedId(d.logScaleId, juce::dontSendNotification);

    yRangeSlider.setMinAndMaxValues(d.yMinHz, d.yMaxHz, juce::dontSendNotification);
    yMinHzEdit.setText(juce::String((int)d.yMinHz), juce::dontSendNotification);
    yMaxHzEdit.setText(juce::String((int)d.yMaxHz), juce::dontSendNotification);

    noteAxisToggle.setToggleState(d.noteAxis, juce::dontSendNotification);

    if (!alsoApplyToSpectrogram)
        return;

    spectrogram.setUiFps(d.fpsId);
    spectrogram.setFFTOrder(d.fftOrderId);
    spectrogram.setOverlap(d.overlapId);

    auto mode = (SpectrogramComponent::SpectrogramMode)d.modeId;
    spectrogram.setSpectrogramMode(mode);
    MenuDisableControl(mode);

    spectrogram.setColourScheme((SpectrogramComponent::ColourScheme)d.colourId);
    spectrogram.setFloorDb((float)d.floorDb);
    spectrogram.setNormFactor((float)d.normFactor);

    spectrogram.setScrollSpeedMultiplier(d.scrollSpeedId);
    spectrogram.setUseLogFrequency(d.logScaleId == 2);

    spectrogram.setFrequencyRangeHz((float)d.yMinHz, (float)d.yMaxHz);
    spectrogram.setShowNoteCAxis(d.noteAxis);

    spectrogram.setRolloffCurveVisible(0, d.rolloffR25Visible);
    spectrogram.setRolloffCurveVisible(1, d.rolloffR50Visible);
    spectrogram.setRolloffCurveVisible(2, d.rolloffR85Visible);
    spectrogram.setRolloffCurveVisible(3, d.rolloffR95Visible);

    updateLegendImage();
    repaint();
}


void SpectrogramAudioProcessorEditor::updateControlsVisibility()
{
    // 1st
    freezeButton.setVisible(controlsVisible);
    presetBox.setVisible(controlsVisible);
    savePresetButton.setVisible(controlsVisible);
    // 2nd
    rowGeneralLabel.setVisible(controlsVisible);
    fpsBox.setVisible(controlsVisible);
    fftSizeBox.setVisible(controlsVisible);
    overlapBox.setVisible(controlsVisible);
    // 3rd
    rowSpectroLabel.setVisible(controlsVisible);
    colourSchemeBox.setVisible(controlsVisible);
    spectrogramModeBox.setVisible(controlsVisible);
    menuDisplayLabel.setVisible(controlsVisible);
    logScaleBox.setVisible(controlsVisible);
    floorDbSlider.setVisible(controlsVisible);
    normFactorSlider.setVisible(controlsVisible);
    // 4th
    rowAxisLabel.setVisible(controlsVisible);
    scrollSpeedBox.setVisible(controlsVisible);
    yRangeSlider.setVisible(controlsVisible);
    yMinHzEdit.setVisible(controlsVisible);
    yMaxHzEdit.setVisible(controlsVisible);
    noteAxisToggle.setVisible(controlsVisible);
}

void SpectrogramAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();

    // fixed small button for menu visibility
    const int baseMargin = 3;
    toggleUiButton.setBounds(getWidth() - baseMargin - toggleW, baseMargin, toggleW, toggleW);

    const int rowH = controlsVisible ? kRowHeight : 0;

    // first row: freeze button & presets
    auto firstRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // freeze button
        freezeButton.setBounds(firstRow.removeFromLeft(100).reduced(5));
        // presets
        presetBox.setBounds(firstRow.removeFromLeft(110).reduced(5));
        savePresetButton.setBounds(firstRow.removeFromLeft(70).reduced(5));
    }

    // second row: general settings
    auto secondRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // label
        rowGeneralLabel.setBounds(secondRow.removeFromLeft(60));
        // UI FPS
        fpsBox.setBounds(secondRow.removeFromLeft(100).reduced(5));
        // FFT size dropdown
        fftSizeBox.setBounds(secondRow.removeFromLeft(100).reduced(5));
        // overlap
        overlapBox.setBounds(secondRow.removeFromLeft(100).reduced(5));
    }

    // third row: spectrogram settings
    auto thirdRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // label
        rowSpectroLabel.setBounds(thirdRow.removeFromLeft(60));
        // spectrogram mode
        spectrogramModeBox.setBounds(thirdRow.removeFromLeft(100).reduced(5));
        // label
        menuDisplayLabel.setBounds(thirdRow.removeFromLeft(60));
        // colour scheme
        colourSchemeBox.setBounds(thirdRow.removeFromLeft(100).reduced(5));
        // slider floor value colour scheme
        floorDbSlider.setBounds(thirdRow.removeFromLeft(200).reduced(5));
        // slider norm factor
        normFactorSlider.setBounds(thirdRow.removeFromLeft(200).reduced(5));
    }

    // fourth row: x/y axis control
    auto fourthRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // label
        rowAxisLabel.setBounds(fourthRow.removeFromLeft(60));
        // x axis scroll speed
        scrollSpeedBox.setBounds(fourthRow.removeFromLeft(100).reduced(5));
        // y axis type: log or linear (for linear STFT spectrogram)
        logScaleBox.setBounds(fourthRow.removeFromLeft(100).reduced(5));
        // slider y axis range
        yMinHzEdit.setBounds(fourthRow.removeFromLeft(60).reduced(5));
        yRangeSlider.setBounds(fourthRow.removeFromLeft(200).reduced(5));
        yMaxHzEdit.setBounds(fourthRow.removeFromLeft(60).reduced(5));
        // y axis (note)
        noteAxisToggle.setBounds(fourthRow.removeFromLeft(130).reduced(5));
    }

    // rest: spectrogram
    spectrogram.setBounds(area);

    // reset tempo avg button in Tempogram
    auto spec = spectrogram.getBounds();
    const int resTempoBtnHeight = 18;
    const int resTempoBtnWidth = 48;
    tempoAvgResetBtn.setBounds(spec.getRight() - 70, spec.getY() + 28, resTempoBtnWidth, resTempoBtnHeight);
}
