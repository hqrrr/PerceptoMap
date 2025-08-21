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

    // Label
    addAndMakeVisible(row1stLabel);
    row1stLabel.setText("General:", juce::dontSendNotification);
    row1stLabel.setJustificationType(juce::Justification::centredRight);
    row1stLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.5f));
    row1stLabel.setInterceptsMouseClicks(false, false);

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
    };

    // Add and configure overlap dropdown
    addAndMakeVisible(overlapBox);
    overlapBox.setTooltip("FFT overlap (1/2/4/8), corresponding to overlap ratio 0/50/75/87.5 %. Affects time resolution. The larger the overlap, the more FFTs are performed per second, which means increased CPU overhead.");
    overlapBox.addItem("0 %", 1);
    overlapBox.addItem("50 %", 2);
    overlapBox.addItem("75 %", 4);
    overlapBox.addItem("87.5 %", 8);
    overlapBox.setSelectedId(2); // default = 2 (50%)
    overlapBox.onChange = [this]()
    {
        spectrogram.setOverlap(overlapBox.getSelectedId());
    };

    // ==== 2nd row ====
    
    // Label
    addAndMakeVisible(row2ndLabel);
    row2ndLabel.setText("Modes:", juce::dontSendNotification);
    row2ndLabel.setJustificationType(juce::Justification::centredRight);
    row2ndLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.5f));
    row2ndLabel.setInterceptsMouseClicks(false, false);

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
        "- Chroma: Chromagram showing the energy distribution across the 12 pitch classes (C to B), regardless of octave. Useful for analyzing harmonic content and key."
    );
    spectrogramModeBox.addItem("Linear", static_cast<int>(SpectrogramComponent::SpectrogramMode::Linear));
    spectrogramModeBox.addItem("Linear+", static_cast<int>(SpectrogramComponent::SpectrogramMode::LinearPlus));
    spectrogramModeBox.addItem("Mel", static_cast<int>(SpectrogramComponent::SpectrogramMode::Mel));
    spectrogramModeBox.addItem("Mel+", static_cast<int>(SpectrogramComponent::SpectrogramMode::MelPlus));
    spectrogramModeBox.addItem("MFCC", static_cast<int>(SpectrogramComponent::SpectrogramMode::MFCC));
    spectrogramModeBox.addItem("Spectral Centroid", static_cast<int>(SpectrogramComponent::SpectrogramMode::LinearWithCentroid));
    spectrogramModeBox.addItem("Chroma", static_cast<int>(SpectrogramComponent::SpectrogramMode::Chroma));

    spectrogramModeBox.setSelectedId(static_cast<int>(SpectrogramComponent::SpectrogramMode::Linear));  // default: linear

    spectrogramModeBox.onChange = [this]()
    {
        auto selectedId = spectrogramModeBox.getSelectedId();
        spectrogram.setSpectrogramMode(static_cast<SpectrogramComponent::SpectrogramMode>(selectedId));
        updateLegendImage();
        repaint();
    };

    // Label
    addAndMakeVisible(menuDisplayLabel);
    menuDisplayLabel.setText("Display:", juce::dontSendNotification);
    menuDisplayLabel.setJustificationType(juce::Justification::centredRight);
    menuDisplayLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.5f));
    menuDisplayLabel.setInterceptsMouseClicks(false, false);

    // Add and configure dB floor slider (lower limit)
    addAndMakeVisible(floorDbSlider);
    floorDbSlider.setTooltip(
        "Set the dB floor (minimum brightness threshold) for spectrogram display.\n"
        "Note: Not applicable in MFCC and Chroma mode"
    );
    floorDbSlider.setRange(-200.0, -1.0, 1.0);  // floor dB from -400 to -20
    floorDbSlider.setValue(-100.0); // default: -100 dB
    floorDbSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 80, 20);

    floorDbSlider.onValueChange = [this]()
        {
            float db = (float)floorDbSlider.getValue();
            spectrogram.setFloorDb(db);
            updateLegendImage();
            repaint();
        };
    // double click to reset
    floorDbSlider.setDoubleClickReturnValue(true, -100.0);

    // Add and configure norm factor slider (scale/brightness gain of dB values)
    addAndMakeVisible(normFactorSlider);
    normFactorSlider.setTooltip(
        "Set brightness scale factor (norm factor) for spectrogram display.\n"
        "Useful for adjusting overall dB level display."
    );
    normFactorSlider.setRange(0.001, 5.0, 0.001); // allow finer range
    normFactorSlider.setSkewFactorFromMidPoint(1.0); // nonlinear feel
    normFactorSlider.setValue(1.0);  // default
    normFactorSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 80, 20);

    normFactorSlider.onValueChange = [this]()
        {
            float scale = (float)normFactorSlider.getValue();
            spectrogram.setNormFactor(scale);
            updateLegendImage();
            repaint();
        };
    // double click to reset
    normFactorSlider.setDoubleClickReturnValue(true, 1.0);

    // ==== 3rd row ====

    // Label
    addAndMakeVisible(row3rdLabel);
    row3rdLabel.setText("Axes:", juce::dontSendNotification);
    row3rdLabel.setJustificationType(juce::Justification::centredRight);
    row3rdLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.5f));
    row3rdLabel.setInterceptsMouseClicks(false, false);

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
            spectrogram.getCurrentMode() == SpectrogramComponent::SpectrogramMode::Chroma)
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

void SpectrogramAudioProcessorEditor::updateControlsVisibility()
{
    // 1st
    freezeButton.setVisible(controlsVisible);
    row1stLabel.setVisible(controlsVisible);
    fpsBox.setVisible(controlsVisible);
    fftSizeBox.setVisible(controlsVisible);
    overlapBox.setVisible(controlsVisible);
    // 2nd
    row2ndLabel.setVisible(controlsVisible);
    colourSchemeBox.setVisible(controlsVisible);
    spectrogramModeBox.setVisible(controlsVisible);
    menuDisplayLabel.setVisible(controlsVisible);
    logScaleBox.setVisible(controlsVisible);
    floorDbSlider.setVisible(controlsVisible);
    normFactorSlider.setVisible(controlsVisible);
    // 3rd
    row3rdLabel.setVisible(controlsVisible);
    scrollSpeedBox.setVisible(controlsVisible);
    yRangeSlider.setVisible(controlsVisible);
    yMinHzEdit.setVisible(controlsVisible);
    yMaxHzEdit.setVisible(controlsVisible);
}

void SpectrogramAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();

    // fixed small button for menu visibility
    const int baseMargin = 6;
    toggleUiButton.setBounds(getWidth() - baseMargin - toggleW, baseMargin / 2, toggleW, toggleW);

    const int rowH = controlsVisible ? kRowHeight : 0;

    // top row: freeze button & FFT settings & legend bar
    auto topRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // freeze button
        freezeButton.setBounds(topRow.removeFromLeft(100).reduced(5));
        // label
        row1stLabel.setBounds(topRow.removeFromLeft(60));
        // UI FPS
        fpsBox.setBounds(topRow.removeFromLeft(90).reduced(5));
        // FFT size dropdown
        fftSizeBox.setBounds(topRow.removeFromLeft(90).reduced(5));
        // overlap
        overlapBox.setBounds(topRow.removeFromLeft(90).reduced(5));
    }

    // second row: spectrogram settings
    auto secondRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // label
        row2ndLabel.setBounds(secondRow.removeFromLeft(60));
        // colour scheme
        colourSchemeBox.setBounds(secondRow.removeFromLeft(100).reduced(5));
        // spectrogram mode
        spectrogramModeBox.setBounds(secondRow.removeFromLeft(100).reduced(5));
        // label
        menuDisplayLabel.setBounds(secondRow.removeFromLeft(60));
        // slider floor value colour scheme
        floorDbSlider.setBounds(secondRow.removeFromLeft(200).reduced(5));
        // slider norm factor
        normFactorSlider.setBounds(secondRow.removeFromLeft(200).reduced(5));
    }

    // third row: x/y axis control
    auto thirdRow = area.removeFromTop(rowH);
    if (controlsVisible)
    {
        // label
        row3rdLabel.setBounds(thirdRow.removeFromLeft(60));
        // x axis scroll speed
        scrollSpeedBox.setBounds(thirdRow.removeFromLeft(100).reduced(5));
        // y axis type: log or linear (for linear STFT spectrogram)
        logScaleBox.setBounds(thirdRow.removeFromLeft(100).reduced(5));
        // slider y axis range
        yMinHzEdit.setBounds(thirdRow.removeFromLeft(80).reduced(5));
        yRangeSlider.setBounds(thirdRow.removeFromLeft(200).reduced(5));
        yMaxHzEdit.setBounds(thirdRow.removeFromLeft(80).reduced(5));
    }

    // rest: spectrogram
    spectrogram.setBounds(area);
}
