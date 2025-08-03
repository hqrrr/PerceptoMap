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

    // Add and configure freeze button
    addAndMakeVisible(freezeButton);
    freezeButton.setTooltip("Freeze or resume spectrogram scrolling");
    freezeButton.onClick = [this]()
    {
        isFrozen = !isFrozen;
        freezeButton.setButtonText(isFrozen ? "Resume" : "Freeze");
        spectrogram.setFrozen(isFrozen);
    };

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
        "- Mel: Mel-scaled spectrogram that spaces frequencies according to nonlinear human pitch perception.\n"
        "- MFCC: Mel-frequency cepstral coefficient, representing timbral texture. Typically used in audio classification and speech recognition.\n"
        "- Spectral Centroid: STFT spectrogram with added curves showing where the energy is centered and how widely it is spread across frequencies.\n"
        "- Chroma: Chromagram showing the energy distribution across the 12 pitch classes (C to B), regardless of octave. Useful for analyzing harmonic content and key."
    );
    spectrogramModeBox.addItem("Linear", static_cast<int>(SpectrogramComponent::SpectrogramMode::Linear));
    spectrogramModeBox.addItem("Mel", static_cast<int>(SpectrogramComponent::SpectrogramMode::Mel));
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

    // Add and configure norm factor slider (scale/brightness gain of dB values)
    addAndMakeVisible(normFactorSlider);
    normFactorSlider.setTooltip(
        "Set brightness scale factor (norm factor) for spectrogram display.\n"
        "Useful for adjusting overall dB level display."
    );
    normFactorSlider.setRange(0.001, 2.0, 0.001); // allow finer range
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
    const int margin = 40;

    const int legendX = getWidth() - legendImage.getWidth() - margin;
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

void SpectrogramAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();
    // top row: freeze button & legend bar
    auto topRow = area.removeFromTop(30);
    freezeButton.setBounds(topRow.removeFromLeft(100).reduced(5));

    // second row: dropdown menu etc.
    auto secondRow = area.removeFromTop(30);
    // colour scheme
    colourSchemeBox.setBounds(secondRow.removeFromLeft(110).reduced(5));
    // spectrogram mode
    spectrogramModeBox.setBounds(secondRow.removeFromLeft(110).reduced(5));
    // y axis type: log or linear (for linear STFT spectrogram)
    logScaleBox.setBounds(secondRow.removeFromLeft(110).reduced(5));
    // slider floor value colour scheme
    floorDbSlider.setBounds(secondRow.removeFromLeft(200).reduced(5));
    // slider norm factor
    normFactorSlider.setBounds(secondRow.removeFromLeft(200).reduced(5));

    // rest: spectrogram
    spectrogram.setBounds(area);
}
