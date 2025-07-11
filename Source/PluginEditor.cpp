/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

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
    setResizeLimits(400, 300, 1600, 1400);

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
    colourSchemeBox.addItem("Grayscale", static_cast<int>(SpectrogramComponent::ColourScheme::Grayscale));

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
        "- MFCC: Mel-frequency cepstral coefficient, representing timbral texture. Typically used in audio classification and speech recognition."
    );
    spectrogramModeBox.addItem("Linear", static_cast<int>(SpectrogramComponent::SpectrogramMode::Linear));
    spectrogramModeBox.addItem("Mel", static_cast<int>(SpectrogramComponent::SpectrogramMode::Mel));
    spectrogramModeBox.addItem("MFCC", static_cast<int>(SpectrogramComponent::SpectrogramMode::MFCC));

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
        "Select frequency axis scale.\n"
        "Note: Not applicable in Mel-spectrogram & MFCC mode"
    );
    logScaleBox.addItem("Linear Scale", 1);
    logScaleBox.addItem("Log Scale", 2);

    logScaleBox.setSelectedId(2);   // default: log
    logScaleBox.onChange = [this]()
    {
        bool useLog = logScaleBox.getSelectedId() == 2;
        spectrogram.setUseLogFrequency(useLog);
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

    if (spectrogram.getCurrentMode() == SpectrogramComponent::SpectrogramMode::MFCC)
    {
        g.drawText("0.0", legendX - 50, legendY, 45, legendImage.getHeight(), juce::Justification::right);
        g.drawText("1.0", legendX + legendImage.getWidth() + 5, legendY, 40, legendImage.getHeight(), juce::Justification::left);
    }
    else
    {
        g.drawText("-100 dB", legendX - 50, legendY, 45, legendImage.getHeight(), juce::Justification::right);
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

    // second row: dropdown menu
    auto secondRow = area.removeFromTop(30);
    colourSchemeBox.setBounds(secondRow.removeFromLeft(110).reduced(5));
    spectrogramModeBox.setBounds(secondRow.removeFromLeft(110).reduced(5));
    logScaleBox.setBounds(secondRow.removeFromLeft(110).reduced(5));

    // rest: spectrogram
    spectrogram.setBounds(area);
}
