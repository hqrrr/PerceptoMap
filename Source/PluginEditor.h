/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "SpectrogramComponent.h"

//==============================================================================
/**
    This class implements the main editor window for the plugin.
    It hosts and manages the SpectrogramComponent for visualization.
*/
class SpectrogramAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    SpectrogramAudioProcessorEditor(SpectrogramAudioProcessor&);
    ~SpectrogramAudioProcessorEditor() noexcept override;

    //==============================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

    // Optional accessor if processor needs to push audio data to this
    SpectrogramComponent& getSpectrogramComponent() { return spectrogram; }

private:
    juce::TooltipWindow tooltipWindow;

    SpectrogramAudioProcessor& audioProcessor;

    // The visual spectrogram component
    SpectrogramComponent spectrogram;
    // legend
    juce::Image legendImage;
    void updateLegendImage();

    // freeze button
    juce::TextButton freezeButton{ "Freeze" };
    bool isFrozen = false;

    // dropdown menu for spectrogram color scheme
    juce::ComboBox colourSchemeBox;
    // dropdown menu for log scale
    juce::ComboBox logScaleBox;
    // dropdown menu for spectrogram mode, e.g. linear / mel-scaled ...
    juce::ComboBox spectrogramModeBox;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramAudioProcessorEditor)
};