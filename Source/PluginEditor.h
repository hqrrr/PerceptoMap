/*
  ==============================================================================

    PluginEditor.h
    GUI interface for the PerceptoMap spectrogram plugin.

    Author: hqrrr
    GitHub: https://github.com/hqrrr/PerceptoMap

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
    SpectrogramComponent& getSpectrogramComponent() {return spectrogram;}

    // display/hide menu button text
    static constexpr const char* ShowMenuText = "+";
    static constexpr const char* HideMenuText = "-";

    // update y range slider if sample rate changed
    void refreshYRangeSliderForSampleRate(double sr);

private:
    juce::TooltipWindow tooltipWindow;

    SpectrogramAudioProcessor& audioProcessor;

    // The visual spectrogram component
    SpectrogramComponent spectrogram;
    // legend
    juce::Image legendImage;
    void updateLegendImage();

    // Disable/enable controls
    void MenuDisableControl(SpectrogramComponent::SpectrogramMode mode);

    // display/hide menu button
    juce::TextButton toggleUiButton{HideMenuText};
    bool controlsVisible = true;
    static constexpr int kRowHeight = 30;    // Height of each row
    const int toggleW = 24;
    const int gap = 8;
    void updateControlsVisibility();

    // freeze button
    juce::TextButton freezeButton{"Freeze"};
    bool isFrozen = false;

    // dropdown menu for UI FPS
    juce::ComboBox fpsBox;
    // dropdown menu for FFT size
    juce::ComboBox fftSizeBox;
    // dropdown menu for scroll speed
    juce::ComboBox scrollSpeedBox;
    // dropdown menu for overlap
    juce::ComboBox overlapBox;

    // dropdown menu for spectrogram color scheme
    juce::ComboBox colourSchemeBox;
    // dropdown menu for log scale
    juce::ComboBox logScaleBox;
    // dropdown menu for spectrogram mode, e.g. linear / mel-scaled ...
    juce::ComboBox spectrogramModeBox;

    // slider to change the lower limit (floor value) of all dB colour scheme
    juce::Slider floorDbSlider;
    // slider to change the norm factor of dB values
    juce::Slider normFactorSlider;

    // slider y axis range
    juce::Slider yRangeSlider;
    juce::TextEditor yMinHzEdit;
    juce::TextEditor yMaxHzEdit;
    double lastYMinHz = 30.0;
    double lastYMaxHz = 20000.0;
    bool   updatingYControls = false;
    // min. y range
    static constexpr double kMinBandWidthHz = 10.0;

    // y axis (note)
    juce::ToggleButton noteAxisToggle;

    // row labels
    float labelAlpha = 0.8f;
    juce::Label row1stLabel;
    juce::Label row2ndLabel;
    juce::Label row3rdLabel;
    
    // other labels
    juce::Label menuDisplayLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramAudioProcessorEditor)
};