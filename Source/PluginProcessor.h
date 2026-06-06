/*
  ==============================================================================

    PluginProcessor.h
    Part of the PerceptoMap audio plugin project.

    Author: hqrrr
    GitHub: https://github.com/hqrrr/PerceptoMap

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "SpectrogramComponent.h"

//==============================================================================
/**
    This class implements the audio processing logic of the plugin.
*/
class SpectrogramAudioProcessor : public juce::AudioProcessor
{
public:
    static constexpr int kNumPresets = 3;

    struct PresetData
    {
        int fpsId = 60;
        int fftOrderId = 11;
        int overlapId = 2;

        int modeId = (int)SpectrogramComponent::SpectrogramMode::Linear;
        int colourId = (int)SpectrogramComponent::ColourScheme::Classic;

        double floorDb = -100.0;
        double normFactor = 1.0;

        int scrollSpeedId = 2;
        int logScaleId = 2;

        double yMinHz = 30.0;
        double yMaxHz = 22050.0;

        bool noteAxis = false;
        bool rolloffR25Visible = true;
        bool rolloffR50Visible = true;
        bool rolloffR85Visible = true;
        bool rolloffR95Visible = true;
    };

    //==============================================================================
    SpectrogramAudioProcessor();
    ~SpectrogramAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;


    // current settings
    const PresetData& getCurrentSettings() const { return currentSettings; }
    void setCurrentSettings(const PresetData& d) { currentSettings = d; }

    // global presets
    PresetData loadPreset(int idx) const;
    void savePreset(int idx, const PresetData& d);

private:
    // current settings
    PresetData currentSettings;

    // global presets
    juce::ValueTree globalPresetTree{ "PerceptoMapGlobalPresets" };

    void loadGlobalPresetsFromDisk();
    void saveGlobalPresetsToDisk() const;
    void ensurePresetSlots();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramAudioProcessor)
};
