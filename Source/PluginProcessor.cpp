/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
SpectrogramAudioProcessor::SpectrogramAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(BusesProperties()
#if !JucePlugin_IsMidiEffect
#if !JucePlugin_IsSynth
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
#endif
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)
#endif
    )
#endif
{
}

SpectrogramAudioProcessor::~SpectrogramAudioProcessor()
{
}

//==============================================================================
const juce::String SpectrogramAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool SpectrogramAudioProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool SpectrogramAudioProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool SpectrogramAudioProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double SpectrogramAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

//==============================================================================
int SpectrogramAudioProcessor::getNumPrograms() { return 1; }
int SpectrogramAudioProcessor::getCurrentProgram() { return 0; }
void SpectrogramAudioProcessor::setCurrentProgram(int) {}
const juce::String SpectrogramAudioProcessor::getProgramName(int) { return {}; }
void SpectrogramAudioProcessor::changeProgramName(int, const juce::String&) {}

//==============================================================================
void SpectrogramAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    if (auto* ed = dynamic_cast<SpectrogramAudioProcessorEditor*>(getActiveEditor()))
    {
        ed->getSpectrogramComponent().setSampleRate(sampleRate);
    }
}

void SpectrogramAudioProcessor::releaseResources()
{
    // Free any resources if needed
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool SpectrogramAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
#else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

#if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
#endif
}
#endif

void SpectrogramAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear output channels that aren't used
    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Get pointer to left channel (or mono)
    auto* channelData = buffer.getReadPointer(0);

    // Forward audio data to spectrogram component in editor
    if (auto* ed = dynamic_cast<SpectrogramAudioProcessorEditor*>(getActiveEditor()))
    {
        ed->getSpectrogramComponent().pushNextFFTBlock(channelData, buffer.getNumSamples());
    }

    // Pass-through processing (no modification of audio)
}

//==============================================================================
bool SpectrogramAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* SpectrogramAudioProcessor::createEditor()
{
    return new SpectrogramAudioProcessorEditor(*this);
}

//==============================================================================
void SpectrogramAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    // Save plugin state (e.g. parameters, settings) here if needed
}

void SpectrogramAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    // Restore plugin state here if needed
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SpectrogramAudioProcessor();
}