/*
  ==============================================================================

    PluginProcessor.cpp
    Handles audio input and forwards data to the spectrogram visualizer.

    Author: hqrrr
    GitHub: https://github.com/hqrrr/PerceptoMap

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
    // try to load global presets from disk
    loadGlobalPresetsFromDisk();
    ensurePresetSlots();

    // current settings: Preset1 is selected by default
    currentSettings = loadPreset(0);
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

        // refresh UI
        juce::MessageManager::callAsync([ed, sampleRate]
        {
            if (ed != nullptr)
                ed->refreshYRangeSliderForSampleRate(sampleRate);
        });
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
    juce::ValueTree state{ "PerceptoMapState" };
    // Save plugin state
    state.setProperty("cur_fpsId", currentSettings.fpsId, nullptr);
    state.setProperty("cur_fftOrderId", currentSettings.fftOrderId, nullptr);
    state.setProperty("cur_overlapId", currentSettings.overlapId, nullptr);
    state.setProperty("cur_modeId", currentSettings.modeId, nullptr);
    state.setProperty("cur_colourId", currentSettings.colourId, nullptr);
    state.setProperty("cur_floorDb", currentSettings.floorDb, nullptr);
    state.setProperty("cur_normFactor", currentSettings.normFactor, nullptr);
    state.setProperty("cur_scrollSpeedId", currentSettings.scrollSpeedId, nullptr);
    state.setProperty("cur_logScaleId", currentSettings.logScaleId, nullptr);
    state.setProperty("cur_yMinHz", currentSettings.yMinHz, nullptr);
    state.setProperty("cur_yMaxHz", currentSettings.yMaxHz, nullptr);
    state.setProperty("cur_noteAxis", currentSettings.noteAxis, nullptr);
    state.setProperty("cur_rolloffR25Visible", currentSettings.rolloffR25Visible, nullptr);
    state.setProperty("cur_rolloffR50Visible", currentSettings.rolloffR50Visible, nullptr);
    state.setProperty("cur_rolloffR85Visible", currentSettings.rolloffR85Visible, nullptr);
    state.setProperty("cur_rolloffR95Visible", currentSettings.rolloffR95Visible, nullptr);

    juce::MemoryOutputStream mos(destData, true);
    state.writeToStream(mos);
}

void SpectrogramAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    // Restore plugin state
    auto vt = juce::ValueTree::readFromData(data, (size_t)sizeInBytes);
    if (!vt.isValid() || !vt.hasType("PerceptoMapState"))
        return;

    currentSettings.fpsId = (int)vt.getProperty("cur_fpsId", currentSettings.fpsId);
    currentSettings.fftOrderId = (int)vt.getProperty("cur_fftOrderId", currentSettings.fftOrderId);
    currentSettings.overlapId = (int)vt.getProperty("cur_overlapId", currentSettings.overlapId);
    currentSettings.modeId = (int)vt.getProperty("cur_modeId", currentSettings.modeId);
    currentSettings.colourId = (int)vt.getProperty("cur_colourId", currentSettings.colourId);
    currentSettings.floorDb = (double)vt.getProperty("cur_floorDb", currentSettings.floorDb);
    currentSettings.normFactor = (double)vt.getProperty("cur_normFactor", currentSettings.normFactor);
    currentSettings.scrollSpeedId = (int)vt.getProperty("cur_scrollSpeedId", currentSettings.scrollSpeedId);
    currentSettings.logScaleId = (int)vt.getProperty("cur_logScaleId", currentSettings.logScaleId);
    currentSettings.yMinHz = (double)vt.getProperty("cur_yMinHz", currentSettings.yMinHz);
    currentSettings.yMaxHz = (double)vt.getProperty("cur_yMaxHz", currentSettings.yMaxHz);
    currentSettings.noteAxis = (bool)vt.getProperty("cur_noteAxis", currentSettings.noteAxis);
    currentSettings.rolloffR25Visible = (bool)vt.getProperty("cur_rolloffR25Visible", currentSettings.rolloffR25Visible);
    currentSettings.rolloffR50Visible = (bool)vt.getProperty("cur_rolloffR50Visible", currentSettings.rolloffR50Visible);
    currentSettings.rolloffR85Visible = (bool)vt.getProperty("cur_rolloffR85Visible", currentSettings.rolloffR85Visible);
    currentSettings.rolloffR95Visible = (bool)vt.getProperty("cur_rolloffR95Visible", currentSettings.rolloffR95Visible);
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SpectrogramAudioProcessor();
}


void SpectrogramAudioProcessor::ensurePresetSlots()
{
    auto presets = globalPresetTree.getChildWithName("Presets");
    if (!presets.isValid())
    {
        presets = juce::ValueTree("Presets");
        globalPresetTree.addChild(presets, -1, nullptr);
    }

    auto makePresetNode = [](const juce::String& name, const PresetData& d)
    {
        juce::ValueTree p{ "Preset" };
        p.setProperty("name", name, nullptr);
        p.setProperty("fpsId", d.fpsId, nullptr);
        p.setProperty("fftOrderId", d.fftOrderId, nullptr);
        p.setProperty("overlapId", d.overlapId, nullptr);
        p.setProperty("modeId", d.modeId, nullptr);
        p.setProperty("colourId", d.colourId, nullptr);
        p.setProperty("floorDb", d.floorDb, nullptr);
        p.setProperty("normFactor", d.normFactor, nullptr);
        p.setProperty("scrollSpeedId", d.scrollSpeedId, nullptr);
        p.setProperty("logScaleId", d.logScaleId, nullptr);
        p.setProperty("yMinHz", d.yMinHz, nullptr);
        p.setProperty("yMaxHz", d.yMaxHz, nullptr);
        p.setProperty("noteAxis", d.noteAxis, nullptr);
        p.setProperty("rolloffR25Visible", d.rolloffR25Visible, nullptr);
        p.setProperty("rolloffR50Visible", d.rolloffR50Visible, nullptr);
        p.setProperty("rolloffR85Visible", d.rolloffR85Visible, nullptr);
        p.setProperty("rolloffR95Visible", d.rolloffR95Visible, nullptr);
        return p;
    };

    PresetData def;

    while (presets.getNumChildren() < kNumPresets)
    {
        const int i = presets.getNumChildren();
        presets.addChild(makePresetNode("Preset" + juce::String(i + 1), def), -1, nullptr);
    }
}

SpectrogramAudioProcessor::PresetData SpectrogramAudioProcessor::loadPreset(int idx) const
{
    PresetData d;
    idx = juce::jlimit(0, kNumPresets - 1, idx);

    auto presets = globalPresetTree.getChildWithName("Presets");
    if (!presets.isValid() || presets.getNumChildren() <= idx)
        return d;

    auto p = presets.getChild(idx);

    d.fpsId = (int)p.getProperty("fpsId", d.fpsId);
    d.fftOrderId = (int)p.getProperty("fftOrderId", d.fftOrderId);
    d.overlapId = (int)p.getProperty("overlapId", d.overlapId);

    d.modeId = (int)p.getProperty("modeId", d.modeId);
    d.colourId = (int)p.getProperty("colourId", d.colourId);

    d.floorDb = (double)p.getProperty("floorDb", d.floorDb);
    d.normFactor = (double)p.getProperty("normFactor", d.normFactor);

    d.scrollSpeedId = (int)p.getProperty("scrollSpeedId", d.scrollSpeedId);
    d.logScaleId = (int)p.getProperty("logScaleId", d.logScaleId);

    d.yMinHz = (double)p.getProperty("yMinHz", d.yMinHz);
    d.yMaxHz = (double)p.getProperty("yMaxHz", d.yMaxHz);

    d.noteAxis = (bool)p.getProperty("noteAxis", d.noteAxis);

    d.rolloffR25Visible = (bool)p.getProperty("rolloffR25Visible", d.rolloffR25Visible);
    d.rolloffR50Visible = (bool)p.getProperty("rolloffR50Visible", d.rolloffR50Visible);
    d.rolloffR85Visible = (bool)p.getProperty("rolloffR85Visible", d.rolloffR85Visible);
    d.rolloffR95Visible = (bool)p.getProperty("rolloffR95Visible", d.rolloffR95Visible);

    return d;
}

void SpectrogramAudioProcessor::savePreset(int idx, const PresetData& d)
{
    idx = juce::jlimit(0, kNumPresets - 1, idx);

    auto presets = globalPresetTree.getChildWithName("Presets");
    if (!presets.isValid() || presets.getNumChildren() <= idx)
        return;

    auto p = presets.getChild(idx);

    p.setProperty("fpsId", d.fpsId, nullptr);
    p.setProperty("fftOrderId", d.fftOrderId, nullptr);
    p.setProperty("overlapId", d.overlapId, nullptr);
    p.setProperty("modeId", d.modeId, nullptr);
    p.setProperty("colourId", d.colourId, nullptr);
    p.setProperty("floorDb", d.floorDb, nullptr);
    p.setProperty("normFactor", d.normFactor, nullptr);
    p.setProperty("scrollSpeedId", d.scrollSpeedId, nullptr);
    p.setProperty("logScaleId", d.logScaleId, nullptr);
    p.setProperty("yMinHz", d.yMinHz, nullptr);
    p.setProperty("yMaxHz", d.yMaxHz, nullptr);
    p.setProperty("noteAxis", d.noteAxis, nullptr);
    p.setProperty("rolloffR25Visible", d.rolloffR25Visible, nullptr);
    p.setProperty("rolloffR50Visible", d.rolloffR50Visible, nullptr);
    p.setProperty("rolloffR85Visible", d.rolloffR85Visible, nullptr);
    p.setProperty("rolloffR95Visible", d.rolloffR95Visible, nullptr);

    saveGlobalPresetsToDisk();
}

static juce::File getGlobalPresetFile()
{
    auto dir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("PerceptoMap");
    dir.createDirectory();
    return dir.getChildFile("PerceptoMapPresets.xml");
}

void SpectrogramAudioProcessor::saveGlobalPresetsToDisk() const
{
    auto file = getGlobalPresetFile();
    if (auto xml = globalPresetTree.createXml())
        xml->writeTo(file);
}

void SpectrogramAudioProcessor::loadGlobalPresetsFromDisk()
{
    auto file = getGlobalPresetFile();
    if (!file.existsAsFile())
        return;

    auto xml = juce::XmlDocument::parse(file);
    if (!xml)
        return;

    auto vt = juce::ValueTree::fromXml(*xml);
    if (vt.isValid() && vt.hasType("PerceptoMapGlobalPresets"))
        globalPresetTree = vt;
}

