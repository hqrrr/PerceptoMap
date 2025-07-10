#pragma once

#include <JuceHeader.h>

class SpectrogramComponent : public juce::Component,
    private juce::Timer
{
public:
    // color scheme for spectrogram
    enum class ColourScheme
    {
        Classic = 1,
        Grayscale,
        Magma
    };
    // spectrogram mode, e.g. linear / mel-scaled ...
    enum class SpectrogramMode
    {
        Linear = 1,
        Mel
        // TODO: add MFCC, Chroma, etc.
    };

    SpectrogramComponent();
    ~SpectrogramComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void pushNextFFTBlock(const float* data, size_t numSamples);
    void setUseLogFrequency(bool shouldUseLog);
    void setSampleRate(double newSampleRate);
    void setGridAlpha(float alpha) { gridAlpha = juce::jlimit(0.0f, 1.0f, alpha); }
    void setColourScheme(ColourScheme scheme);
    void setSpectrogramMode(SpectrogramMode mode);
    void setFrozen(bool shouldFreeze);
    // color map spectrogram
    juce::Colour getColourForValue(float normValue);
    ColourScheme getColourScheme() const { return colourScheme; }

private:
    juce::Image spectrogramImage;

    static constexpr int fftOrder = 11; // 2^11 = 2048
    static constexpr int fftSize = 1 << fftOrder;

    juce::dsp::FFT forwardFFT;
    juce::dsp::WindowingFunction<float> window;

    float fifo[fftSize] = { 0.0f };
    float fftData[fftSize * 2] = { 0.0f };  // doubled for real+imag
    int fifoIndex = 0;

    bool nextFFTBlockReady = false;

    bool useLogFrequency = true;

    double sampleRate = 44100.0;

    bool isFrozen = false;

    // alpha value grid lines
    float gridAlpha = 0.75f;
    // color map spectrogram
    ColourScheme colourScheme = ColourScheme::Classic;

    void timerCallback() override;
    void drawNextLineOfSpectrogram();

    SpectrogramMode currentMode = SpectrogramMode::Linear;

    // mouse tooltip
    std::vector<std::vector<float>> dBBuffer;
    juce::Point<int> mousePosition;
    bool showMouseCrosshair = false;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;

    std::vector<float> melBandFrequencies;
    
};