/*
  ==============================================================================

    SpectrogramComponent.h
    Defines the component responsible for rendering the spectrogram.

    Author: hqrrr
    GitHub: https://github.com/hqrrr/PerceptoMap

  ==============================================================================
*/

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
        GrayscaleEnhanced,
        Magma,
        MagmaEnhanced
    };
    // spectrogram mode, e.g. linear / mel-scaled ...
    enum class SpectrogramMode
    {
        Linear = 1,
        Mel,
        MFCC,
        LinearWithCentroid,
        Chroma
        // TODO: add Tempogram, Rhythm, etc.
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
    // set and get floor value for spectrogram colour scheme
    void setFloorDb(float db) { floorDb = juce::jlimit(-200.0f, -1.0f, db); }
    float getFloorDb() const { return floorDb; }
    // set and get norm factor for spectrogram dB values
    float normFactor = 1.0f;
    void setNormFactor(float f) { normFactor = juce::jlimit(0.001f, 10.0f, f); }
    float getNormFactor() const { return normFactor; }

    // Getter for current drawing mode
    SpectrogramMode getCurrentMode() const { return currentMode; }

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

    // spectrogram lower limit dB value (floor value), default: -100 dB
    float floorDb = -100.0f;
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

    // Mel scale:
    // Number of Mel filterbanks used to approximate the auditory scale.
    const int melBands = 128;
    // MFCC:
    // Number of MFCC coefficients to extract after DCT.
    // Only the lowest `numCoeffs` (typically 13–20) are retained,
    // matching librosa's `n_mfcc=20` default.
    // These are what we visualize along the vertical axis of the MFCC image.
    const int numCoeffs = 20;
    // The number of MFCC labels shown, must be less than or equal to numCoeffs.
    const int numLabels = 10;
    // MFCC value range;
    static constexpr float mfccMin = -100.0f;
    static constexpr float mfccMax = 100.0f;

    // Spectral centroid
    float computeSpectralCentroid(const float* magnitude, int numBins) const;
    int lastCentroidY = -1;
    // Exponential Moving Average to smooth the centroid curve
    float centroidSmoothed = 0.0f;
    bool hasPreviousCentroid = false;
    // Values of close to 1 have less of a smoothing effect and give greater weight to recent changes in the data, 
    // while values of closer to 0 have a greater smoothing effect and are less responsive to recent changes.
    const float smoothingFactor = 0.3f;

    // Chromagram
    const int numChroma = 12;
    const char* pitchNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
};

// color schemes
namespace ColourMaps
{
    static const juce::Colour magmaColors[10] = {
            juce::Colour::fromRGB(0, 0, 4),          // #000004
            juce::Colour::fromRGB(24, 15, 61),       // #180f3d
            juce::Colour::fromRGB(68, 15, 118),      // #440f76
            juce::Colour::fromRGB(114, 31, 129),     // #721f81
            juce::Colour::fromRGB(158, 47, 127),     // #9e2f7f
            juce::Colour::fromRGB(205, 64, 113),     // #cd4071
            juce::Colour::fromRGB(241, 96, 93),      // #f1605d
            juce::Colour::fromRGB(253, 150, 104),    // #fd9668
            juce::Colour::fromRGB(254, 202, 141),    // #feca8d
            juce::Colour::fromRGB(252, 253, 191)     // #fcfdbf
    };
}