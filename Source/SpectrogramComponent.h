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
        Chroma,
        LinearPlus,
        MelPlus
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

    // set UI refresh rate (frames per second)
    void setUiFps(int fps);
    // change FFT order at runtime
    void setFFTOrder(int newOrder);
    // change FFT overlap at runtime
    void setOverlap(int newOverlap);
    // scroll speed
    void setScrollSpeedMultiplier(int mulX);
    float getScrollPixelsPerSecond() const { return pixelsPerSecond; }

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

    // y axis frequency range
    void setFrequencyRangeHz(float minHz, float maxHz)
    {
        const float nyquist = static_cast<float>(sampleRate) * 0.5f;
        minHz = juce::jlimit(1.0f, nyquist - 1.0f, minHz);
        maxHz = juce::jlimit(minHz + 1.0f, nyquist, maxHz);
        minFreqHz = minHz;
        maxFreqHz = maxHz;
        repaint();
    }
    std::pair<float, float> getFrequencyRangeHz() const { return { minFreqHz, maxFreqHz }; }

private:
    juce::Image spectrogramImage;

    // y axis frequency range
    float minFreqHz = 30.0f;
    float maxFreqHz = 22050.0f;

    // FFT state
    int fftOrder = 11;  // default 2^11 = 2048
    int fftSize = 1 << fftOrder;

    juce::dsp::FFT forwardFFT{fftOrder};
    std::unique_ptr<juce::dsp::WindowingFunction<float>> window;

    std::vector<float> fifo;    // size = fftSize
    std::vector<float> fftData; // size = fftSize * 2
    int fifoIndex = 0;

    // FFT settings
    int overlap = 2;           // default: 2x overlap
    int hopSize = 0;
    int samplesSinceHop = 0;
    std::vector<float> ring;   // size = fftSize
    size_t ringWrite = 0;
    

    bool nextFFTBlockReady = false;

    bool useLogFrequency = true;

    double sampleRate = 44100.0;

    // scrolling
    float pixelsPerSecond = 0.0f;
    float pixelAccum = 0.0f;
    int   uiFps = 60;
    std::vector<int> imgColAge;
    float baseScrollPps = 20.0f;  // base speed: 20 px/s
    int   scrollSpeedMul = 2;   // default: 2x scroll speed

    bool isFrozen = false;

    // spectrogram lower limit dB value (floor value), default: -100 dB
    float floorDb = -100.0f;
    // alpha value grid lines
    float gridAlpha = 0.75f;
    // color map spectrogram
    ColourScheme colourScheme = ColourScheme::Classic;

    void timerCallback() override;
    void drawNextLineOfSpectrogram();

    void drawLinearSpectrogram(int x, std::vector<float>& dBColumn, const int imageHeight, const float maxFreq);
    void drawMelSpectrogram(int x, std::vector<float>& dBColumn, const int imageHeight);
    void drawMFCC(int x, std::vector<float>& dBColumn, const int imageHeight);
    void drawLinearWithCentroid(int x, std::vector<float>& dBColumn, const int imageHeight, const float maxFreq);
    void drawChroma(int x, std::vector<float>& dBColumn, const int imageHeight);
    void drawReassignedSpectrogram(int x, std::vector<float>& dBColumn, const int imageHeight, const float maxFreq);
    void drawReassignedMelSpectrogram(int x, std::vector<float>& dBColumn, int imageHeight);

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
    void buildChromaFilterBank(int fftSize, double sampleRate);
    std::vector<std::vector<float>> chromaFilterBank;
    const int numChroma = 12;
    const char* pitchNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

    // Time-Frequency reassigned STFT spectrogram
    // adjust to control density
    const float thresholdFactor = 0.05f;
    // Gaussian window width
    const float sigmaCoef = 0.3f;
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