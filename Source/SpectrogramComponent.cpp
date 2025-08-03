/*
  ==============================================================================

    SpectrogramComponent.cpp
    Implements STFT processing, visualization logic, and user interaction.

    Author: hqrrr
    GitHub: https://github.com/hqrrr/PerceptoMap

  ==============================================================================
*/

#include "SpectrogramComponent.h"

SpectrogramComponent::SpectrogramComponent()
    : forwardFFT(fftOrder),
    window(fftSize, juce::dsp::WindowingFunction<float>::hann)
{
    spectrogramImage = juce::Image(juce::Image::RGB, fftSize, fftSize / 2, true);
    // 60 FPS
    startTimerHz(60); 
    // mouse listener
    addMouseListener(this, true);
}

void SpectrogramComponent::timerCallback()
{
    if (isFrozen || !nextFFTBlockReady)
        return;

    drawNextLineOfSpectrogram();
    nextFFTBlockReady = false;
    //repaint(getWidth() - 1, 0, 1, getHeight());
    repaint();
}

void SpectrogramComponent::pushNextFFTBlock(const float* data, size_t numSamples)
{
    for (size_t i = 0; i < numSamples; ++i)
    {
        if (nextFFTBlockReady)
            return;

        fifo[fifoIndex++] = data[i];

        if (fifoIndex == fftSize)
        {
            std::copy(std::begin(fifo), std::end(fifo), fftData);
            window.multiplyWithWindowingTable(fftData, fftSize);

            forwardFFT.performFrequencyOnlyForwardTransform(fftData);

            nextFFTBlockReady = true;
            fifoIndex = 0;
        }
    }
}

void SpectrogramComponent::setSampleRate(double newSampleRate)
{
    sampleRate = newSampleRate;
    // init filter bank for chromagram
    buildChromaFilterBank(fftSize, sampleRate);
}

void SpectrogramComponent::setUseLogFrequency(bool shouldUseLog)
{
    useLogFrequency = shouldUseLog;
}

void SpectrogramComponent::setColourScheme(ColourScheme scheme)
{
    colourScheme = scheme;
}

void SpectrogramComponent::setSpectrogramMode(SpectrogramMode mode)
{
    currentMode = mode;
}

void SpectrogramComponent::setFrozen(bool shouldFreeze)
{
    isFrozen = shouldFreeze;
}

void SpectrogramComponent::mouseMove(const juce::MouseEvent& event)
{
    mousePosition = event.getPosition();
    showMouseCrosshair = true;
    repaint();
}

void SpectrogramComponent::mouseExit(const juce::MouseEvent&)
{
    showMouseCrosshair = false;
    repaint();
}

void SpectrogramComponent::drawLinearSpectrogram(int x, std::vector<float>& dBColumn, const int imageHeight, const float maxFreq)
{
    // default: linear STFT spectrogram
    for (int y = 0; y < imageHeight; ++y)
    {
        int binIndex = 0;

        if (useLogFrequency)
        {
            float logMinFreq = std::log10(30.0f); // lower bound (must match label start)
            float logMaxFreq = std::log10(maxFreq);
            float frac = 1.0f - static_cast<float>(y) / imageHeight; // low -> high

            float logFreq = logMinFreq + frac * (logMaxFreq - logMinFreq);
            float freq = std::pow(10.0f, logFreq);
            binIndex = static_cast<int>((freq / maxFreq) * (fftSize / 2));
        }
        else
        {
            float frac = 1.0f - static_cast<float>(y) / imageHeight;
            binIndex = static_cast<int>(frac * (fftSize / 2));
        }

        binIndex = std::clamp(binIndex, 0, fftSize / 2 - 1);

        // convert magnitude to dB
        float magnitude = fftData[binIndex] * normFactor;
        float dB = 20.0f * std::log10(magnitude + 1e-6f); // avoid log(0)
        float clippedDB = juce::jlimit(floorDb, 0.0f, dB);
        dBColumn[y] = clippedDB;
        float brightness = juce::jmap(clippedDB, floorDb, 0.0f, 0.0f, 1.0f);

        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
    }
}

void SpectrogramComponent::drawMelSpectrogram(int x, std::vector<float>& dBColumn, const int imageHeight)
{
    // melspectrogram
    std::vector<float> melEnergies(melBands, 0.0f);

    const float minHz = 0.0f;
    const float maxHz = static_cast<float>(sampleRate) / 2.0f;

    // Slaney-style Mel: mel = 2595 * log10(1 + f / 700)
    auto hzToMel = [](float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
        };

    auto melToHz = [](float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
        };

    float melMin = hzToMel(minHz);
    float melMax = hzToMel(maxHz);

    // 128 Mel bands
    std::vector<float> melCenterFreqs(melBands);
    melBandFrequencies.resize(melBands);
    for (int m = 0; m < melBands; ++m)
    {
        float mel = melMin + (melMax - melMin) * m / (melBands - 1);
        float freq = melToHz(mel);
        melBandFrequencies[m] = freq;
        int bin = static_cast<int>((freq / maxHz) * (fftSize / 2));
        bin = std::clamp(bin, 0, fftSize / 2 - 1);
        // Simplified approximation: instead of applying a weighted mel filter,
        // directly sample the FFT magnitude at the center frequency of each mel band.
        // fast but less accurate than full mel filterbank convolution (as in librosa) but acceptable for visualization
        melEnergies[m] = fftData[bin];
    }

    for (int y = 0; y < imageHeight; ++y)
    {
        int melIndex = juce::jmap(y, 0, imageHeight - 1, melBands - 1, 0);
        melIndex = std::clamp(melIndex, 0, melBands - 1);

        float scaled = melEnergies[melIndex] * normFactor;
        float dB = 20.0f * std::log10(scaled + 1e-6f);
        float clippedDB = juce::jlimit(floorDb, 0.0f, dB);
        dBColumn[y] = clippedDB;
        float brightness = juce::jmap(clippedDB, floorDb, 0.0f, 0.0f, 1.0f);

        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
    }
}

void SpectrogramComponent::drawMFCC(int x, std::vector<float>& dBColumn, const int imageHeight)
{
    // Mel-frequency cepstral coefficient (MFCC)
    std::vector<float> melEnergies(melBands, 0.0f);
    const float maxHz = static_cast<float>(sampleRate) / 2.0f;
    // Slaney-style mel scale
    auto hzToMel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto melToHz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

    float melMin = hzToMel(0.0f), melMax = hzToMel(maxHz);

    // Calculate the center frequency of the mel filter and take the fft bin
    for (int m = 0; m < melBands; ++m)
    {
        float mel = melMin + (melMax - melMin) * m / (melBands - 1);
        float freq = melToHz(mel);
        int bin = static_cast<int>((freq / maxHz) * (fftSize / 2));
        bin = std::clamp(bin, 0, fftSize / 2 - 1);
        // Simplified approximation: instead of applying a weighted mel filter,
        // directly sample the FFT magnitude at the center frequency of each mel band.
        // fast but less accurate than full mel filterbank convolution (as in librosa) but acceptable for visualization
        melEnergies[m] = fftData[bin];
    }

    // Perform log on mel energies.
    for (auto& e : melEnergies)
    {
        e = std::log(e + 1e-6f);
    }

    // Simple DCT implementation (Type-II)
    std::vector<float> mfcc(numCoeffs, 0.0f);
    for (int k = 0; k < numCoeffs; ++k)
    {
        float sum = 0.0f;
        for (int n = 0; n < melBands; ++n)
            sum += melEnergies[n] * std::cos(juce::MathConstants<float>::pi * k * (n + 0.5f) / melBands);
        mfcc[k] = sum;
    }

    // Map MFCC to image columns.
    for (int y = 0; y < imageHeight; ++y)
    {
        //int coeffIndex = juce::jmap(y, 0, imageHeight - 1, numCoeffs - 1, 0);
        float frac = static_cast<float>(y) / (imageHeight - 1);
        int coeffIndex = static_cast<int>((1.0f - frac) * (numCoeffs - 1));
        coeffIndex = std::clamp(coeffIndex, 0, numCoeffs - 1);

        float value = mfcc[coeffIndex];
        // limit value in range [mfccMin, mfccMax]
        float norm = juce::jlimit(mfccMin, mfccMax, value);
        // 0 - 1 normalization
        float brightness = juce::jmap(norm, mfccMin, mfccMax, 0.0f, 1.0f) * normFactor;

        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
        dBColumn[y] = value;
    }
}

void SpectrogramComponent::drawLinearWithCentroid(int x, std::vector<float>& dBColumn, const int imageHeight, const float maxFreq)
{
    // Spectral Centroid
    float rawCentroidHz = computeSpectralCentroid(fftData, fftSize / 2);

    if (!hasPreviousCentroid || rawCentroidHz <= 0.0f || rawCentroidHz > maxFreq)
    {
        centroidSmoothed = rawCentroidHz;
        hasPreviousCentroid = true;
    }
    else
    {
        // Exponential Moving Average
        centroidSmoothed += smoothingFactor * (rawCentroidHz - centroidSmoothed);
    }

    for (int y = 0; y < imageHeight; ++y)
    {
        int binIndex = 0;

        if (useLogFrequency)
        {
            float logMinFreq = std::log10(30.0f);
            float logMaxFreq = std::log10(maxFreq);
            float frac = 1.0f - static_cast<float>(y) / imageHeight;

            float logFreq = logMinFreq + frac * (logMaxFreq - logMinFreq);
            float freq = std::pow(10.0f, logFreq);
            binIndex = static_cast<int>((freq / maxFreq) * (fftSize / 2));
        }
        else
        {
            float frac = 1.0f - static_cast<float>(y) / imageHeight;
            binIndex = static_cast<int>(frac * (fftSize / 2));
        }

        binIndex = std::clamp(binIndex, 0, fftSize / 2 - 1);

        float magnitude = fftData[binIndex] * normFactor;
        float dB = 20.0f * std::log10(magnitude + 1e-6f);
        float clippedDB = juce::jlimit(floorDb, 0.0f, dB);
        dBColumn[y] = clippedDB;
        float brightness = juce::jmap(clippedDB, floorDb, 0.0f, 0.0f, 1.0f);

        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
    }

    // Draw spectral centroid
    auto mapHzToY = [imageHeight = this->getHeight(), this, maxFreq](float freqHz) -> int
        {
            if (useLogFrequency)
            {
                float logMinFreq = std::log10(30.0f);
                float logMaxFreq = std::log10(maxFreq);
                float logFreq = std::log10(freqHz);
                float normY = 1.0f - (logFreq - logMinFreq) / (logMaxFreq - logMinFreq);
                return juce::jlimit(0, imageHeight - 1, static_cast<int>(normY * imageHeight));
            }
            else
            {
                float normY = 1.0f - freqHz / maxFreq;
                return juce::jlimit(0, imageHeight - 1, static_cast<int>(normY * imageHeight));
            }
        };

    int centroidY = mapHzToY(centroidSmoothed);

    // Define highlight colors
    juce::Colour centroidColor;

    switch (colourScheme)
    {
    case ColourScheme::Grayscale:
        centroidColor = juce::Colour::fromRGB(0, 200, 255);   // blue
        break;
    case ColourScheme::GrayscaleEnhanced:
        centroidColor = juce::Colour::fromRGB(0, 200, 255);   // blue
        break;
    case ColourScheme::Magma:
        centroidColor = juce::Colour::fromRGB(0, 255, 128);   // light green
        break;
    case ColourScheme::MagmaEnhanced:
        centroidColor = juce::Colour::fromRGB(0, 255, 128);   // light green
        break;

    default:
        centroidColor = juce::Colour::fromRGB(255, 255, 255);   // white
        break;
    }

    // Draw centroid as thick line
    if (lastCentroidY >= 0)
    {
        juce::Graphics g(spectrogramImage);
        g.setColour(centroidColor);
        g.drawLine((float)(x - 1), (float)lastCentroidY,
            (float)x, (float)centroidY,
            2.0f);  // line width
    }
    else
    {
        // If it is the first centroid and there is no previous value, draw a small vertical line.
        for (int dy = -1; dy <= 1; ++dy)
        {
            int y = juce::jlimit(0, imageHeight - 1, centroidY + dy);
            spectrogramImage.setPixelAt(x, y, centroidColor);
        }
    }

    lastCentroidY = centroidY;
}

void SpectrogramComponent::drawChroma(int x, std::vector<float>& dBColumn, const int imageHeight)
{
    // Chromagram
    std::vector<float> chroma(numChroma, 0.0f);
    const float binFreqResolution = sampleRate / fftSize;

    for (int bin = 1; bin < fftSize / 2; ++bin)
    {
        float freq = bin * binFreqResolution;
        if (freq < 20.0f || freq > 5000.0f) continue;

        float magnitude = fftData[bin];
        float dB = 20.0f * std::log10(magnitude + 1e-6f);

        // Simplified implementation: Allocate the energy of each FFT bin completely to a 
        // single pitch class (i.e., the frequency corresponding to the bin -> MIDI note -> pitch class), 
        // ignoring the actual frequency uncertainty and frequency distribution continuity.
        //int midiNote = static_cast<int>(std::round(69.0 + 12.0 * std::log2(freq / 440.0)));
        //int pitchClass = ((midiNote % 12) + 12) % 12;
        //chroma[pitchClass] += std::pow(10.0f, dB / 10.0f);  // accumulate energy

        // More accurate implementation: Each FFT bin's linear energy is distributed across all 12 pitch classes
        // using a chroma filter bank (precomputed based on bin frequencies and pitch class proximity).
        float linearEnergy = std::pow(10.0f, dB / 10.0f);
        for (int pc = 0; pc < 12; ++pc)
        {
            chroma[pc] += chromaFilterBank[pc][bin] * linearEnergy;
        }
    }

    // Normalize chroma vector to [0, 1]
    float sum = std::accumulate(chroma.begin(), chroma.end(), 0.0f);
    if (sum > 1e-6f)
        for (auto& val : chroma)
            val /= sum;

    const float blockHeight = static_cast<float>(imageHeight) / numChroma;

    for (int y = 0; y < imageHeight; ++y)
    {
        //int chromaIndex = juce::jmap(y, 0, imageHeight - 1, numChroma - 1, 0);
        int chromaIndex = static_cast<int>((imageHeight - 1 - y) / blockHeight);
        chromaIndex = juce::jlimit(0, numChroma - 1, chromaIndex);

        float brightness = juce::jlimit(0.0f, 1.0f, chroma[chromaIndex] * normFactor);
        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
        dBColumn[y] = brightness;
    }
}

void SpectrogramComponent::drawNextLineOfSpectrogram()
{
    const int imageWidth = spectrogramImage.getWidth();
    const int imageHeight = spectrogramImage.getHeight();
    std::vector<float> dBColumn(imageHeight);

    // Scroll the image left by one pixel
    spectrogramImage.moveImageSection(0, 0, 1, 0, imageWidth - 1, imageHeight);
    const int x = imageWidth - 1; // rightmost column

    const float maxFreq = sampleRate / 2.0f;

    switch (currentMode)
    {
    case SpectrogramMode::Mel:
        SpectrogramComponent::drawMelSpectrogram(x, dBColumn, imageHeight);
        break;
    case SpectrogramMode::MFCC:
        SpectrogramComponent::drawMFCC(x, dBColumn, imageHeight);
        break;
    case SpectrogramMode::LinearWithCentroid:
        SpectrogramComponent::drawLinearWithCentroid(x, dBColumn, imageHeight, maxFreq);
        break;
    case SpectrogramMode::Chroma:
        SpectrogramComponent::drawChroma(x, dBColumn, imageHeight);
        break;
    default:
        SpectrogramComponent::drawLinearSpectrogram(x, dBColumn, imageHeight, maxFreq);
        break;
    }

    if (dBBuffer.size() >= imageWidth)
        dBBuffer.erase(dBBuffer.begin());

    dBBuffer.push_back(dBColumn);
}

void SpectrogramComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    // draw spectrogram
    g.drawImage(spectrogramImage, getLocalBounds().toFloat());

    // font size
    g.setFont(12.0f);

    const int width = getWidth();
    const int height = getHeight();
    const float maxFreq = sampleRate / 2.0f;
    const int imageHeight = spectrogramImage.getHeight();

    // draw y axis (frequency)
    if (currentMode == SpectrogramMode::Mel)
    {
        // Mel scale tick positions (Slaney-style spacing, approximate)
        const int melBands = imageHeight;
        const int numLabels = 10;

        for (int i = 0; i < numLabels; ++i)
        {
            float mel = (static_cast<float>(i) / (numLabels - 1)) * 2595.0f * std::log10(1 + maxFreq / 700.0f);
            float freq = 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);

            float yNorm = static_cast<float>(i) / (numLabels - 1);
            int y = imageHeight - 1 - static_cast<int>(yNorm * imageHeight);

            juce::String label = freq >= 1000.0f ? juce::String(freq / 1000.0f, 1) + " kHz"
                : juce::String(static_cast<int>(freq)) + " Hz";

            // draw label box with dark background
            juce::Rectangle<int> textBounds(2, y - 12, 60, 16);
            g.setColour(juce::Colours::black.withAlpha(0.6f));
            g.fillRect(textBounds);
            // draw label
            g.setColour(juce::Colours::white);
            g.drawText(label, textBounds, juce::Justification::left);
            // draw grid line
            g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
            g.drawHorizontalLine(y, 55.0f, static_cast<float>(width));
        }
    }
    else if (currentMode == SpectrogramMode::MFCC)
    {
        // MFCC labels
        for (int i = 0; i < numLabels; ++i)
        {
            float frac = static_cast<float>(i) / (numLabels - 1);
            int coeffIndex = static_cast<int>(frac * (numCoeffs - 1));
            int y = static_cast<int>((1.0f - frac) * (imageHeight - 1));

            // Draw label
            juce::Rectangle<int> textBounds(2, y - 8, 60, 16);
            g.setColour(juce::Colours::black.withAlpha(0.6f));
            g.fillRect(textBounds);
            g.setColour(juce::Colours::white);
            g.drawText("MFCC " + juce::String(coeffIndex), textBounds, juce::Justification::left);

            // Grid line
            g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
            g.drawHorizontalLine(y, 55.0f, static_cast<float>(width));
        }
    }
    else if (currentMode == SpectrogramMode::Chroma)
    {
        const float blockHeight = static_cast<float>(imageHeight) / numChroma;

        for (int i = 0; i < 12; ++i)
        {
            //float normY = static_cast<float>(i) / 11.0f;
            int y = static_cast<int>((numChroma - 1 - i) * blockHeight);

            juce::Rectangle<int> textBounds(2, y - 8, 60, 16);
            g.setColour(juce::Colours::black.withAlpha(0.6f));
            g.fillRect(textBounds);
            g.setColour(juce::Colours::white);
            g.drawText(pitchNames[i], textBounds, juce::Justification::left);

            g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
            g.drawHorizontalLine(y, 55.0f, static_cast<float>(width));
        }
    }
    else
    {
        // log y axis or linear, for linear STFT spectrogram & with spectral centroid + bandwidth
        if (useLogFrequency)
        {
            std::vector<float> freqsToLabel = { 30, 64, 128, 256, 512, 1024, 2048, 4096,
                                                8192, 16384, 32768 };

            float logMinFreq = std::log10(freqsToLabel.front());
            float logMaxFreq = std::log10(maxFreq);

            for (float freq : freqsToLabel)
            {
                if (freq >= maxFreq)
                    break;

                float logFreq = std::log10(freq);
                float yNorm = (logFreq - logMinFreq) / (logMaxFreq - logMinFreq);
                int y = imageHeight - 1 - static_cast<int>(yNorm * imageHeight);

                // draw label box with dark background
                juce::Rectangle<int> textBounds(2, y - 12, 60, 16);
                g.setColour(juce::Colours::black.withAlpha(0.6f));
                g.fillRect(textBounds);
                // draw label
                g.setColour(juce::Colours::white);
                g.drawText(juce::String(static_cast<int>(freq)) + " Hz",
                    textBounds, juce::Justification::left);
                // draw grid line
                g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
                g.drawHorizontalLine(y, 55.0f, static_cast<float>(width));
            }
        }
        else
        {
            const int numFreqLabels = 6;

            for (int i = 0; i < numFreqLabels; ++i)
            {
                float normY = static_cast<float>(i) / (numFreqLabels - 1);
                //int y = static_cast<int>(normY * imageHeight);
                int y = imageHeight - 1 - static_cast<int>(normY * imageHeight);

                float freq = normY * (sampleRate / 2.0f);

                juce::String freqLabel = juce::String(freq / 1000.0f, 1) + " kHz";

                // draw label box with dark background
                juce::Rectangle<int> textBounds(2, y - 12, 60, 16);
                g.setColour(juce::Colours::black.withAlpha(0.6f));
                g.fillRect(textBounds);
                // draw label
                g.setColour(juce::Colours::white);
                g.drawText(freqLabel, textBounds, juce::Justification::left);
                // draw grid line
                g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
                g.drawHorizontalLine(y, 55.0f, static_cast<float>(width));
            }

        }
    }

    // draw x axis (time)
    const int numTimeLabels = 8;
    for (int i = 0; i < numTimeLabels; ++i)
    {
        float frac = (float)i / (numTimeLabels - 1);        // 0 ~ 1
        int x = getWidth() - static_cast<int>(frac * getWidth());

        g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
        g.drawVerticalLine(x, 0.0f, static_cast<float>(getHeight() - 20));
    }

    // draw mouse crosshair and tooltip
    if (showMouseCrosshair && getLocalBounds().contains(mousePosition))
    {
        g.setColour(juce::Colours::white.withAlpha(0.4f));
        g.drawLine(0, mousePosition.y, (float)getWidth(), mousePosition.y, 1.0f);
        g.drawLine(mousePosition.x, 0, mousePosition.x, (float)getHeight(), 1.0f);

        int x = mousePosition.x;
        int y = mousePosition.y;

        // convert GUI coords -> spectrogram image coords
        auto bounds = getLocalBounds();

        const int imgWidth = spectrogramImage.getWidth();
        const int imgHeight = spectrogramImage.getHeight();

        // X
        int imgX = juce::jlimit(0, imgWidth - 1,
            imgWidth - 1 -
            ((mousePosition.x - bounds.getX()) * imgWidth / bounds.getWidth()));

        // Y
        int imgY = juce::jlimit(0, imgHeight - 1,
            (mousePosition.y - bounds.getY()) * imgHeight / bounds.getHeight());

        int dBIndex = (int)dBBuffer.size() - 1 - imgX;
        if (imgX >= 0 && imgX < (int)dBBuffer.size() &&
            imgY >= 0 && imgY < (int)dBBuffer[imgX].size())
        {
            float dB = dBBuffer[dBIndex][imgY];
            float maxFreq = sampleRate / 2.0f;
            float freq = 0.0f;

            juce::String labelText;

            if (currentMode == SpectrogramMode::Mel)
            {
                // melspectrogram
                int melIndex = juce::jlimit(0, (int)melBandFrequencies.size() - 1,
                    (int)((float)imgY / getHeight() * melBandFrequencies.size()));
                freq = melBandFrequencies[melBandFrequencies.size() - 1 - melIndex];
                // get midi note name
                juce::String noteName;
                if (freq >= 20.0f && freq <= 20000.0f)
                {
                    int midiNote = (int)std::round(69 + 12 * std::log2(freq / 440.0f));
                    if (midiNote >= 0 && midiNote <= 127)
                    {
                        // octave for middle C: C4
                        noteName = juce::MidiMessage::getMidiNoteName(midiNote, true, true, 4);
                    }
                    else
                    {
                        noteName = "(out of range)";
                    }
                }
                else
                {
                    noteName = "(out of range)";
                }
                // generate tooltip text
                labelText = juce::String(freq, 1) + " Hz, " + (noteName.isNotEmpty() ? "note: " + noteName : "") + ", " + juce::String(dB, 1) + " dB";
            }
            else if (currentMode == SpectrogramMode::MFCC)
            {
                // MFCC
                float frac = static_cast<float>(imgY) / (imageHeight - 1);
                int coeffIndex = static_cast<int>((1.0f - frac) * (numCoeffs - 1));
                coeffIndex = juce::jlimit(0, numCoeffs - 1, coeffIndex);

                // Note: dB is actually MFCC (unitless), not decibel
                float norm = juce::jlimit(mfccMin, mfccMax, dB);
                float brightness = juce::jmap(norm, mfccMin, mfccMax, 0.0f, 1.0f);

                labelText = "MFCC " + juce::String(coeffIndex) + ", " + juce::String(brightness, 2) + " (normalized)";
            }
            else if (currentMode == SpectrogramMode::Chroma)
            {
                // Chromagram
                float blockHeight = static_cast<float>(imageHeight) / numChroma;
                int chromaIndex = static_cast<int>((imageHeight - 1 - imgY) / blockHeight);
                chromaIndex = juce::jlimit(0, numChroma - 1, chromaIndex);
                juce::String pitchName = pitchNames[chromaIndex];

                float brightness = juce::jlimit(0.0f, 1.0f, dBBuffer[dBIndex][imgY]);
                //float brightness = juce::jlimit(0.0f, 1.0f, dBBuffer[dBIndex][chromaIndex]);

                labelText = "Pitch Class: " + pitchName + ", " + juce::String(brightness, 2);
            }
            else
            {
                // STFT spectrogram & with spectral centroid + bandwidth
                if (useLogFrequency)
                {
                    float logMinFreq = std::log10(30.0f);
                    float logMaxFreq = std::log10(maxFreq);
                    float logFreq = logMinFreq + (1.0f - (float)imgY / getHeight()) * (logMaxFreq - logMinFreq);
                    freq = std::pow(10.0f, logFreq);
                }
                else
                {
                    freq = (1.0f - (float)imgY / getHeight()) * maxFreq;
                }
                // get midi note name
                juce::String noteName;
                if (freq >= 20.0f && freq <= 20000.0f)
                {
                    int midiNote = (int)std::round(69 + 12 * std::log2(freq / 440.0f));
                    if (midiNote >= 0 && midiNote <= 127)
                    {
                        // octave for middle C: C4
                        noteName = juce::MidiMessage::getMidiNoteName(midiNote, true, true, 4);
                    }
                    else
                    {
                        noteName = "(out of range)";
                    }
                }
                else
                {
                    noteName = "(out of range)";
                }
                // generate tooltip text
                labelText = juce::String(freq, 1) + " Hz, " + (noteName.isNotEmpty() ? "note: " + noteName : "") + ", " + juce::String(dB, 1) + " dB";
            }

            // Draw fixed box under legend bar (top right)
            const int boxWidth = 240;
            const int boxHeight = 20;
            const int padding = 5;

            int boxX = getWidth() - boxWidth - padding;
            int boxY = padding;  // below legend

            g.setColour(juce::Colours::black.withAlpha(0.6f));
            g.fillRect(boxX, boxY, boxWidth, boxHeight);

            g.setColour(juce::Colours::white);
            g.setFont(12.0f);
            g.drawText(labelText, boxX, boxY, boxWidth, boxHeight, juce::Justification::centredLeft);
        }
    }

}

// Return the center frequency in Hz units.
float SpectrogramComponent::computeSpectralCentroid(const float* magnitude, int numBins) const
{
    float weightedSum = 0.0f;
    float energySum = 0.0f;
    float freqSquaredSum = 0.0f;

    for (int i = 0; i < numBins; ++i)
    {
        float mag = magnitude[i];
        float power = mag * mag;
        float binWidth = sampleRate / fftSize;
        float freq = i * binWidth;
        weightedSum += freq * power;
        energySum += power;
        freqSquaredSum += freq * freq * power;
    }
    // Avoid numerical instability at extremely low signal energies (e.g., silent passages).
    if (energySum < 1e-6f)
        return 0.0f;

    float centroid = (energySum > 0.0f) ? weightedSum / energySum : 0.0f;

    return centroid;
}

// Chromagram filter bank
void SpectrogramComponent::buildChromaFilterBank(int fftSize, double sampleRate)
{
    const int numBins = fftSize / 2;
    chromaFilterBank.assign(12, std::vector<float>(numBins, 0.0f));

    const float binFreqResolution = static_cast<float>(sampleRate) / fftSize;
    const float tuningHz = 440.0f;

    for (int bin = 0; bin < numBins; ++bin)
    {
        float freq = bin * binFreqResolution;
        if (freq < 20.0f || freq > 5000.0f)
            continue;

        float midiNote = 69.0f + 12.0f * std::log2(freq / tuningHz);
        float pitchClassFloat = std::fmod(midiNote, 12.0f);
        if (pitchClassFloat < 0.0f)
            pitchClassFloat += 12.0f;

        // Distribute to neighboring pitch classes with Gaussian weights
        for (int pc = 0; pc < 12; ++pc)
        {
            float distance = std::min(std::abs(pitchClassFloat - pc), 12.0f - std::abs(pitchClassFloat - pc));
            float weight = std::exp(-0.5f * std::pow(distance / 1.0f, 2)); // sigma=1.0
            chromaFilterBank[pc][bin] = weight;
        }
    }

    // Normalize each bin's contribution to 1
    for (int bin = 0; bin < numBins; ++bin)
    {
        float sum = 0.0f;
        for (int pc = 0; pc < 12; ++pc)
            sum += chromaFilterBank[pc][bin];

        if (sum > 1e-6f)
            for (int pc = 0; pc < 12; ++pc)
                chromaFilterBank[pc][bin] /= sum;
    }
}

juce::Colour SpectrogramComponent::getColourForValue(float value)
{
    value = juce::jlimit(0.0f, 1.0f, value);

    switch (colourScheme)
    {
    case ColourScheme::Classic:
        return juce::Colour::fromHSV(value, 1.0f, value, 1.0f);

    case ColourScheme::Grayscale:
        return juce::Colour::fromFloatRGBA(value, value, value, 1.0f);

    case ColourScheme::GrayscaleEnhanced:
    {
        float adjusted = 0.0f;

        if (value <= 0.8f)
            adjusted = 0.25f * value;  // y=[0.0, 0.2]
        else if (value <= 0.9f)
            adjusted = 0.2f + 1.0f * (value - 0.8f);  // y=[0.2, 0.3]
        else if (value <= 0.95f)
            adjusted = 0.3f + 5.0f * (value - 0.9f);  // y=[0.3, 0.55]
        else
            adjusted = 0.55f + 9.0f * (value - 0.95f);  // y=[0.55, 1.0]

        // Clip to [0, 1] to avoid rounding issues
        adjusted = juce::jlimit(0.0f, 1.0f, adjusted);

        return juce::Colour::fromFloatRGBA(adjusted, adjusted, adjusted, 1.0f);
    }

    case ColourScheme::Magma:
    {
        using ColourMaps::magmaColors;
        float scaled = value * 9.0f;
        int idxLow = static_cast<int>(std::floor(scaled));
        int idxHigh = std::min(idxLow + 1, 9);
        float t = scaled - idxLow;

        juce::Colour c1 = magmaColors[idxLow];
        juce::Colour c2 = magmaColors[idxHigh];

        return c1.interpolatedWith(c2, t);
    }

    case ColourScheme::MagmaEnhanced:
    {
        using ColourMaps::magmaColors;
        float adjusted = 0.0f;
        if (value <= 0.8f)
            adjusted = 0.25f * value;  // y=[0.0, 0.2]
        else if (value <= 0.9f)
            adjusted = 0.2f + 1.0f * (value - 0.8f);  // y=[0.2, 0.3]
        else if (value <= 0.95f)
            adjusted = 0.3f + 5.0f * (value - 0.9f);  // y=[0.3, 0.55]
        else
            adjusted = 0.55f + 9.0f * (value - 0.95f);  // y=[0.55, 1.0]

        adjusted = juce::jlimit(0.0f, 1.0f, adjusted);

        // === Magma color lookup ===
        float scaled = adjusted * 9.0f;
        int idxLow = static_cast<int>(std::floor(scaled));
        int idxHigh = std::min(idxLow + 1, 9);
        float t = scaled - idxLow;

        juce::Colour c1 = magmaColors[idxLow];
        juce::Colour c2 = magmaColors[idxHigh];

        return c1.interpolatedWith(c2, t);
    }

    default:
        return juce::Colours::black;
    }

}

void SpectrogramComponent::resized()
{
    spectrogramImage = juce::Image(juce::Image::RGB, getWidth(), getHeight(), true);
}