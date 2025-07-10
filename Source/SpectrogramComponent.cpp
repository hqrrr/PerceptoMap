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

void SpectrogramComponent::drawNextLineOfSpectrogram()
{
    const int imageWidth = spectrogramImage.getWidth();
    const int imageHeight = spectrogramImage.getHeight();
    std::vector<float> dBColumn(imageHeight);

    // Scroll the image left by one pixel
    spectrogramImage.moveImageSection(0, 0, 1, 0, imageWidth - 1, imageHeight);
    const int x = imageWidth - 1; // rightmost column

    const float maxFreq = sampleRate / 2.0f;

    if (currentMode == SpectrogramMode::Mel)
    {
        const int melBands = 128;
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
            // Simplified: directly take the FFT value corresponding to this frequency
            melEnergies[m] = fftData[bin];
        }

        const int imageHeight = spectrogramImage.getHeight();

        for (int y = 0; y < imageHeight; ++y)
        {
            int melIndex = juce::jmap(y, 0, imageHeight - 1, melBands - 1, 0);
            melIndex = std::clamp(melIndex, 0, melBands - 1);

            float dB = 20.0f * std::log10(melEnergies[melIndex] + 1e-6f);
            float clippedDB = juce::jlimit(-100.0f, 0.0f, dB);
            dBColumn[y] = clippedDB;
            float brightness = juce::jmap(clippedDB, -100.0f, 0.0f, 0.0f, 1.0f);

            juce::Colour colour = getColourForValue(brightness);
            spectrogramImage.setPixelAt(x, y, colour);
        }
    }
    else
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
            float magnitude = fftData[binIndex];
            float dB = 20.0f * std::log10(magnitude + 1e-6f); // avoid log(0)
            float clippedDB = juce::jlimit(-100.0f, 0.0f, dB);
            dBColumn[y] = clippedDB;
            float brightness = juce::jmap(clippedDB, -100.0f, 0.0f, 0.0f, 1.0f);

            juce::Colour colour = getColourForValue(brightness);
            spectrogramImage.setPixelAt(x, y, colour);
        }
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
    else
    {
        // log y axis or linear, for linear STFT spectrogram
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

            if (currentMode == SpectrogramMode::Mel)
            {
                int melIndex = juce::jlimit(0, (int)melBandFrequencies.size() - 1,
                    (int)((float)imgY / getHeight() * melBandFrequencies.size()));
                freq = melBandFrequencies[melBandFrequencies.size() - 1 - melIndex];
            }
            else
            {
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
            }

            juce::String labelText = juce::String(freq, 1) + " Hz, " + juce::String(dB, 1) + " dB";

            // Draw fixed box under legend bar (top right)
            const int boxWidth = 160;
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

juce::Colour SpectrogramComponent::getColourForValue(float value)
{
    value = juce::jlimit(0.0f, 1.0f, value);

    switch (colourScheme)
    {
    case ColourScheme::Classic:
        return juce::Colour::fromHSV(value, 1.0f, value, 1.0f);

    case ColourScheme::Grayscale:
        return juce::Colour::fromFloatRGBA(value, value, value, 1.0f);

    case ColourScheme::Magma:
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

        float scaled = value * 9.0f;
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