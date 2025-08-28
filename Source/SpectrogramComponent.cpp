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
{
    // allocate default FFT state
    fftSize = 1 << fftOrder;
    // init FFT
    forwardFFT = juce::dsp::FFT(fftOrder);
    // init window
    window = std::make_unique<juce::dsp::WindowingFunction<float>>(
        fftSize, juce::dsp::WindowingFunction<float>::hann, true);

    fifo.assign(fftSize, 0.0f);
    fftData.assign(fftSize * 2, 0.0f);
    fifoIndex = 0;
    nextFFTBlockReady = false;

    // init overlap/hop ring buffer
    hopSize = std::max(1, fftSize / overlap);
    ring.assign(fftSize, 0.0f);
    ringWrite = 0;
    samplesSinceHop = 0;

    // pre-build
    buildChromaFilterBank(fftSize, sampleRate);

    spectrogramImage = juce::Image(juce::Image::RGB, fftSize, fftSize / 2, true);
    // default FPS: 60 FPS
    startTimerHz(60);
    // scrolling
    //pixelsPerSecond = (hopSize > 0) ? (static_cast<float>(sampleRate) / hopSize) : 0.0f;
    pixelsPerSecond = baseScrollPps * static_cast<float>(scrollSpeedMul);
    pixelAccum = 0.0f;
    imgColAge.assign(spectrogramImage.getWidth(), -1);

    // init fourier tempogram
    initFourierTempogram();

    // mouse listener
    addMouseListener(this, true);
}

void SpectrogramComponent::setUiFps(int fps)
{
    // limit FPS between 15 and 240
    uiFps = juce::jlimit(15, 240, fps);
    startTimerHz(uiFps);
}

void SpectrogramComponent::setFFTOrder(int newOrder)
{
    // 512 - 8192
    newOrder = juce::jlimit(9, 13, newOrder);

    if (newOrder == fftOrder)
        return;

    fftOrder = newOrder;
    fftSize = 1 << fftOrder;

    forwardFFT = juce::dsp::FFT(fftOrder);

    window = std::make_unique<juce::dsp::WindowingFunction<float>>(
        fftSize, juce::dsp::WindowingFunction<float>::hann, true);

    fifo.assign(fftSize, 0.0f);
    fftData.assign(fftSize * 2, 0.0f);
    fifoIndex = 0;
    nextFFTBlockReady = false;

    // init FFT settings
    hopSize = std::max(1, fftSize / overlap);
    ring.assign(fftSize, 0.0f);
    ringWrite = 0;
    samplesSinceHop = 0;

    buildChromaFilterBank(fftSize, sampleRate);

    lastCentroidY = -1;
    hasPreviousCentroid = false;
    centroidSmoothed = 0.0f;

    dBBuffer.clear();

    juce::Graphics g(spectrogramImage);
    g.fillAll(juce::Colours::black);

    // scrolling
    imgColAge.assign(spectrogramImage.getWidth(), -1);

    // init fourier tempogram
    initFourierTempogram();

    repaint();
}

void SpectrogramComponent::setOverlap(int newOverlap)
{
    // only 1/2/4/8 allowed
    int allowed[] = {1, 2, 4, 8};
    int best = 1;
    for (int v : allowed) if (std::abs(v - newOverlap) < std::abs(best - newOverlap)) best = v;

    if (overlap == best) return;
    overlap = best;

    // Recalculate hop based on the current fftSize and reset the hop counter
    hopSize = std::max(1, fftSize / overlap);
    samplesSinceHop = 0;

    // Ensure that the ring size is correct
    if ((int)ring.size() != fftSize)
        ring.assign(fftSize, 0.0f);

    // scrolling
    if ((int)imgColAge.size() != spectrogramImage.getWidth())
        imgColAge.assign(spectrogramImage.getWidth(), -1);
    else
        std::fill(imgColAge.begin(), imgColAge.end(), -1);

    // init fourier tempogram
    initFourierTempogram();
}

void SpectrogramComponent::setScrollSpeedMultiplier(int mulX)
{
    // only 1/2/4/8/12/16 allowed
    int allowed[] = {1, 2, 4, 8, 12, 16};
    int best = 1;
    for (int v : allowed)
        if (std::abs(v - mulX) < std::abs(best - mulX)) best = v;

    if (scrollSpeedMul == best) return;
    scrollSpeedMul = best;

    pixelsPerSecond = baseScrollPps * static_cast<float>(scrollSpeedMul);
    pixelAccum = 0.0f;
}

void SpectrogramComponent::timerCallback()
{
    if (isFrozen)
        return;

    // cumulative pixels that should be rolled in this frame.
    pixelAccum += (pixelsPerSecond / static_cast<float>(uiFps));

    // number of “whole pixel columns” that need to be advanced in this frame.
    int steps = static_cast<int>(std::floor(pixelAccum));
    if (steps <= 0)
        return; // If less than 1px, wait for the next frame; ensure sub-pixel smoothness.

    pixelAccum -= static_cast<float>(steps);

    // Use a new column when there is a new FFT column
    // reuse the previous column if there isn't one
    for (int s = 0; s < steps; ++s)
    {
        if (nextFFTBlockReady)
        {
            drawNextLineOfSpectrogram();  // use one frame real data and scroll 1px
            nextFFTBlockReady = false;
        }
        else
        {
            // no new data: scroll 1px and repeat the previous column
            const int imageWidth = spectrogramImage.getWidth();
            const int imageHeight = spectrogramImage.getHeight();
            if (imageWidth < 2 || imageHeight <= 0)
                continue;
            // 1px to left
            spectrogramImage.moveImageSection(0, 0, 1, 0, imageWidth - 1, imageHeight);
            // copy last column
            spectrogramImage.moveImageSection(imageWidth - 1, 0, imageWidth - 2, 0, 1, imageHeight);
            if ((int)imgColAge.size() != imageWidth)
                imgColAge.assign(imageWidth, -1);

            if (!imgColAge.empty())
                std::move(imgColAge.begin() + 1, imgColAge.end(), imgColAge.begin());

            if (imageWidth >= 2)
                imgColAge[imageWidth - 1] = imgColAge[imageWidth - 2];
        }
    }
    repaint();
}

void SpectrogramComponent::pushNextFFTBlock(const float* data, size_t numSamples)
{
    for (size_t i = 0; i < numSamples; ++i)
    {
        ring[ringWrite] = data[i];
        ringWrite = (ringWrite + 1) % fftSize;
        samplesSinceHop++;

        if (samplesSinceHop >= hopSize && !nextFFTBlockReady)
        {
            // Copy the last fftSize samples
            const size_t start = (ringWrite + fftSize - fftSize) % fftSize;
            size_t firstLen = std::min((size_t)fftSize, (size_t)(fftSize - start));
            std::copy(ring.begin() + start, ring.begin() + start + firstLen, fftData.begin());
            if (firstLen < (size_t)fftSize)
                std::copy(ring.begin(), ring.begin() + (fftSize - firstLen), fftData.begin() + firstLen);

            std::fill(fftData.begin() + fftSize, fftData.end(), 0.0f);

            // no window if linear+ mode
            if (currentMode != SpectrogramMode::LinearPlus)
                window->multiplyWithWindowingTable(fftData.data(), fftSize);

            forwardFFT.performFrequencyOnlyForwardTransform(fftData.data());

            // Fourier tempogram
            // compute spectral flux novelty and push into novelty ring buffer
            if (currentMode == SpectrogramMode::FourierTempogram)
            {
                // current magnitude spectrum in [0 ... fftSize/2)
                const int nBins = fftSize / 2;
                const float C = 10.0f;  // Compression constant, e.g. 5-20
                float flux = 0.0f;
                for (int k = 0; k < nBins; ++k)
                {
                    // log(1 + C|X|)
                    float y = std::log1p(C * fftData[k]);
                    float py = haveLastMag ? lastCompSpec[k] : y;
                    float diff = y - py;
                    if (diff > 0.0f) flux += diff;
                    lastCompSpec[k] = y;
                }
                if (!haveLastMag) haveLastMag = true;
                flux /= (float)nBins;   // normalization

                //for (int k = 0; k < nBins; ++k) lastMagSpec[k] = fftData[k];

                // Write to buffer
                if (!noveltyRing.empty())
                {
                    noveltyRing[noveltyWrite] = flux;
                    noveltyWrite = (noveltyWrite + 1) % noveltyRing.size();
                }
            }

            nextFFTBlockReady = true;
            samplesSinceHop = 0;
        }
    }
}

void SpectrogramComponent::setSampleRate(double newSampleRate)
{
    sampleRate = newSampleRate;
    // init filter bank for chromagram
    buildChromaFilterBank(fftSize, sampleRate);
    // scrolling
    if ((int)imgColAge.size() != spectrogramImage.getWidth())
        imgColAge.assign(spectrogramImage.getWidth(), -1);
    else
        std::fill(imgColAge.begin(), imgColAge.end(), -1);
    // limit max y frequency
    const float nyquist = static_cast<float>(sampleRate) * 0.5f;
    maxFreqHz = juce::jlimit(minFreqHz + 1.0f, nyquist, maxFreqHz);

    // init fourier tempogram
    initFourierTempogram();
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

int SpectrogramComponent::hzToY(float hz) const
{
    const int imageHeight = spectrogramImage.isValid() ? spectrogramImage.getHeight() : getHeight();
    if (imageHeight <= 0) return 0;

    if (currentMode == SpectrogramMode::Mel || currentMode == SpectrogramMode::MelPlus)
    {
        const float maxHz = static_cast<float>(sampleRate) * 0.5f;
        const float fMin = juce::jlimit(1.0f, maxHz - 1.0f, minFreqHz);
        const float fMax = juce::jlimit(fMin + 1.0f, maxHz, maxFreqHz);

        const float melMin = hzToMel(fMin);
        const float melMax = hzToMel(fMax);
        const float mel = hzToMel(juce::jlimit(fMin, fMax, hz));
        const float yNorm = (melMax - mel) / juce::jmax(1.0f, (melMax - melMin));
        return juce::jlimit(0, imageHeight - 1, (int)std::lround(yNorm * (imageHeight - 1)));
    }
    else
    {
        const float fMin = minFreqHz;
        const float fMax = maxFreqHz;
        float yNorm = 0.0f;

        if (useLogFrequency)
        {
            const float logMin = std::log10(juce::jmax(1.0f, fMin));
            const float logMax = std::log10(juce::jmax(fMin + 1.0f, fMax));
            const float logF = std::log10(juce::jlimit(juce::jmax(1.0f, fMin), juce::jmax(fMin + 1.0f, fMax), hz));
            yNorm = (logMax - logF) / juce::jmax(1.0f, (logMax - logMin));
        }
        else
        {
            const float f = juce::jlimit(fMin, fMax, hz);
            yNorm = (fMax - f) / juce::jmax(1.0f, (fMax - fMin));
        }

        return juce::jlimit(0, imageHeight - 1, (int)std::lround(yNorm * (imageHeight - 1)));
    }
}

float SpectrogramComponent::bpmToImageY(float bpm, int imageHeight) const
{
    if (imageHeight <= 1) return -1.0f;

    const float bmin = juce::jmax(1.0f, tempoMinBPM);
    const float bmax = juce::jmax(tempoMinBPM + 1.0f, tempoMaxBPM);
    const float logMin = std::log10(bmin);
    const float logMax = std::log10(bmax);

    const float logB = std::log10(juce::jlimit(bmin, bmax, bpm));
    float alpha = (logB - logMin) / juce::jmax(1e-9f, (logMax - logMin));
    alpha = juce::jlimit(0.0f, 1.0f, alpha);

    const float fracTop = 1.0f - alpha;
    return fracTop * (imageHeight - 1);
}

void SpectrogramComponent::drawLinearSpectrogram(int x, std::vector<float>& dBColumn, const int imageHeight, const float maxFreq)
{
    // default: linear STFT spectrogram
    const float maxHz = static_cast<float>(sampleRate) * 0.5f;
    const int   nBins = fftSize / 2;

    for (int y = 0; y < imageHeight; ++y)
    {
        //int binIndex = 0;
        const float fracTopToBottom = static_cast<float>(y) / imageHeight;
        const float fracBottomToTop = 1.0f - fracTopToBottom;
        float freqHz = 0.0f;

        if (useLogFrequency)
        {
            //float logMinFreq = std::log10(30.0f); // lower bound (must match label start)
            //float logMaxFreq = std::log10(maxFreq);
            //float frac = 1.0f - static_cast<float>(y) / imageHeight; // low -> high

            //float logFreq = logMinFreq + frac * (logMaxFreq - logMinFreq);
            //float freq = std::pow(10.0f, logFreq);
            //binIndex = static_cast<int>((freq / maxFreq) * (fftSize / 2));
            const float logMin = std::log10(std::max(1.0f, minFreqHz));
            const float logMax = std::log10(std::max(minFreqHz + 1.0f, maxFreqHz));
            const float logVal = logMin + fracBottomToTop * (logMax - logMin);
            freqHz = std::pow(10.0f, logVal);
        }
        else
        {
            //float frac = 1.0f - static_cast<float>(y) / imageHeight;
            //binIndex = static_cast<int>(frac * (fftSize / 2));
            freqHz = minFreqHz + fracBottomToTop * (maxFreqHz - minFreqHz);
        }

        int binIndex = static_cast<int>((freqHz / maxHz) * nBins);
        //binIndex = std::clamp(binIndex, 0, fftSize / 2 - 1);
        binIndex = juce::jlimit(0, nBins - 1, binIndex);

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

    const float maxHz = static_cast<float>(sampleRate) * 0.5f;

    const float melMin = hzToMel(std::max(1.0f, minFreqHz));
    const float melMax = hzToMel(std::max(minFreqHz + 1.0f, maxFreqHz));

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
    float rawCentroidHz = computeSpectralCentroid(fftData.data(), fftSize / 2);

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

    // Draw STFT
    SpectrogramComponent::drawLinearSpectrogram(x, dBColumn, imageHeight, maxFreq);

    // Draw spectral centroid
    auto mapHzToY = [imageHeight = this->getHeight(), this, maxFreq](float freqHz) -> int
    {
        freqHz = juce::jlimit(minFreqHz, maxFreqHz, freqHz);
        float yNorm = 0.0f;
        if (useLogFrequency)
        {
            const float logMin = std::log10(std::max(1.0f, minFreqHz));
            const float logMax = std::log10(std::max(minFreqHz + 1.0f, maxFreqHz));
            const float logF = std::log10(std::max(1.0f, freqHz));
            yNorm = 1.0f - (logF - logMin) / (logMax - logMin);
        }
        else
        {
            yNorm = 1.0f - (freqHz - minFreqHz) / (maxFreqHz - minFreqHz);
        }
        return juce::jlimit(0, imageHeight - 1, static_cast<int>(yNorm * imageHeight));
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

void SpectrogramComponent::drawReassignedSpectrogram(
    int x, std::vector<float>& dBColumn, int imageHeight, float maxFreq)
{
    // init Gaussian window, time-weighted, and analytic derivative
    std::vector<float> gauss(fftSize), gauss_t(fftSize), gauss_d(fftSize);
    const float N = static_cast<float>(fftSize);
    const float mu = (N - 1) / 2.0f;
    const float sigma = sigmaCoef * mu;

    for (int n = 0; n < fftSize; ++n)
    {
        float t = static_cast<float>(n) - mu;
        float w = std::exp(-0.5f * (t * t) / (sigma * sigma));
        gauss[n] = w;
        gauss_t[n] = t * w;
        gauss_d[n] = -(t / (sigma * sigma)) * w;
    }

    std::vector<float> timeBlock(fftSize);
    const size_t w = ringWrite;
    size_t firstLen = std::min((size_t)fftSize, ring.size() > 0 ? (ring.size() - w) : (size_t)0);
    if (firstLen > 0 && ring.size() >= (size_t)fftSize)
    {
        std::copy(ring.begin() + w, ring.begin() + w + firstLen, timeBlock.begin());
        if (firstLen < (size_t)fftSize)
            std::copy(ring.begin(), ring.begin() + (fftSize - firstLen), timeBlock.begin() + firstLen);
    }
    else
    {
        std::fill(timeBlock.begin(), timeBlock.end(), 0.0f);
    }

    // prepare FFT arrays
    std::vector<float> Xf(fftSize * 2, 0.0f), Xtf(fftSize * 2, 0.0f), Xdf(fftSize * 2, 0.0f);
    for (int n = 0; n < fftSize; ++n)
    {
        float x = timeBlock[n];
        Xf[n] = x * gauss[n];
        Xtf[n] = x * gauss_t[n];
        Xdf[n] = x * gauss_d[n];
    }
    juce::dsp::FFT fft(fftOrder);
    fft.performRealOnlyForwardTransform(Xf.data());
    //fft.performRealOnlyForwardTransform(Xtf.data());    // not used yet
    fft.performRealOnlyForwardTransform(Xdf.data());

    // clear output column and prepare accumulator
    for (int y = 0; y < imageHeight; ++y)
        dBColumn[y] = floorDb;
    std::vector<float> pixelEnergy(imageHeight, floorDb);

    auto getComplexBin = [&](const std::vector<float>& buf, int k) -> std::complex<float>
    {
        if (k == 0)               return { buf[0], 0.0f };
        else if (k == fftSize / 2)  return { buf[1], 0.0f };
        else                      return { buf[2 * k], buf[2 * k + 1] };
    };

    const int nBins = fftSize / 2;
    // for each positive frequency bin, compute reassigned frequency
    for (int k = 0; k < nBins; ++k)
    {
        std::complex<float> stft = getComplexBin(Xf, k);
        std::complex<float> stft_d = getComplexBin(Xdf, k);

        float mag2 = std::norm(stft * normFactor);
        if (mag2 < thresholdFactor) continue; // reject low magnitude
        if (std::abs(stft) < 1e-12f) continue; // avoid division by zero

        // frequency bin center
        float omega = 2.0f * juce::MathConstants<float>::pi * k / float(fftSize);
        float omega_hat = omega + std::imag(stft_d / stft);

        float freq_hat = omega_hat * sampleRate / (2.0f * juce::MathConstants<float>::pi);

        if (!std::isfinite(freq_hat) || freq_hat < minFreqHz || freq_hat > maxFreqHz)
            continue;

        // map freq to vertical pixel
        int y = 0;
        if (useLogFrequency)
        {
            const float logMin = std::log10(std::max(1.0f, minFreqHz));
            const float logMax = std::log10(std::max(minFreqHz + 1.0f, maxFreqHz));
            const float logF = std::log10(std::max(1.0f, freq_hat));
            const float yNorm = 1.0f - (logF - logMin) / (logMax - logMin);
            y = juce::jlimit(0, imageHeight - 1, (int)std::lround(yNorm * (imageHeight - 1)));
        }
        else
        {
            const float yNorm = 1.0f - (freq_hat - minFreqHz) / (maxFreqHz - minFreqHz);
            y = juce::jlimit(0, imageHeight - 1, (int)std::lround(yNorm * (imageHeight - 1)));
        }
        if (y >= 0 && y < imageHeight)
        {
            float dB = 10.0f * std::log10(mag2 + 1e-9f);
            dB = juce::jlimit(floorDb, 0.0f, dB);
            pixelEnergy[y] = std::max(pixelEnergy[y], dB);
        }
    }

    // paint the reassigned column into the spectrogram image and buffer
    for (int y = 0; y < imageHeight; ++y)
    {
        dBColumn[y] = pixelEnergy[y];
        float brightness = juce::jmap(pixelEnergy[y], floorDb, 0.0f, 0.0f, 1.0f);
        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
    }
}

void SpectrogramComponent::drawReassignedMelSpectrogram(
    int x, std::vector<float>& dBColumn, int imageHeight)
{
    const float maxHz = static_cast<float>(sampleRate) / 2.0f;
    const float melMin = hzToMel(std::max(1.0f, minFreqHz));
    const float melMax = hzToMel(std::max(minFreqHz + 1.0f, maxFreqHz));

    // Accumulate by mag^2, then convert to dB.
    std::vector<float> melEnergyLin(melBands, 0.0f);

    // Same as Linear+: construct three windows and perform 3 FFTs with Gaussian window.
    std::vector<float> gauss(fftSize), gauss_t(fftSize), gauss_d(fftSize);
    const float N = static_cast<float>(fftSize);
    const float mu = (N - 1.0f) / 2.0f;
    const float sigma = sigmaCoef * mu;

    for (int n = 0; n < fftSize; ++n)
    {
        float t = static_cast<float>(n) - mu;
        float w = std::exp(-0.5f * (t * t) / (sigma * sigma));
        gauss[n] = w;
        gauss_t[n] = t * w;
        gauss_d[n] = -(t / (sigma * sigma)) * w;
    }

    std::vector<float> timeBlock(fftSize, 0.0f);
    const size_t w = ringWrite;
    size_t firstLen = std::min((size_t)fftSize, ring.size() > 0 ? (ring.size() - w) : (size_t)0);
    if (firstLen > 0 && ring.size() >= (size_t)fftSize)
    {
        std::copy(ring.begin() + w, ring.begin() + w + firstLen, timeBlock.begin());
        if (firstLen < (size_t)fftSize)
            std::copy(ring.begin(), ring.begin() + (fftSize - firstLen), timeBlock.begin() + firstLen);
    }
    else
    {
        std::fill(timeBlock.begin(), timeBlock.end(), 0.0f);
    }

    // prepare FFT arrays
    std::vector<float> Xf(fftSize * 2, 0.0f), Xtf(fftSize * 2, 0.0f), Xdf(fftSize * 2, 0.0f);
    for (int n = 0; n < fftSize; ++n)
    {
        float x = timeBlock[n];
        Xf[n] = x * gauss[n];
        Xtf[n] = x * gauss_t[n];
        Xdf[n] = x * gauss_d[n];
    }
    juce::dsp::FFT fft(fftOrder);
    fft.performRealOnlyForwardTransform(Xf.data());
    //fft.performRealOnlyForwardTransform(Xtf.data());    // not used yet
    fft.performRealOnlyForwardTransform(Xdf.data());

    // clear output column and prepare accumulator
    for (int y = 0; y < imageHeight; ++y)
        dBColumn[y] = floorDb;
    std::vector<float> pixelEnergy(imageHeight, floorDb);

    auto getComplexBin = [&](const std::vector<float>& buf, int k) -> std::complex<float>
    {
        if (k == 0)               return { buf[0], 0.0f };
        else if (k == fftSize / 2)  return { buf[1], 0.0f };
        else                      return { buf[2 * k], buf[2 * k + 1] };
    };

    const int nBins = fftSize / 2;
    // Frequency reassignment: assign the energy of each bin to the corresponding Mel band
    for (int k = 0; k < nBins; ++k)
    {
        std::complex<float> stft = getComplexBin(Xf, k);
        std::complex<float> stft_d = getComplexBin(Xdf, k);

        float mag2 = std::norm(stft * normFactor);
        if (mag2 < thresholdFactor) continue; // reject low magnitude
        if (std::abs(stft) < 1e-12f) continue; // avoid division by zero

        // frequency bin center
        float omega = 2.0f * juce::MathConstants<float>::pi * k / float(fftSize);
        float omega_hat = omega + std::imag(stft_d / stft);
        float freq_hat = omega_hat * sampleRate / (2.0f * juce::MathConstants<float>::pi);

        if (!std::isfinite(freq_hat) || freq_hat < minFreqHz || freq_hat > maxFreqHz)
            continue;

        // Frequency -> Mel index
        //float m = (hzToMel(freq_hat) - melMin) / (melMax - melMin) * (melBands - 1);
        //int   mi = juce::jlimit(0, melBands - 1, int(std::round(m)));

        //melEnergyLin[mi] += mag2;   // Cumulative energy
        //melEnergyLin[mi] = juce::jmax(melEnergyLin[mi], mag2);  // Alternative: sharper transients
        
        // Frequency -> Mel
        float mNorm = (hzToMel(freq_hat) - melMin) / (melMax - melMin); // [0,1]
        int yPix = juce::jlimit(0, imageHeight - 1,
            (int)std::lround((1.0f - mNorm) * (imageHeight - 1)));

        float dB = 10.0f * std::log10(mag2 + 1e-9f);
        dB = juce::jlimit(floorDb, 0.0f, dB);
        pixelEnergy[yPix] = std::max(pixelEnergy[yPix], dB);
    }

    // Map Mel energy to pixel
    for (int y = 0; y < imageHeight; ++y)
    {
        //int mi = juce::jmap(y, 0, imageHeight - 1, melBands - 1, 0);
        //mi = std::clamp(mi, 0, melBands - 1);
        //float dB = 10.0f * std::log10(melEnergyLin[mi] + 1e-9f);

        dBColumn[y] = pixelEnergy[y];

        float brightness = juce::jmap(pixelEnergy[y], floorDb, 0.0f, 0.0f, 1.0f);
        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
    }
}

void SpectrogramComponent::drawFourierTempogram(int x, std::vector<float>& dBColumn, int imageHeight)
{
    if (noveltyRing.empty() || noveltyWinLen <= 2 || imageHeight <= 0)
        return;

    // tempo axis (logarithmic uniform)
    const int M = juce::jmax(8, tempoBins);
    std::vector<float> tempoBPM(M);

    float logMin = std::log10(juce::jmax(1.0f, tempoMinBPM));
    float logMax = std::log10(juce::jmax(tempoMinBPM + 1.0f, tempoMaxBPM));
    for (int m = 0; m < M; ++m)
    {
        float alpha = (float)m / (float)(M - 1);
        float logBpm = logMin + alpha * (logMax - logMin);
        tempoBPM[m] = std::pow(10.0f, logBpm);
    }

    // Get a novelty segment of length (L)
    std::vector<float> seg(noveltyWinLen, 0.0f);
    for (int i = 0; i < noveltyWinLen; ++i)
    {
        int idx = (int)noveltyWrite - 1 - i;
        if (idx < 0) idx += (int)noveltyRing.size();
        seg[noveltyWinLen - 1 - i] = noveltyRing[(size_t)idx];
    }

    // local mean removal + half-wave
    float mu = 0.0f;
    for (float v : seg) mu += v;
    mu /= (float)noveltyWinLen;
    // activity gate
    float activity = 0.0f;
    for (int i = 0; i < noveltyWinLen; ++i)
    {
        seg[i] = std::max(0.0f, seg[i] - mu);
        activity += seg[i];
    }
    // low-activity gating
    if (activity < activityThresh)
    {
        // Draw the entire column directly near the floor
        for (int y = 0; y < imageHeight; ++y)
        {
            dBColumn[y] = floorDb;
            float brightness = 0.0f;
            juce::Colour colour = getColourForValue(brightness);
            spectrogramImage.setPixelAt(x, y, colour);
        }
        lastFourierTempoY = -1;
        return;
    }

    // tempogram STFT
    const double r = noveltySamplePeriod; // seconds/sample
    std::vector<float> mag(M, 0.0f);
    const double winSum = std::accumulate(noveltyWin.begin(), noveltyWin.end(), 0.0);
    const double norm = (winSum > 1e-12) ? (1.0 / winSum) : 1.0;
    const double twoPi = 2.0 * juce::MathConstants<double>::pi;

    for (int m = 0; m < M; ++m)
    {
        const double cyclesPerSample = (tempoBPM[m] / 60.0) * r;
        double re = 0.0, im = 0.0;

        for (int i = 0; i < noveltyWinLen; ++i)
        {
            double phase = -twoPi * cyclesPerSample * (double)i;
            double wv = (double)noveltyWin[i] * (double)seg[i] * norm;
            re += wv * std::cos(phase);
            im += wv * std::sin(phase);
        }
        mag[m] = (float)std::sqrt(re * re + im * im);
    }

    // Map M tempo bins to imageHeight
    for (int y = 0; y < imageHeight; ++y)
    {
        float frac = 1.0f - ((float)y / (float)(imageHeight - 1));
        int m = juce::jlimit(0, M - 1, (int)std::round(frac * (M - 1)));

        float v = mag[m] * normFactor;
        float dB = 20.0f * std::log10(v + 1e-9f);
        float clipped = juce::jlimit(floorDb, 0.0f, dB);
        dBColumn[y] = clipped;

        float brightness = juce::jmap(clipped, floorDb, 0.0f, 0.0f, 1.0f);
        juce::Colour colour = getColourForValue(brightness);
        spectrogramImage.setPixelAt(x, y, colour);
    }

    // Overlay Fourier tempo line
    // Define highlight colors
    juce::Colour fourierTempoLineColour;

    switch (colourScheme)
    {
    case ColourScheme::Grayscale:
        fourierTempoLineColour = juce::Colour::fromRGB(0, 200, 255);   // blue
        break;
    case ColourScheme::GrayscaleEnhanced:
        fourierTempoLineColour = juce::Colour::fromRGB(0, 200, 255);   // blue
        break;
    case ColourScheme::Magma:
        fourierTempoLineColour = juce::Colour::fromRGB(0, 255, 128);   // light green
        break;
    case ColourScheme::MagmaEnhanced:
        fourierTempoLineColour = juce::Colour::fromRGB(0, 255, 128);   // light green
        break;

    default:
        fourierTempoLineColour = juce::Colour::fromRGB(255, 255, 255);   // white
        break;
    }
    if (showFourierTempoLine && M >= 2)
    {
        int   mPeak = 0;
        float bestVal = -1.0e30f;
        float bestBPM = tempoBPM[0];

        for (int m = 0; m < M; ++m)
        {
            const float bpm = tempoBPM[m];
            const float prior = fourierTempoPrior(bpm);
            const float score = mag[m] * prior;
            if (score > bestVal)
            {
                bestVal = score;
                mPeak = m;
                bestBPM = bpm;
            }
        }

        const int yTempo = (int)std::round(bpmToImageY(bestBPM, imageHeight));
        if (yTempo >= 0 && yTempo < imageHeight)
        {
            // draw on tempogram
            juce::Graphics imgG(spectrogramImage);
            imgG.setColour(fourierTempoLineColour.withAlpha(0.95f));

            if (lastFourierTempoY >= 0)
            {
                imgG.drawLine((float)(x - 1), (float)lastFourierTempoY,
                    (float)x, (float)yTempo,
                    2.0f); // line width
            }
            else
            {
                // draw first point
                imgG.fillRect(x, yTempo, 1, 1);
            }
            lastFourierTempoY = yTempo;
        }
        else
        {
            lastFourierTempoY = -1;
        }
    }
}

void SpectrogramComponent::drawNextLineOfSpectrogram()
{
    const int imageWidth = spectrogramImage.getWidth();
    const int imageHeight = spectrogramImage.getHeight();
    std::vector<float> dBColumn(imageHeight);

    // Scroll the image left by one pixel
    spectrogramImage.moveImageSection(0, 0, 1, 0, imageWidth - 1, imageHeight);

    if ((int)imgColAge.size() != imageWidth)
        imgColAge.assign(imageWidth, -1);

    if (!imgColAge.empty())
        std::move(imgColAge.begin() + 1, imgColAge.end(), imgColAge.begin());

    for (int i = 0; i < imageWidth - 1; ++i)
        if (imgColAge[i] >= 0) ++imgColAge[i];

    imgColAge[imageWidth - 1] = 0;

    const int x = imageWidth - 1; // rightmost column

    const float maxFreq = sampleRate / 2.0f;

    switch (currentMode)
    {
        case SpectrogramMode::Mel:
            drawMelSpectrogram(x, dBColumn, imageHeight);
            break;
        case SpectrogramMode::MFCC:
            drawMFCC(x, dBColumn, imageHeight);
            break;
        case SpectrogramMode::LinearWithCentroid:
            drawLinearWithCentroid(x, dBColumn, imageHeight, maxFreq);
            break;
        case SpectrogramMode::Chroma:
            drawChroma(x, dBColumn, imageHeight);
            break;
        case SpectrogramMode::LinearPlus:
            drawReassignedSpectrogram(x, dBColumn, imageHeight, maxFreq);
            break;
        case SpectrogramMode::MelPlus:
            drawReassignedMelSpectrogram(x, dBColumn, imageHeight);
            break;
        case SpectrogramMode::FourierTempogram:
            drawFourierTempogram(x, dBColumn, imageHeight);
            break;
        default:
            drawLinearSpectrogram(x, dBColumn, imageHeight, maxFreq);
            break;
    }

    if (dBBuffer.size() >= imageWidth)
        dBBuffer.erase(dBBuffer.begin());

    dBBuffer.push_back(dBColumn);
}

void SpectrogramComponent::paintMelYAxis(juce::Graphics& g, const int width, const int imageHeight)
{
    // Mel scale tick positions (Slaney-style spacing, approximate)
    const int melBands = imageHeight;
    const int numLabels = 10;

    const float melMin = hzToMel(std::max(1.0f, minFreqHz));
    const float melMax = hzToMel(std::max(minFreqHz + 1.0f, maxFreqHz));

    for (int i = 0; i < numLabels; ++i)
    {
        float yNorm = static_cast<float>(i) / (numLabels - 1);
        float mel = melMax - yNorm * (melMax - melMin);
        float freq = melToHz(mel);
        const int y = juce::jlimit(0, imageHeight - 1, (int)std::lround(yNorm * (imageHeight - 1)));

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

void SpectrogramComponent::paintMFCCYAxis(juce::Graphics& g, const int width, const int imageHeight)
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

void SpectrogramComponent::paintChromaYAxis(juce::Graphics& g, const int width, const int imageHeight)
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

void SpectrogramComponent::paintSTFTYAxis(juce::Graphics& g, const int width, const int imageHeight)
{
    // log y axis or linear, for Linear / Linear+ / Linear+Centroid
    const int numLabels = 10;
    for (int i = 0; i < numLabels; ++i)
    {
        const float t = static_cast<float>(i) / (numLabels - 1);
        float fHz = 0.0f;
        float y = 0.0f;

        if (useLogFrequency)
        {
            const float logMin = std::log10(std::max(1.0f, minFreqHz));
            const float logMax = std::log10(std::max(minFreqHz + 1.0f, maxFreqHz));
            const float logF = logMin + (1.0f - t) * (logMax - logMin);
            fHz = std::pow(10.0f, logF);
            y = (1.0f - (logF - logMin) / (logMax - logMin)) * imageHeight;
        }
        else
        {
            fHz = minFreqHz + (1.0f - t) * (maxFreqHz - minFreqHz);
            y = (1.0f - (fHz - minFreqHz) / (maxFreqHz - minFreqHz)) * imageHeight;
        }

        const int yi = juce::jlimit(0, imageHeight - 1, static_cast<int>(std::round(y)));
        juce::String label = (fHz >= 1000.0f) ? juce::String(fHz / 1000.0f, 1) + " kHz"
            : juce::String(static_cast<int>(std::round(fHz))) + " Hz";

        juce::Rectangle<int> textBounds(2, yi - 12, 60, 16);
        g.setColour(juce::Colours::black.withAlpha(0.6f)); g.fillRect(textBounds);
        g.setColour(juce::Colours::white); g.drawText(label, textBounds, juce::Justification::left);
        g.setColour(juce::Colours::darkgrey.withAlpha(gridAlpha));
        g.drawHorizontalLine(yi, 55.0f, static_cast<float>(width));
    }
}

void SpectrogramComponent::paintNoteYAxis(juce::Graphics& g, const bool modeSupportsNoteAxis)
{
    if (modeSupportsNoteAxis)
    {
        const int width = getWidth();
        const int imageHeight = spectrogramImage.getHeight();

        // inner margin and scale settings
        const int axisRightX = width - 30;
        const int majorTickLen = 12;  // C
        const int minorTickLen = 6;   // others
        const int labelBoxW = 18;
        const int labelBoxH = 16;

        // A4 = 440 Hz，midi = 69 + 12*log2(f/440)
        auto midiToHz = [](int m) { return 440.0f * std::pow(2.0f, (m - 69) / 12.0f); };

        // Calculate the MIDI range
        const float fMin = juce::jmax(1.0f, minFreqHz);
        const float fMax = juce::jmax(fMin + 1.0f, maxFreqHz);
        auto hzToMidi = [](float hz) { return 69.0f + 12.0f * std::log2(hz / 440.0f); };
        int midiStart = (int)std::floor(hzToMidi(fMin));
        int midiEnd = (int)std::ceil(hzToMidi(fMax));

        // limit to 0-127
        midiStart = juce::jlimit(0, 127, midiStart);
        midiEnd = juce::jlimit(0, 127, midiEnd);

        // Draw axis.
        g.setColour(juce::Colours::white.withAlpha(0.4f));
        g.drawLine((float)axisRightX, 0.0f, (float)axisRightX, (float)imageHeight);

        // draw long tick for C note (m % 12 == 0) and add a label
        // draw short ticks for all others
        for (int m = midiStart; m <= midiEnd; ++m)
        {
            const float hz = midiToHz(m);
            const int y = hzToY(hz);

            const bool isC = (m % 12 == 0);
            const int tickLen = isC ? majorTickLen : minorTickLen;

            // tick
            g.setColour(juce::Colours::white.withAlpha(isC ? 0.9f : 0.5f));
            g.drawLine((float)axisRightX, (float)y, (float)(axisRightX - tickLen), (float)y);

            // label for C
            if (isC)
            {
                // octave = floor(m/12) - 1, m=60 -> C4
                const int octave = (m / 12) - 1;
                juce::String label = "C" + juce::String(octave);

                // background frame
                juce::Rectangle<int> textBounds(axisRightX - tickLen - labelBoxW - 2, y - labelBoxH / 2, labelBoxW, labelBoxH);
                g.setColour(juce::Colours::black.withAlpha(0.4f));
                g.fillRect(textBounds);

                g.setColour(juce::Colours::white);
                g.setFont(12.0f);
                g.drawText(label, textBounds, juce::Justification::centred, true);
            }
        }
    }
}

void SpectrogramComponent::paintTempoYAxis(juce::Graphics& g, int width, int imageHeight)
{
    if (width <= 0 || imageHeight <= 0 || tempoMinBPM <= 0.0f || tempoMaxBPM <= tempoMinBPM)
        return;

    const int x0 = getWidth() - width;
    const int yTop = 0;
    const int yBottom = imageHeight - 1;

    // background and axis
    g.setColour(juce::Colours::black.withAlpha(0.25f));
    g.fillRect(x0, yTop, width, imageHeight);

    g.setColour(juce::Colours::grey);
    g.drawLine((float)x0 + 0.5f, (float)yTop, (float)x0 + 0.5f, (float)yBottom, 1.0f);

    // map BMP to Y
    const double logMin = std::log10(juce::jmax(1.0f, tempoMinBPM));
    const double logMax = std::log10(juce::jmax(tempoMinBPM + 1.0f, tempoMaxBPM));
    auto bpmToY = [&](double bpm) -> float
    {
        if (bpm < tempoMinBPM || bpm > tempoMaxBPM) return -1.0f;
        const double pos = (std::log10(bpm) - logMin) / juce::jmax(1e-9, (logMax - logMin));
        const double y = (1.0 - pos) * (double)yBottom;
        return (float)std::round(y);
    };

    // Generate primary/secondary scales (Primary: 30*2^n; Secondary: 45*2^n)
    std::vector<double> majorTicks, minorTicks;
    for (int n = -10; n <= 10; ++n)
    {
        double vMajor = 30.0 * std::pow(2.0, n);
        double vMinor = 45.0 * std::pow(2.0, n);
        if (vMajor >= tempoMinBPM && vMajor <= tempoMaxBPM) majorTicks.push_back(vMajor);
        if (vMinor >= tempoMinBPM && vMinor <= tempoMaxBPM) minorTicks.push_back(vMinor);
    }

    // Draw grid lines
    g.setColour(juce::Colours::darkgrey.withAlpha(0.25f));
    for (double bpm : majorTicks)
    {
        float y = bpmToY(bpm);
        if (y >= 0.0f) g.drawLine(0.0f, y + 0.5f, (float)x0, y + 0.5f, 1.0f);
    }

    // Draw minor ticks
    g.setColour(juce::Colours::grey.withAlpha(0.8f));
    const int minorLen = juce::jmax(6, (int)(width * 0.03f));
    for (double bpm : minorTicks)
    {
        float y = bpmToY(bpm);
        if (y < 0.0f) continue;
        g.drawLine((float)x0 + 0.5f, y + 0.5f, (float)x0 + minorLen, y + 0.5f, 1.0f);
    }

    // Draw major ticks and text
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(11.0f));
    const int majorLen = juce::jmax(10, (int)(width * 0.06f));
    const int textPad = 2;

    for (double bpm : majorTicks)
    {
        float y = bpmToY(bpm);
        if (y < 0.0f) continue;

        g.drawLine((float)x0 + 0.5f, y + 0.5f, (float)x0 + majorLen, y + 0.5f, 1.2f);

        // label
        juce::String label = juce::String(juce::roundToInt(bpm));
        juce::Rectangle<int> textArea(x0 + majorLen + textPad, (int)y - 8, width - majorLen - textPad - 2, 16);
        g.drawFittedText(label, textArea, juce::Justification::left, 1);
    }

    // Title "BPM"
    g.setColour(juce::Colours::lightgrey);
    g.setFont(juce::Font(11.0f, juce::Font::italic));
    juce::String title = "BPM";
    g.drawFittedText(title, { x0 + 4, 2, width - 8, 16 }, juce::Justification::centredLeft, 1);
}

juce::String SpectrogramComponent::drawMelTooltip(float dB, const int imgY, float freq)
{
    // melspectrogram
    const float maxHz = static_cast<float>(sampleRate) * 0.5f;
    const float fMin = juce::jlimit(1.0f, maxHz - 1.0f, minFreqHz);
    const float fMax = juce::jlimit(fMin + 1.0f, maxHz, maxFreqHz);

    const int imageHeight = spectrogramImage.isValid() ? spectrogramImage.getHeight() : getHeight();
    const float yNormTopToBottom = (float)imgY / juce::jmax(1, imageHeight - 1);
    const float yFromBottom = 1.0f - yNormTopToBottom;

    const float melMin = hzToMel(fMin);
    const float melMax = hzToMel(fMax);
    const float melVal = melMin + yFromBottom * (melMax - melMin);
    freq = juce::jlimit(fMin, fMax, melToHz(melVal));

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
    juce::String labelText = juce::String(freq, 1) + " Hz, " + (noteName.isNotEmpty() ? "note: " + noteName : "")
        + ", " + juce::String(dB, 1) + " dB";
    
    return labelText;
}

juce::String SpectrogramComponent::drawMFCCTooltip(float dB, const int imgY, const int imageHeight)
{
    // MFCC
    float frac = static_cast<float>(imgY) / (imageHeight - 1);
    int coeffIndex = static_cast<int>((1.0f - frac) * (numCoeffs - 1));
    coeffIndex = juce::jlimit(0, numCoeffs - 1, coeffIndex);

    // Note: dB is actually MFCC (unitless), not decibel
    float norm = juce::jlimit(mfccMin, mfccMax, dB);
    float brightness = juce::jmap(norm, mfccMin, mfccMax, 0.0f, 1.0f);

    juce::String labelText = "MFCC " + juce::String(coeffIndex) + ", " + juce::String(brightness, 2)
        + " (normalized)";

    return labelText;
}

juce::String SpectrogramComponent::drawChromaTooltip(const int dBIndex, const int imgY, const int imageHeight)
{
    // Chromagram
    float blockHeight = static_cast<float>(imageHeight) / numChroma;
    int chromaIndex = static_cast<int>((imageHeight - 1 - imgY) / blockHeight);
    chromaIndex = juce::jlimit(0, numChroma - 1, chromaIndex);
    juce::String pitchName = pitchNames[chromaIndex];

    float brightness = juce::jlimit(0.0f, 1.0f, dBBuffer[dBIndex][imgY]);
    //float brightness = juce::jlimit(0.0f, 1.0f, dBBuffer[dBIndex][chromaIndex]);

    juce::String labelText = "Pitch Class: " + pitchName + ", " + juce::String(brightness, 2);

    return labelText;
}

juce::String SpectrogramComponent::drawSTFTTooltip(float dB, const int imgY, float freq)
{
    // STFT spectrogram
    const float maxHz = static_cast<float>(sampleRate) * 0.5f;

    const float fMin = juce::jlimit(1.0f, maxHz - 1.0f, minFreqHz);
    const float fMax = juce::jlimit(fMin + 1.0f, maxHz, maxFreqHz);

    const float yNormTopToBottom = (float)imgY / juce::jmax(1, getHeight() - 1);
    const float yFromBottom = 1.0f - yNormTopToBottom;

    if (useLogFrequency)
    {
        const float logMin = std::log10(fMin);
        const float logMax = std::log10(fMax);
        const float logFreq = logMin + yFromBottom * (logMax - logMin);
        freq = std::pow(10.0f, logFreq);
    }
    else
    {
        freq = fMin + yFromBottom * (fMax - fMin);
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
    juce::String labelText = juce::String(freq, 1) + " Hz, " + (noteName.isNotEmpty() ? "note: " + noteName : "")
        + ", " + juce::String(dB, 1) + " dB";

    return labelText;
}

juce::String SpectrogramComponent::drawTempogramTooltip(float dB, const int imgY, const int imageHeight)
{
    if (imageHeight <= 1 || tempoMaxBPM <= tempoMinBPM)
        return "BPM: -, " + juce::String(dB, 1) + " dB";

    const double logMin = std::log10(std::max(1.0f, tempoMinBPM));
    const double logMax = std::log10(std::max(tempoMinBPM + 1.0f, tempoMaxBPM));

    const double fracBottomToTop = 1.0 - (double)imgY / (double)(imageHeight - 1);
    const double logBpm = logMin + fracBottomToTop * (logMax - logMin);
    const double bpm = std::pow(10.0, logBpm);

    const double secPerBeat = 60.0 / std::max(1e-9, bpm);

    return juce::String(bpm, 1) + " BPM  (" + juce::String(secPerBeat * 1000.0, 0) + " ms/beat),  "
        + juce::String(dB, 1) + " dB";
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
    switch (currentMode)
    {
        case SpectrogramMode::Mel:
            paintMelYAxis(g, width, imageHeight);
            break;
        case SpectrogramMode::MelPlus:
            paintMelYAxis(g, width, imageHeight);
            break;
        case SpectrogramMode::MFCC:
            paintMFCCYAxis(g, width, imageHeight);
            break;
        case SpectrogramMode::Chroma:
            paintChromaYAxis(g, width, imageHeight);
            break;
        case SpectrogramMode::LinearWithCentroid:
            paintSTFTYAxis(g, width, imageHeight);
            break;
        case SpectrogramMode::LinearPlus:
            paintSTFTYAxis(g, width, imageHeight);
            break;
        case SpectrogramMode::FourierTempogram:
            paintTempoYAxis(g, width, imageHeight);
            break;
    
        default:
            paintSTFTYAxis(g, width, imageHeight);
            break;
    }

    // draw y axis (note)
    const bool modeSupportsNoteAxis =
        (currentMode == SpectrogramMode::Linear) ||
        (currentMode == SpectrogramMode::LinearPlus) ||
        (currentMode == SpectrogramMode::LinearWithCentroid) ||
        (currentMode == SpectrogramMode::Mel) ||
        (currentMode == SpectrogramMode::MelPlus);
    if (showNoteCAxis)
    {
        paintNoteYAxis(g, modeSupportsNoteAxis);
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

        // map GUI coords -> spectrogram image coords
        const auto bounds = getLocalBounds();
        const int  imgW = spectrogramImage.getWidth();
        const int  imgH = spectrogramImage.getHeight();
        if (imgW <= 0 || imgH <= 0)
            return;

        // X
        const int colX = juce::jlimit(0, imgW - 1,
            (mousePosition.x - bounds.getX()) * imgW / juce::jmax(1, bounds.getWidth()));

        // Y
        const int imgY = juce::jlimit(0, imgH - 1,
            (mousePosition.y - bounds.getY()) * imgH / juce::jmax(1, bounds.getHeight()));

        // how many columns actually exist
        const int cols = (int)dBBuffer.size();
        if (cols <= 0)
            return;

        if ((int)imgColAge.size() != imgW) return;
        const int age = imgColAge[colX];
        if (age < 0) return;
        if (age >= cols) return;

        const int dBIndex = cols - 1 - age;
        jassert(dBIndex >= 0 && dBIndex < cols);

        const auto& col = dBBuffer[dBIndex];
        const int colH = (int)col.size();
        if (imgY < 0 || imgY >= colH)
            return;
        // read value
        float dB = col[imgY];
        float maxFreq = sampleRate / 2.0f;
        float freq = 0.0f;

        juce::String labelText;

        switch (currentMode)
        {
            case SpectrogramMode::Mel:
                labelText = drawMelTooltip(dB, imgY, freq);
                break;
            case SpectrogramMode::MelPlus:
                labelText = drawMelTooltip(dB, imgY, freq);
                break;
            case SpectrogramMode::MFCC:
                labelText = drawMFCCTooltip(dB, imgY, imageHeight);
                break;
            case SpectrogramMode::Chroma:
                labelText = drawChromaTooltip(dBIndex, imgY, imageHeight);
                break;
            case SpectrogramMode::LinearWithCentroid:
                labelText = drawSTFTTooltip(dB, imgY, freq);
                break;
            case SpectrogramMode::LinearPlus:
                labelText = drawSTFTTooltip(dB, imgY, freq);
                break;
            case SpectrogramMode::FourierTempogram:
                labelText = drawTempogramTooltip(dB, imgY, imageHeight);
                break;

            default:
                labelText = drawSTFTTooltip(dB, imgY, freq);
                break;
        }

        // Draw fixed box under legend bar (top right)
        const int boxWidth = 240;
        const int boxHeight = 20;
        const int padding = 5;

        int boxX = getWidth() - boxWidth - padding - 6;
        int boxY = padding;  // below legend

        g.setColour(juce::Colours::black.withAlpha(0.6f));
        g.fillRect(boxX, boxY, boxWidth, boxHeight);

        g.setColour(juce::Colours::white);
        g.setFont(12.0f);
        g.drawText(labelText, boxX, boxY, boxWidth, boxHeight, juce::Justification::centredLeft);
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

void SpectrogramComponent::initFourierTempogram()
{
    noveltySamplePeriod = (hopSize > 0) ? (double)hopSize / sampleRate : 0.0;
    {
        noveltyWinLen = (noveltySamplePeriod > 0.0) ? (int)std::round(wantWinSec / noveltySamplePeriod) : 0;
        noveltyWinLen = juce::jlimit(8, 8192, noveltyWinLen);
        noveltyRingSize = (int)std::round(60.0 / juce::jmax(1e-9, noveltySamplePeriod)); // 60s should be enough
        noveltyRing.assign(juce::jmax(1, noveltyRingSize), 0.0f);
        noveltyWrite = 0;
        noveltyWin.resize(noveltyWinLen);
        for (int i = 0; i < noveltyWinLen; ++i)
            noveltyWin[i] = 0.5f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / juce::jmax(1, noveltyWinLen - 1));
        lastMagSpec.assign(fftSize / 2, 0.0f);
        lastCompSpec.assign(fftSize / 2, 0.0f);
        haveLastMag = false;
    }
    lastFourierTempoY = -1;
}

float SpectrogramComponent::fourierTempoPrior(float bpm) const
{
    // Log-normal prior (Gaussian in log2 space)
    const float mu = juce::jmax(1.0f, tempoPriorStartBPM);
    const float sigma = juce::jmax(1e-3f, tempoPriorSigmaLog2);
    const float z = std::log2(juce::jmax(1e-6f, bpm / mu)) / sigma;
    return std::exp(-0.5f * z * z);
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

void SpectrogramComponent::setShowNoteCAxis(bool shouldShow)
{
    if (showNoteCAxis == shouldShow) return;
    showNoteCAxis = shouldShow;
    repaint();
}

void SpectrogramComponent::resized()
{
    spectrogramImage = juce::Image(juce::Image::RGB, getWidth(), getHeight(), true);
    imgColAge.assign(spectrogramImage.getWidth(), -1);
}