function rdMap = compute_range_doppler(adcData)

% adcData: [chirps x RX x samples]

% Use RX1 only (simplest & stable)
data = squeeze(adcData(:,1,:));

% Windowing
winRange = hann(size(data,2))';
winDopp  = hann(size(data,1));

% Range FFT
rangeFFT = fft(data .* winRange, [], 2);

% Doppler FFT
dopplerFFT = fftshift(fft(rangeFFT .* winDopp, [], 1), 1);

rdMap = dopplerFFT;
end
