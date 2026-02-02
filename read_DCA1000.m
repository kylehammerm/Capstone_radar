function adcData = read_dca1000_bin(filePath, numRX, numADCSamples, numChirps)

fid = fopen(filePath,'r');
raw = fread(fid,'int16');
fclose(fid);

% Convert to complex I/Q
iq = raw(1:2:end) + 1j*raw(2:2:end);

samplesPerChirp = numRX * numADCSamples;
totalChirps = floor(length(iq) / samplesPerChirp);

iq = iq(1:totalChirps*samplesPerChirp);
iq = reshape(iq, samplesPerChirp, totalChirps);

% Reshape to [chirp, RX, ADC]
adcData = zeros(totalChirps, numRX, numADCSamples);

for c = 1:totalChirps
    tmp = reshape(iq(:,c), numRX, numADCSamples);
    adcData(c,:,:) = tmp;
end

adcData = adcData(1:numChirps,:,:);  % limit to one frame
end
