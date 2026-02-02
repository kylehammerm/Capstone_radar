clc; clear; close all;

%% USER SETTINGS
dataFolder = 'C:\mmwave_captures';   % <-- SAME folder as mmWave Studio output
numRX = 4;
numADCSamples = 256;
numChirps = 64;
bytesPerSample = 4;  % int16 I + int16 Q

%% Figure
figure('Name','Near-Live Doppler');
colormap jet;

processedFiles = {};

disp('Waiting for new mmWave captures...');

while true
    files = dir(fullfile(dataFolder,'*.bin'));
    if isempty(files)
        pause(0.1);
        continue;
    end

    % Get newest file
    [~, idx] = max([files.datenum]);
    filePath = fullfile(dataFolder, files(idx).name);

    if ismember(filePath, processedFiles)
        pause(0.1);
        continue;
    end

    fprintf('Processing %s\n', files(idx).name);

    adcData = read_dca1000_bin(filePath, numRX, numADCSamples, numChirps);

    rdMap = compute_range_doppler(adcData);

    imagesc(20*log10(abs(rdMap)));
    xlabel('Range Bin');
    ylabel('Doppler Bin');
    title('Near-Live Range–Doppler');
    colorbar;
    drawnow;

    processedFiles{end+1} = filePath;
end
