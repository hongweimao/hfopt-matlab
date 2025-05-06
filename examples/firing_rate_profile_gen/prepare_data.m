%% make data for RNN
%  prepare single-trial un-aligned data

function data = prepare_data(subject)

%% load data
% Data = load_monkey_data(subject, session);
dataDir = './data/';
fileName = sprintf('MonkeyC_subset.mat');
load([dataDir fileName]);

% prepare data for rnn
numTrial  = length(Data.Rate);  % total number of trials
numBlock  = max(Data.blockNo);  % number of blocks
numTarget = numTrial / numBlock;  % number of target/condition per block
dt        = Data.dt;


%% create input signals
cueAngle = 0:(360/numTarget):359;  % target direction in degrees for 2-D center-out reaching task
cueValue = [cosd(cueAngle); sind(cueAngle)];

% inputs - cosine-tuned
inputPDs = [10:(360/18):359]';  % 18 inputs with different PDs 
numInputPD = length(inputPDs);
inputValue = cosd(repmat(inputPDs, 1, numTarget) - repmat(cueAngle, numInputPD, 1));  % magnitude is cosine tuned

% inputs - linear-tuned
angDiff = repmat(cueAngle, numInputPD, 1) - repmat(inputPDs, 1, numTarget);
angDiff(angDiff < 0) = angDiff(angDiff < 0) + 360;  % values in range [0 360]
linearValue = 1 - 2*angDiff/360;  % magnitude changes linearly from 1 to -1 as angular difference between target direction and
                                  % PD increases from 0 to 360 degrees



Times    = cell(1, numTrial);
inputGau = cell(1, numTrial);
inputLin = cell(1, numTrial);

numGau = length(Data.rateEpochMu);
gaussOffsets = Data.rateEpochMu - Data.eventTime;


for i = 1:numTrial
    itgt = Data.targetNo(i);

    Times{1, i} = dt * [0 : (size(Data.Rate{i}, 2) - 1)];
    
    trialLandmarks = Data.landmarkTime(3:5, i)';
    trialGaussMu = gaussOffsets + trialLandmarks;

    gaussProfiles = nan(numGau, length(Times{1, i}));
    for igauss = 1:numGau
        tmp = normpdf(Times{1, i}, trialGaussMu(igauss), Data.rateEpochSigma(igauss));
        gaussProfiles(igauss, :) = tmp ./ max(tmp);
    end

    gauss_inputs = [];
    linear_inputs = [];
    for igauss = 1:numGau
        % gauss_inputs = [gauss_inputs; inputValue(:, itgt) * gauss_profiles(igauss, :)];  % value in [-1 1]
        gauss_inputs = [gauss_inputs; 0.5 + 0.5 * inputValue(:, itgt) * gaussProfiles(igauss, :)];  % value in range [0 1]
        
        linear_inputs = [linear_inputs; 0.5 + 0.5 * linearValue(:, itgt) * gaussProfiles(igauss, :)];
    end

    % inputGau{1, i} = [gauss_inputs; RawData.inputHoldA{i}];
    inputGau{1, i} = gauss_inputs;
    inputLin{1, i} = linear_inputs;
end


%% create output data for RNN
data.dt         = dt;
data.times      = Times;

data.inputGau   = inputGau;
data.inputLin   = inputLin;

data.fr         = Data.normRate;
data.ic         = Data.normIC;

data.numBlock   = numBlock;
data.numTarget  = numTarget;

data.landmarkTime = Data.landmarkTime;
data.landmarkType = Data.landmarkType;

% % % fileName = sprintf('rnnData_%s.mat', subject);
% % % save(fileName, 'data');

end

