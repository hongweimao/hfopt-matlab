%% Train RNN to generate recorded firing rate profiles
%
%  When trained, activities of M out of the N (N >> M) recurrent units in the network 
%  match the firing rate profiles of M simultaneously recorded neurons provided to 
%  the network as desired outputs.
%
%  5-fold cross validation is used to prevent overfitting. Training is repeated 5 times
%  with each of the 5 data partitions as the testing dataset.
%
%  By H.M., 6/29/2022. (Adapted from the sinewave_gen example.)

function train_firing_rate_profile_generator_CV(subject, inputType, irun)

%% start up
close all; clc; % clear
addpath(genpath('./../../'));

if ~exist('irun', 'var')
    irun = 1;
end

if ~exist('subject', 'var')
    subject = 'monkeyC';
end

if ~exist('inputType', 'var')
    inputType = 'Gau';      % Gau: inputs have a bell-shaped profile (Gaussian PDF) over time
end


%% PREAMBLE
rng_type    = 'default';   % default (seed 0); shuffle (current time); 
% globalRandStream = RandStream('threefry4x64_20', 'Seed', 0);
globalRandStream = RandStream('threefry4x64_20', 'Seed', 'shuffle');
globalRandStream.Substream = irun * 100;    % set substream of rng
RandStream.setGlobalStream(globalRandStream);
% randStream = globalRandStream;  % returns the current settings of the RNG

if isempty(gcp('nocreate'))
    parpool(8);
end


%%  Prepare Data
dt          = 0.005;  % sec
tau         = 0.020;  % sec
numUnitRNN  = 1000;   % N: total number of units in the recurrent layer

Data      = prepare_data(subject);
numTarget = Data.numTarget;     % number of trials per block
numBlock  = Data.numBlock;      % number of blocks or repetitions
numTrial  = numBlock * numTarget;  % total number of trials

Inputs      = Data.(sprintf('input%s', inputType));
numInput    = size(Inputs{1}, 1);
Targets     = Data.fr;
numOutput   = size(Targets{1}, 1);
ICs         = Data.ic;


%% 5-fold cross-validation partitions: 
%   Training:Validation:Test = 60:20:20
numCVFold = 5;
blockIdx = cell(numCVFold, 3);
cvPartition = cvpartition(numBlock, 'KFold', numCVFold);
for ifold = 1:numCVFold
    tmpIdx   = find(training(cvPartition, ifold));  % block indices for training and validation
    validIdx = randperm(length(tmpIdx), round(numBlock * 0.2));
    trainIdx = setdiff(1:length(tmpIdx), validIdx);

    blockIdx{ifold, 1} = tmpIdx(trainIdx);
    blockIdx{ifold, 2} = tmpIdx(sort(validIdx));
    blockIdx{ifold, 3} = find(test(cvPartition, ifold));   
end

trialMask = repmat({false(1, numTrial)}, numCVFold, 3);
for i = 1:numCVFold
    for j = 1:3
        blockIndices = blockIdx{i, j};
        for iblock = blockIndices
            trialIndices = (iblock-1)*numTarget + [1:numTarget];
            trialMask{i, j}(1, trialIndices) = true;
        end
    end
end


%% per-trial initial conditions
%   last data sample before the start of the trial of recorded units (Data.ic)
% + random ICs for other hidden units in the RNN
R0s         = ICs;
invalidInd  = find(R0s > 0.9);
R0s(invalidInd) = 0.9;
X0s         = atanh(R0s);  % r = tanh(x), so x(0) = atanh(r(0))

% calc ICs for remaining units of RNN
gammaParams = gamfit(mean(X0s, 2));  % est. distribution of embedded unit ICs
x0 = gamrnd(gammaParams(1), gammaParams(2), numUnitRNN-numOutput, 1);  % generate ICs for remaining units
        
% gammaParams = gamfit(std(X0s, 0, 2));
% std0 = gamrnd(gammaParams(1), gammaParams(2), N-numOutput, 1);
std0 = 0.05 * ones(numUnitRNN-numOutput, 1);

% ICs for remaining units
hiddenX0s = repmat(x0, 1, numTrial) + ...
            repmat(std0, 1, numTrial) .* randn(numUnitRNN-numOutput, numTrial); 
rnnICs    = [X0s; hiddenX0s];


%% ------------------------------------------------------------------------
%  Create Network Parameters for training
save_name = [subject];
save_path_part = './networks/';

gValue = 1.5;  % linspace(0.9, 1.5, n_ind_params)
aValue = 1;

do_debug = false;
if ~do_debug
    do_parallel_network = true;     do_parallel_objfun = true;
    do_parallel_gradient = true;    do_parallel_cg_afun = true;
else
    do_parallel_network = false;    do_parallel_objfun = false;
    do_parallel_gradient = false;   do_parallel_cg_afun = false;
end


%% ------------------------------------------------------------------------
%  Initialize network
net_type        = 'rnn-trials';
N               = numUnitRNN;   % Defined above for dynamical noise.
M               = numOutput;
layer_sizes     = [numInput N N M];
layer_types     = {'linear', 'recttanh', 'linear'};
objective_function = 'sum-of-squares';
numconn         = round(N * 0.05);    
network_noise   = 0.01;  % don't forget multiplication by sqrt(dt)
    
g_by_layer      = zeros(3,1);  % Should be size 3, (I->N) (N->N) (N->M)   
g_by_layer(1)   = 1.0 / numInput;
g_by_layer(2)   = gValue;    
g_by_layer(3)   = 0.0;

save_every      = 10;
mu              = 0.03;				

lambda          = 0.0002;           % lambda and minibatch size are intimately related.
minibatch_size  = numTarget * 1;

% These are the two main stopping criteria.  Useful to set well, if
% there's a bunch of sims
max_hf_iters    = 1000;			
max_hf_failures = 1000;
max_consec_test_increase = 10;
objfuntol       = 1e-8;  % use 1e-9 for actural training
    
% Training related stuff.
do_learn_biases = 1;
do_init_state_biases_random = 0;
do_learn_state_init = 0;
do_init_state_init_random = 0;  % zero is probably good init for oscillatory behavior

maxcgiter       = 100;
mincgiter       = 10;
cgepsilon       = 1e-6;
    
weight_cost     = 1e-5;
cm_fac_by_layer = [1.0 1.0 0.0]';  % cost mask layer fac
mod_fac_by_layer = [1.0 1.0 0.0]';

disp('Initializing new network with a static output layer.');
net0 = init_rnn_recurr_as_outputs(layer_sizes, layer_types, g_by_layer, objective_function, ...
	   'cmfacbylayer', cm_fac_by_layer, 'modmaskbylayer', mod_fac_by_layer, ...
       'numconn',  numconn, 'tau', tau, 'dt', dt, 'netnoisesigma', network_noise, 'mu', mu, 'transfunparams', aValue, ...
       'dolearnbiases', do_learn_biases, 'doinitstatebiasesrandom', do_init_state_biases_random, ...
       'dolearnstateinit', do_learn_state_init, 'doinitstateinitrandom', do_init_state_init_random);
net0.layers(2).initLength = 1;		% burn at the beginning
    

% additional regularization terms
net0.frobeniusNormRecRecRegularizer = 1e-5;
    
net0.firingRateSumSquares.weight = 1e-3;
net0.firingRateSumSquares.mask   = [zeros(M, 1); ones(N-M, 1)];

% use trial ICs
net0.trialICs.doUseTrialICs = true;


%% Train the Networks
allnets = cell(1, numCVFold);
for icv = 1:numCVFold  
    close all;

    net = net0;  % start with the same initialization of net for all CV folds

    % Create the directories, if they don't already eixst.   
    save_path_end_dir = [save_name '_' inputType '_Run' num2str(irun) '_CV' num2str(icv)];
    save_path = [save_path_part save_path_end_dir];

    if ( ~exist(save_path, 'dir') )
        mkdir(save_path);
    end

    % prepare data
    inputsTrain     = Inputs(trialMask{icv, 1});
    inputsValid     = Inputs(trialMask{icv, 2});
    targetsTrain    = Targets(trialMask{icv, 1});
    targetsValid    = Targets(trialMask{icv, 2});
    net.trialICs.ICs{1} = rnnICs(:, trialMask{icv, 1});
    net.trialICs.ICs{2} = rnnICs(:, trialMask{icv, 2});
    
    do_plot = 0;  % Somehow after running a bunch of these, I beleive the figures are screwing up matlab so everything comes to a screaming halt.
    
    % sim_params.randStream    = randStream;
    sim_params.doNewNet      = true;
    sim_params.doLoadOldNet  = false;
    sim_params.doLoadOldNetFromFile = false;
    sim_params.doOneOutput   = false;
    sim_params.oldNetLoadPath = '';
    sim_params.oldNetPackageName = '';
    sim_params.doResetToOriginalTheta = false;
    sim_params.doMatrixExponential = false;
    sim_params.trainingDataString = 'simulated';
    sim_params.g             = gValue;
    sim_params.alpha         = aValue;
    sim_params.trMaskTrain   = trialMask{icv, 1};
    sim_params.trMaskValid   = trialMask{icv, 2};
    sim_params.trMaskTest    = trialMask{icv, 3};
    sim_params.inputType     = inputType;
    sim_params.nInput        = numInput;
    sim_params.nTotalUnits   = numUnitRNN;
    sim_params.nEmbeddedUnits = numOutput;
    
    
    % Now train the network to do the task.  
    [opttheta, objfun_train, objfun_test, stats] = hfopt2(net, inputsTrain, targetsTrain, inputsValid, targetsValid, ...
						  'maxhfiters', max_hf_iters, 'maxcgfailures', max_hf_failures, 'maxconsecutivetestincreases', max_consec_test_increase, ...
						  'S', minibatch_size, 'doplot', do_plot, 'objfuntol', objfuntol, ...
						  'initlambda', lambda, 'weightcost', weight_cost, ...
						  'highestmaxcg', maxcgiter, 'lowestmaxcg', mincgiter, ...
						  'cgtol', cgepsilon, 'nettype', net_type, ...
                          'samplestyle', 'random_rows', ...
						  'savepath', save_path, 'filenamepart', save_name, 'saveevery', save_every, ...
                          'paramstosave', sim_params, ...
                          'doparallelnetwork', do_parallel_network, ...
                          'doparallelobjfun', do_parallel_objfun, ...
                          'doparallelgradient', do_parallel_gradient, ...
                          'doparallelcgafun', do_parallel_cg_afun);

end 


%% cleanup
rmpath(genpath('./../../'));
