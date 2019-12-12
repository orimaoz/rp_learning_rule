% Demo of RP model learning rule from the paper:
% Learning probabilistic representations with randomly connected neural circuits
% (preprint: https://www.biorxiv.org/content/10.1101/478545v1.abstract)
% Ori Maoz, 2019. contact: orimaoz@gmail.com
%
% This is a simple demonstration of the learning rule presented in the paper, used to train an RP model.
% this code requires the maxent_toolbox package for MATLAB, which is available from:
% https://orimaoz.github.io/maxent_toolbox/
% For training RP models with classical maximum-entropy methods, please refer to the toolbox code.
%
% STEP 1: load data

% the sample data provided here are actual recording from groups of 50 cells from V1/V2 or PFC, as was used in the
% original paper. The data provided here was binned at 20ms and taken without context or particular order from various 
% parts of the experiment. It is provided here mostly so that we have "real" data from groups of neurons to fit, 
% but please don't try to over-analyze it. If what you care about is the data itself, refer to the paper where links 
% to the "proper" versions of the neural recordings can be obtained.

% data = load('data/prefrontal_50.mat');
data = load('data/visual_50.mat');


% number of samples to train and test the data
nsamples_train = 100000;     % number of samples to learn from
nsamples_test = 100000;


% randomly divide data intO train/test
raster = data.raster(:, randperm(size(data.raster,2)));  % shuffle raster
ncells = size(raster,1);
xtrain = raster(:,1:nsamples_train);                     % data used to train the model 
xtest = raster(:,(nsamples_train+1):(nsamples_train+nsamples_test)); % data used to evaluate predictions


%%
% STEP 2: initialize and train the model

% initialize an RP model
num_projections = 1000; % number of random projections used in the RP model
model = maxent.createModel(ncells,'rp', 'nprojections', num_projections);

% train it on the data.
% the core presented here is a simple proof-of-concept. Better results can be achieved by properly adjusting
% the learning rate across training epochs, and possibly using a better optimizer on the gradient
% (here this is basic stochastic gradient descent)
learning_rate = 0.01;
disp('training model...');
model = learningRule(model,xtrain, 'epochs', 100, 'learning_rate', learning_rate, 'noise', 0.03);



%%
% STEP 3: evaluate predictions (probabilities of individual activity patterns)


% the model that the MCMC solver returns is not normalized. If we want to compare the predicted and actual probabilities
% of individual firing patterns, we will need to first normalize the model. We will use the wang-landau algorithm for
% this. We chose parameters which are less strict than the default settings so that we will have a faster runtime.
% for a more proper evaluation, please use maxent.normalize() which runs this with more strict arguments.
disp('Normalizing model so it can be evaluated...');
model = maxent.wangLandau(model,'binsize',0.04,'depth',15);
%%
% the normalization factor was added to the model structure. Now that we have a normalized model, we'll use it to
% predict the frequency of activity patterns. We will start by observing all the patterns that repeated at least twice
% (because a pattern that repeated at least once may grossly overrepresent its probability and is not meaningful in this
% sort of analysis)
limited_empirical_distribution = maxent.getEmpiricalModel(xtest,'min_count',2);

% get the model predictions for these patterns
model_logprobs = maxent.getLogProbability(model,limited_empirical_distribution.words);

% plot on a log scale
figure
color_model = [0.8500, 0.3250, 0.0980];
sct = scatter(limited_empirical_distribution.logprobs,model_logprobs,20,'Marker','o','MarkerFaceColor',color_model,'MarkerEdgeColor',color_model);
sct.MarkerFaceAlpha = .5;
sct.MarkerEdgeAlpha = .5;
hold on;
minval = min(limited_empirical_distribution.logprobs);
plot([minval 0],[minval 0],'-r', 'LineWidth', 1, 'LineStyle', '--');  % identity line
xlabel('empirical pattern log frequency');
ylabel('predicted pattern log frequency');
title(sprintf('activity pattern frequency in %d cells',ncells));

%%
% STEP 4: evaluate predictions (k-synchrony plots)

% we start by sampling from the model. we will compare these 
disp('sampling from model...');
sampled_x = maxent.generateSamples(model,nsamples_test);

% get the k-synchrony of the model and the data. We'll use an uninitialized ksync model as a shortcut.
% the alternative would be to sum up the spikes in each pattern, and then take a histogram.
msync = maxent.createModel(ncells,'ksync');
synchrony_empirical = maxent.getEmpiricalMarginals(xtest,msync);
synchrony_model = maxent.getEmpiricalMarginals(sampled_x,msync);

% plot the synchrony of the empirical data and the synchrony of what we sampled in the model.
figure
hold on
color_empirical = 'black';
color_model =    [0.8500, 0.3250, 0.0980]	;
plot(0:ncells,synchrony_empirical,'LineStyle','none','MarkerSize',8,'Marker','d','MarkerFaceColor',color_empirical,'MarkerEdgeColor',color_empirical);
plot(0:ncells,synchrony_model,'LineStyle', 'none','MarkerSize',8,'Marker','d','MarkerFaceColor',color_model,'MarkerEdgeColor',color_model);
set(gca,'yscale','log');
legend({'empirical', 'model'});
xlabel('population synchrony, K');  
ylabel('p(K)');
title(sprintf('Spiking synchrony prediction in %d cells',ncells));




