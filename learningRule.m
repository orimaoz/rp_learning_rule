
% Demonstrates the use of the biological learning rule on an RP model from the paper:
% Learning probabilistic representations with randomly connected neural circuits
% (preprint: https://www.biorxiv.org/content/10.1101/478545v1.abstract)
% Ori Maoz, 2019. contact: orimaoz@gmail.com
%
%
% Input arguments:
%      model - initialized RP model (as created by the maxint_toolbox)
%      x - input raster to train from, in dimensions (ncells x nsamples)
% optional named arguments (in the form of key, value pairs):
%      epochs        - number of training epochs to use
%      learning_rate - learning rate to apply in the gradient descent. This is simple demo code and the learning
%                      rate is not adjusted, but you can call the function several times with decreasing learning rates.
%      order         - order to traverse the samples each epoch. can be either 'random' (default) or 'sequential' to 
%                      go over them one by one in their original order
%      noise         - amount of noise used to generate the echo pattern. This is the probability for each bit
%                      in the input pattern to be flipped.
%
%  ouput: the trained model
function model = learningRule(model, x, varargin)


[ncells,nsamples] = size(x);

    
% parse our optional arguments
p = inputParser;
addOptional(p,'epochs',100,@isnumeric); % number of steps to train
addOptional(p,'learning_rate',0.01,@isnumeric);  % base size of step (this is scaled down as we progress)
addOptional(p,'order','sequential');    % 'sequential' to use samples in order or 'random' for random order
addOptional(p,'noise',0.015);    % 'sequential' to use samples in order or 'random' for random order



p.parse(varargin{:});
epochs = p.Results.epochs;
learning_rate = p.Results.learning_rate;
order = p.Results.order;
noise_level = p.Results.noise;


% check if we should go randomly or sequentially through the input patterns
sequential_order = (strcmpi(order,'sequential'));

nfactors = maxent.getNumFactors(model);

% we use this to keep track of the target function to see that we don't diverge 
likelihood_geometric = 0.5;
likelihood_decay = 1 - 0.0001;
TARGET_FUNCTION_DIVERGE_THRESH = 1000;

% we use this to keep track of the gradient so that we can scale it for better step sizes
gradient_norm_geometric = nfactors;
gradient_norm_decay = 1 - 0.0001;

num_steps = epochs * nsamples;
for curr_step = 1:num_steps    

    % take inputs one by one or randomly
    if sequential_order
        input_pattern = x(:,mod(curr_step,nsamples)+1);   
    else
         input_pattern = x(:,randi(nsamples));   
    end
    
    % make the echo pattern as a noisy version of the input pattern
    randbit = rand(ncells,1) < noise_level;
    echo_pattern = xor(input_pattern,randbit);

    % get the activity of the intermediate layer for the vector and flipped vector
    vx = model.A * input_pattern > model.threshold';
    vflipped = model.A * echo_pattern > model.threshold';
    intermediate_layer_diffs = (vx - vflipped);

    % This is the actual learning rule - the gradient is the differences of the intermediate layer
    % scale by the different activities of the output layer
    expdiffs = exp(dot(intermediate_layer_diffs, maxent.getFactors(model))/2);
    K = expdiffs;                     % target function 
    dK = (intermediate_layer_diffs .* expdiffs);  % gradient
    
     
    % check that the gradient descent is not diverging
    likelihood_geometric = likelihood_geometric * likelihood_decay + K * (1-likelihood_decay);
    if(any(isnan(dK)) || (likelihood_geometric > TARGET_FUNCTION_DIVERGE_THRESH))
        ME = MException('MerpFlowDescent:NaNSolution', ...
            'NaN/diverge encountered while solving, step size too big?');
            throw(ME);        
    end
    
    if (mod(curr_step,20000)==0)
        fprintf('%d/%d: gradient norm %f\n',curr_step,num_steps,gradient_norm_geometric);
    end
       
    
    % remember the norm of the gradient.
    % an estimation of this will be used to adjust the step size
    gradient_norm_geometric = gradient_norm_geometric * gradient_norm_decay + dot(dK,dK) * (1-gradient_norm_decay);
    
    % adjust the the learning rate by the square root of the (decaying) average norm of the gradient.
    % this is not required by the learning rule but has been empirically observed to give better results.
    % if you change the optimizer used to apply the gradient, get rid of this.
    step_size = learning_rate ./ sqrt(gradient_norm_geometric);    

    % Vanilla gradient descent. You can change this to use your favorite gradient-based optimizer.
    model = maxent.setFactors(model,maxent.getFactors(model) - step_size .* dK');

    
end

end