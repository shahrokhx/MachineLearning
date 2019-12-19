function [ class ] = mycluster( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

% YOUR IMPLEMENTATION SHOULD START HERE!

% hard-coded parameters
MAX_IT = 200;
EPS    = 100 * eps;

% input parameters
[nDocs, nWords] = size(bow);
nClusters = K;

% initializing the mixture coefficient p(c)= \pi_c 
pi_c = rand(nClusters, 1);
pi_c = pi_c ./ sum(pi_c);   % normalizing

% initialization \mu
mu = rand(nWords, nClusters);
mu = mu ./ sum(mu);  
% mu = mu ./ repmat(sum(mu), nWords, 1);

% initializing \gamma
gamma = zeros(nDocs, nClusters);
gamma_prev = gamma;

% iterations 
for iter = 1 : MAX_IT
    % ----------------------------- E-step ------------------------------ %
    % p(Di) = sum(p(Di|c)p(c))
    p_Di = zeros(nDocs, 1);
    p_Di_c = ones(nDocs, nClusters);
    
    for i = 1 : nDocs
        for c = 1 : nClusters
            for j = 1 : nWords
                p_Di_c(i,c) = p_Di_c(i,c) * mu(j,c) ^ bow(i,j);
            end
            p_Di(i) = p_Di(i) + p_Di_c(i,c) * pi_c(c);
        end
        
        for c = 1 : nClusters
            gamma(i,c) = pi_c(c) * p_Di_c(i,c) / p_Di(i);
        end 
    end

    % ----------------------------- M-step ------------------------------ %
    % mu = X / Y:
    X = (gamma' * bow)';
    Y  = zeros(1, nClusters);
    for c = 1 : nClusters
        for i = 1 : nDocs
            for l = 1 : nWords
                Y(c) = Y(c) + gamma(i,c) * bow(i,l);
            end
        end
    end
    % updating mu
    mu = X ./ repmat(Y, nWords, 1);
    
    % updating p(c)
    pi_c = sum(gamma) ./ nDocs;
    
    % ------------------------------------------------------------------- %
    % checking convergency
    % the convergency check is currently disabled, since it will be 
    % converged quickly (usually within <5 iterations. But it can easily
    % be used by uncommenting the following lines:
    
    % if sum(sum(gamma-gamma_prev)) < EPS
    %    break
    % end
    % gamma_prev = gamma;
    % ------------------------------------------------------------------- %
end
% fprintf('clustering converged at iteration = %3d\n',iter);

% class indices (the index of maximums)
[~, class] = max(gamma,[],2);

end