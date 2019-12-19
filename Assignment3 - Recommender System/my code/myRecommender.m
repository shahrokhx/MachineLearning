function [ U, V ] = myRecommender( rateMatrix, lowRank )
    
    % Please type your name here:
    name = 'Shahi, Shahrokh';
    disp(name); % Do not delete this line.

    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;
    
    % ---------------------------- Parameters --------------------------- %
    maxIter      = 1000  ;   % maximum number of itrations
    learningRate = 0.0005;   % mu
    regularizer  = 0.0008;   % lambda
    maxErr       = 0.8   ;   % maximum absolute error
    maxErrRel    = 5e-5  ;   % maximum relative error
    
    % ----------------------------- MY CODE ----------------------------- %
    M_nonZeros = rateMatrix > 0;
    sqNonZeros = sqrt(nnz(M_nonZeros));
    
    err_prev = 1;
    % Gradient Descent:
    
    for iter = 1 : maxIter
        % mu = learningRate * (1-1/(maxIter-iter+1));
        
        R = (rateMatrix - U*V') .* M_nonZeros;
        U = U  +  2*learningRate*R *V  -  2*regularizer*U ;
        V = V  +  2*learningRate*R'*U  -  2*regularizer*V ;
        
        % convergency check & avoiding over-fitting
        err = norm((U*V' - rateMatrix) .* M_nonZeros, 'fro') / sqNonZeros;
        if (err < maxErr) || (abs(err-err_prev)/err_prev < maxErrRel)
            break;
        end
        err_prev = err;
    end
    % ------------------------------------------------------------------- %
    
end