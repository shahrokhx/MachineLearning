function driver(lowRank)


% Use real data:
load ('movie_data');
rateMatrix = train;
testMatrix = test;

% Global SVD Test:
for l=1:size(lowRank, 2)
    tic;
    [U, V] = myRecommender(rateMatrix, lowRank(l));
    logTime = toc;
    
    trainRMSE = norm((U*V' - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));
    testRMSE = norm((U*V' - testMatrix) .* (testMatrix > 0), 'fro') / sqrt(nnz(testMatrix > 0));
    
    fprintf('SVD-%d\t%.4f\t%.4f\t%.2f\n', lowRank(l), trainRMSE, testRMSE, logTime);
end


end