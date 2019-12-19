function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with size(pixels, 1) rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.



    % constant parameters
    MAX_ITER    = 100;
    COLOR_RANGE = 255;
    EPS         = 1e-5;
    
        
    
    % initialization
    [nRows, ~]  = size (pixels);
    centroid    = floor(rand(K    ,3) * COLOR_RANGE + 1);
    class       = floor(rand(nRows,1) * K           + 1);
%   class       = ones (nRows,1);
%   class       = ones(nRows,1) * K;    

    iter = 0;
    cost = zeros(MAX_ITER,1);
    flag = true;
    while (flag)
        iter = iter + 1;
        flag = false   ;  
        % if there is no update in class, flag will stay false and breaks
        % the loop
        
        currentCost = 0;
        % loop over all pixels
        for pix = 1 : nRows
            index = 1;
            currentDistance = norm(centroid(index,:) - pixels(pix,:));
            
            for i = 1 : K
                newDistance = norm(centroid(i,:) - pixels(pix,:));
                if (newDistance < currentDistance)
                    currentDistance  = newDistance;
                    index            = i;
                end
            end
            
            currentCost = currentCost + currentDistance;
            
            % check if there is an update, one update is enough for one
            % more iteration
            if (class(pix)~= index)
                class(pix) = index;
                flag       = true;
            end
        end
        
        % check updates in cost (loss) function values
        cost(iter) = currentCost;
        if iter > 1
            if abs(cost(iter) - cost(iter-1)) < EPS
                flag = false;
            end
        end
        
        % updating centroids
        for i = 1 : K
            classIndex = find(class == i);
            if ~isempty(classIndex)  % to treat empty cluster(s) exceptions
                centroid(i,:) = mean(pixels(classIndex,:));
            end
        end
        
        % break if maximum iteration reached
        if iter == MAX_ITER
            warning ('K-means reached max iterations');
            break;
        end
    end
    fprintf ('K-means   clustering accomplished after %2i iterations.\n',iter)
end

