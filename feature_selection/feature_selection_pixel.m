% Select features (pixels) based on boys and girls
clear; 
close all; 

%% load boys and girls dataset
% new_boys.mat and new_girls.mat are larger dataset which covers all students'
% faces, including 461 boys and 176 girls. The original images are stored in
% the folder boys and girls. The script generating these two data sets is
% 'generate_new_dataset.m'.

% boys.mat and girls.mat are the old and smaller dataset which only
% includes students in 2014 Fall class(the images are those only having 
% numbers as names in the two folders). There are 116 boys and 63 girls.

% load('boys');
% boys_data = double(data) / 255;
% load('girls');
% girls_data = double(data) / 255;
% d = size(data, 1);

load('new_boys');
boys_data = double(boyRaw) / 255;
load('new_girls');
girls_data = double(girlRaw) / 255;
d = size(girlRaw, 1);

H = 65; 
W = 65; 

figure; 
show_image(boyRaw', H, W); 
figure; 
show_image(girlRaw', H, W); 

input('press key to continous ...\n'); 

% Discretize into nbins
nbins = 50;
mi = zeros(d, 1);
for i = 1:d
    boys_pix_i_freq = histc(boys_data(i, :), linspace(0, 1, nbins));
    girls_pix_i_freq = histc(girls_data(i, :), linspace(0, 1, nbins));
    
    % Calculate joint distribution between feature i and label/class.
    joint_distr = [boys_pix_i_freq; girls_pix_i_freq];
    all_sum = sum(sum(joint_distr));
    % P(X, Y)
    joint_distr = joint_distr / all_sum;
    
    % Calculate marginal distribution of feature i and label/class.
    % P(X)
    feat_distr = sum(joint_distr, 1);
    % P(Y)
    class_distr = sum(joint_distr, 2);
    % P(X) * P(Y)
    cross_prod = class_distr * feat_distr;
    
    tmp = joint_distr .* log(joint_distr ./ cross_prod);
    % We define 0 * log 0 to be 0
    tmp(isnan(tmp)) = 0;
    mi(i) = sum(tmp(:));
end

% Visualize feature mutual information
figure; 
imagesc(reshape(mi, H, W))
colorbar
