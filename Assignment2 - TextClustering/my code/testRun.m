%% MY TEST SUIT
%  Developed by: Shahrokh Shahi (sshahi3)
%  I wrote this simple code to check the outputs of my clustering function


%% Initialization
clc
clear
close all

format short g
format compact

%% Hard-coded Values & Loading Data

MAX_IT = 100;

load('data');
T = X(:,1:100);
label = X(:,101);

%% Run Loop
acc = zeros(MAX_IT, 1);

for iter = 1 : MAX_IT
    index     = mycluster(T,4);
    acc(iter) = AccMeasure(label,index);
end

%% Plot Outputs
figure(1);
clf;
hold on;
grid on;
norm=histfit(acc,floor(MAX_IT/10),'normal');
[mean, var] = normfit(acc)
line([mean, mean], ylim, 'Color', [0, .6, 0], 'LineWidth', 3);
