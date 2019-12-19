%% CDA HW4
%  Developed by SHAHROKH SHAHI (sshahi3)

clc
clear
close all

%% Inputs
qList = [0.7, 0.9];

data = load('sp500.mat'); 
y = data.price_move;
n = length(y);
y = (3-y)/2;  % map to {1,2}

for iter = 1 : length(qList)
    %% Initialization
    q = qList(iter);

    p = [0.2, 0.8];
    A = [0.8, 0.2 ;
         0.2, 0.8];
    B = [  q, 1-q ;
         1-q,   q];

    alpha = zeros(n,2);
    beta  = zeros(n,2);

    alpha(1,1) = p(1) * B(1,y(1));
    alpha(1,2) = p(2) * B(2,y(1));
    beta (n,:) = [1, 1]; 

    %% Procedure

    % forward 
    for i = 1 : n-1
        alpha(i+1,1) = B(1,y(i+1))  *  (A(1,1)*alpha(i,1) + A(2,1)*alpha(i,2));
        alpha(i+1,2) = B(2,y(i+1))  *  (A(1,2)*alpha(i,1) + A(2,2)*alpha(i,2));
    end

    % backward
    for i = n : -1 : 2
        beta(i-1,1) = beta(i,1)*B(1,y(i))*A(1,1) + beta(i,2)*B(2,y(i))*A(1,2);
        beta(i-1,2) = beta(i,1)*B(1,y(i))*A(2,1) + beta(i,2)*B(2,y(i))*A(2,2);
    end

    % p_{joint}
    P = alpha .* beta;
    for i = 1 : n
        P(i,:) = P(i,:) / (P(i,1)+P(i,2));
    end
    prob = P(n);

    %% plots

    figure(1)
    hold on
    grid on
    style = {'-sr','--ob','-*k'};
    plot(1:n,P(:,1),style{mod(iter,3)})


    figure()
    hold on
    grid on
    c ='r';
    plot(1:n, P(:,1), ['--s',c], ...
         'MarkerSize'     , 10 , ...
         'MarkerFaceColor', c  , ...
         'MarkerIndices'  , 1:n, ...
         'LineWidth'      , 2   )

    bar(P(:,1),'FaceAlpha',0.3)

    % increasing fonts for PDF
    set(gca,'FontSize',20)
    xlabel('Week','FontSize', 30)
    ylabel('Conditional Probability','FontSize', 30)
    title(['q = ',num2str(q)],'FontSize', 30)

end