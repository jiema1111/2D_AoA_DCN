%%
clc;
clear;
close all;
%% 估计仰角和方位角

derad = pi/180;      %角度->弧度
radeg = 180/pi;
N = 8;               % 阵元个数    
M = 1;               % 信源数目
K = 200;              % 快拍数

dd = 0.5;            % 阵元间距
d = 0: dd : (N-1)*dd; % 总间距为0.5*7 =3.5

data_length = 30000;
%traindata_theta = multiLabel(10000,2);
traindata_theta = unidrnd(90,[data_length,1]);  % 整数
%traindata_fe = multiLabel(10000,2);
traindata_fe = unidrnd(180,[data_length,1]);    % 整数
%%
Rxx_input = zeros(data_length,N*N,N*N,2);
Rxx_noise_k_means = zeros(data_length,N*N*N*N*2);
for i = 1:data_length
    theta = traindata_theta(i,:);       % 待估计角度 信号的入射角
    fe = traindata_fe(i,:);
    %A = exp(-1i*2*pi*d.'*sin(theta*derad));  %方向矢量 8*3
    A0 = exp(-1i*2*pi*d.'*(sin(theta*derad).*cos(fe*derad)))/sqrt(N);  %A0方向矩阵
    A1 = exp(-1i*2*pi*d.'*(sin(theta*derad).*sin(fe*derad)))/sqrt(N);  %A1方向矩阵
    S = randn(M,K);                          %信源信号（正态分布）N*100的矩阵
    data = sign(S);
    power = 0.1 + rand(1);                         % 信号的功率
    s = diag(power)*data;
    X = [];
    for im=1:N
        X=[X;A0*diag(A1(im,:))*s];           %接收信号
    end
    X1 = awgn(X,(14*rand)+1,'measured');             %添加高斯白噪声 
    Rxx_noise = X1*X1'/K; 
    Rxx_noise = Rxx_noise/max(max(Rxx_noise));
    Rxx_input(i,:,:,1) = real(Rxx_noise);
    Rxx_input(i,:,:,2) = imag(Rxx_noise);
    Rxx_noise_vec = Rxx_noise(:);
    Rxx_noise_k_means(i,:) = [real(Rxx_noise_vec); imag(Rxx_noise_vec)].';
    %Rxx_output(i,:,:,1) = real(Rxx_no_noise);
    %Rxx_output(i,:,:,2) = imag(Rxx_no_noise);
    disp(i);
end
%% 使用k_means 进行标注
rng('default')  % For reproducibility
Rxx_noise_k_means_test = Rxx_noise_k_means;

% 进行并行计算
stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);

%result = zeros(6,2);
k = 50;
[idx,C,sumd,D] = kmeans(Rxx_noise_k_means_test,k,'Options',options,'MaxIter',10000,...
    'Display','final','Replicates',10); % C 是质心的坐标


%% 画图 只画cluster1 cluster2 Cluster 3

figure;
plot(traindata_fe,traindata_theta,'.');
title 'Randomly Generated angle';
xlabel('fe angle');
ylabel('theta angle');

figure;
plot(traindata_fe(idx==1),traindata_theta(idx==1),'r.');
hold on;
plot(traindata_fe(idx==2),traindata_theta(idx==2),'b.');
plot(traindata_fe(idx==3),traindata_theta(idx==3),'k.');
legend('Cluster 1','Cluster 2','Cluster 3');
xlabel('fe angle');
ylabel('theta angle');


%% 打标签
label = zeros(data_length,50);
for i = 1:data_length
    id_every_sample = idx(i);
    label(i,id_every_sample) = 1;
end



