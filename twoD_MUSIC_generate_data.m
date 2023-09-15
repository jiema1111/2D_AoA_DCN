%%
clc;
clear;
close all;
%% 估计仰角和方位角

derad = pi/180;      %角度->弧度
radeg = 180/pi;
N = 8;                   
M = 1;               % 信源数目
K = 200;             % 快拍数

dd = 0.5;            
d = 0: dd : (N-1)*dd; 

data_length = 30000;
%traindata_theta = multiLabel(10000,2);
traindata_theta = unidrnd(90,[data_length,1]);  
%traindata_fe = multiLabel(10000,2);
traindata_fe = unidrnd(180,[data_length,1]);    
%%
Rxx_input = zeros(data_length,N*N,N*N,2);
Rxx_noise_k_means = zeros(data_length,N*N*N*N*2);
for i = 1:data_length
    theta = traindata_theta(i,:);       % 待估计角度 信号的入射角
    fe = traindata_fe(i,:);
    A0 = exp(-1i*2*pi*d.'*(sin(theta*derad).*cos(fe*derad)))/sqrt(N);  %A0方向矩阵
    A1 = exp(-1i*2*pi*d.'*(sin(theta*derad).*sin(fe*derad)))/sqrt(N);  %A1方向矩阵
    S = randn(M,K);                          %信源信号
    data = sign(S);
    power = 0.1 + rand(1);                % 信号的功率
    s = diag(power)*data;
    X = [];
    for im=1:N
        X=[X;A0*diag(A1(im,:))*s];          
    end
    X1 = awgn(X,(14*rand)+1,'measured');             %添加高斯白噪声 
    Rxx_noise = X1*X1'/K; 
    Rxx_noise = Rxx_noise/max(max(Rxx_noise));
    Rxx_input(i,:,:,1) = real(Rxx_noise);
    Rxx_input(i,:,:,2) = imag(Rxx_noise);
    Rxx_noise_vec = Rxx_noise(:);
    Rxx_noise_k_means(i,:) = [real(Rxx_noise_vec); imag(Rxx_noise_vec)].';
    disp(i);
end
%% label by k-means
rng('default')  % For reproducibility
Rxx_noise_k_means_test = Rxx_noise_k_means;

stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);

%result = zeros(6,2);
k = 50;
[idx,C,sumd,D] = kmeans(Rxx_noise_k_means_test,k,'Options',options,'MaxIter',10000,...
    'Display','final','Replicates',10);

%% label data
label = zeros(data_length,50);
for i = 1:data_length
    id_every_sample = idx(i);
    label(i,id_every_sample) = 1;
end



