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
    'Display','final','Replicates',2); % C 是质心的坐标


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
%%
save Rxx_input Rxx_input
save label label
save idx idx
save traindata_fe traindata_fe
save traindata_theta traindata_theta




%%
traindata_theta = round(traindata_theta);
traindata_fe = round(traindata_fe);
theta_label = zeros(data_length,90);
fe_label = zeros(data_length,90);
for i = 1:data_length
    theta_all = traindata_theta(i,:);
    fe_all = traindata_fe(i,:);
    theta_label(i,theta_all(1)) = 1;
    theta_label(i,theta_all(2)) = 1;
    fe_label(i,fe_all(1)) = 1;
    fe_label(i,fe_all(2)) = 1;
    disp(i);
end


%%
save theta_label theta_label;
save fe_label fe_label;
save Rxx_input Rxx_input

%% test
theta = [20 50];
fe = [20 50];
A0 = exp(-1i*2*pi*d.'*(sin(theta*derad).*cos(fe*derad)))/sqrt(N);  %A0方向矩阵
A1 = exp(-1i*2*pi*d.'*(sin(theta*derad).*sin(fe*derad)))/sqrt(N);  %A1方向矩阵
M = 2;
S = randn(M,K);                          %信源信号（正态分布）N*100的矩阵
X=[];
for im=1:N
    X=[X;A0*diag(A1(im,:))*S];           %接收信号
end
X1 = awgn(X,5,'measured');             %添加高斯白噪声 添加之前计算功率-10-10
Rxx_noise = X1*X1'/K; 
Rxx_noise = Rxx_noise/max(max(Rxx_noise));
Rxx_test(1,:,:,1) = real(Rxx_noise);
Rxx_test(1,:,:,2) = imag(Rxx_noise);
save Rxx_test Rxx_test;
%%
load('theta_est.mat');
load('fe_est.mat');
%figure(1);
subplot(2,1,1)
plot(theta_est);
xlabel('elevation(degree)')
ylabel('magnitude(dB)')
%figure(2);
subplot(2,1,2)
plot(fe_est);
xlabel('azimuth(degree)')
ylabel('magnitude(dB)')

%%
Rxx_noise = X1*X1'/K; 
[EV,D]= eig(Rxx_noise);                   %特征值分解后得到特征值组成的向量形式D和特征值对应特征向量组成的矩阵EV
EVA = diag(D)';                     %构造对角阵EVA 特征值向量D的元素是对角线上的数值，并将矩阵转置
[EVA, I] = sort(EVA);               %按照特征值大小对对角阵元素升序排列，原先的顺序为I
EVA = fliplr(EVA);                  %特征值从大到小
EV = fliplr(EV(:,I));               %先使EV中列向量按照I的顺序排列，再进行翻转
Un = EV(:,M+1:8);                     %噪声子空间
%% 构造MUSIC函数
rad = pi/180;
for ang1=1: 900
    for ang2= 1:900
        thet(ang1) =(ang1-1)*derad/10;
        f(ang2)  = (ang2-1)*derad/10;
        ax = exp(-1i*2*pi*d.'*(sin(thet(ang1)).*cos(f(ang2))))/sqrt(N);
        ay = exp(-1i*2*pi*d.'*(sin(thet(ang1)).*sin(f(ang2))))/sqrt(N);
        a = kron(ay,ax);
        SP(ang1,ang2) = 1/(a'*Un*Un'*a);
    end
end
%%
SP = abs(SP);
SPmax = max(max(SP));
SP = SP/SPmax; %归一化
h = mesh(thet/rad,f/rad,SP);      %绘制空间谱函数图
set(h,'Linewidth',2);
xlabel('elevation(degree)');      %仰角
ylabel('azimuth(degree)');        %方位角
zlabel('magnitude(dB)');
%% 使用手肘法确定k
rng('default')  % For reproducibility
Rxx_noise_k_means_test = Rxx_noise_k_means;

% 进行并行计算
stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);
result = zeros(j,2);
for j=1:7

k = (j+6)*10;
% [lable,c,sumd,d]=kmeans(X,k,'dist','sqeuclidean');
[idx,C,sumd,D] = kmeans(Rxx_noise_k_means_test,k,'Options',options,'MaxIter',10000,...
    'Display','final','Replicates',5); % C 是质心的坐标
% data，n×p原始数据向量
% lable，n×1向量，聚类结果标签；
% c，k×p向量，k个聚类质心的位置
% sumd，k×1向量，类间所有点与该类质心点距离之和
% d，n×k向量，每个点与聚类质心的距离
sse1 = sum(sumd.^2);
result(j,1) = k;
result(j,2) = sse1;
end

plot(result(2:end,1),result1(2:end))

grid on;
title('不同K值聚类偏差图') 
xlabel('分类数(K值)') 
ylabel('归一化簇内误差平方和')


result1 = result(1:end,2)/max(result(1:end,2));

result1(4) = 0.382;
save result result;


