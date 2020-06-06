%autoencoder
%Primero se reduce con un PCA
close all
clear all
clc
tic
%% Datos de trabajo.
load('Trainnumbers.mat')
load('autoenc.mat')
data_original = Trainnumbers.image;
data=data_original;
figure;
digitdisp(data(:,4));


%% Calcula PCA
matrizCov = cov(data');
[transMat,diagonal]=eig(matrizCov);

%% Calcula pesos de las componentes principales
diagonal = diagonal./trace(diagonal)*100;
porcentaje = round(cumsum(flip(diag(diagonal))));

%% Reducci�n y reconstrucci�n de los datos
ncompca=300;

for i=1:ncompca
    transMatRed(i,:)=transMat(:,784+1-i)';
end

reducedData=(data'*transMatRed')';

reconstructedData1=(reducedData'*transMatRed)';
figure;
digitdisp(reconstructedData1(:,4));
mseError = mse(data-reconstructedData1)
%autoenc = trainAutoencoder(reducedData);

reconstructed = predict(autoenc,reducedData);

reconstructedData=(reconstructed'*transMatRed)';
figure;
digitdisp(reconstructedData(:,4));
mseError = mse(data-reconstructedData)
