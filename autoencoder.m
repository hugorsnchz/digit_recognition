%autoencoder
%Primero se reduce con un PCA a dimension 300
close all
clear all
clc

% Datos de trabajo.
load('Trainnumbers.mat')
load('autoenc.mat')
data_original = Trainnumbers.image;
data=data_original;
figure;
digitdisp(data(:,4));
title('Original');


% Calcula PCA
matrizCov = cov(data');
[transMat,diagonal]=eig(matrizCov);

% Calcula pesos de las componentes principales
diagonal = diagonal./trace(diagonal)*100;
porcentaje = round(cumsum(flip(diag(diagonal))));

% Reduccion de los datos
ncompca=300;

for i=1:ncompca
    transMatRed(i,:)=transMat(:,784+1-i)';
end

reducedData=(data'*transMatRed')';

%Entrenamiento de la red
% hiddenSize = 55;
% autoenc = trainAutoencoder(reducedData,hiddenSize);

%aqui se codifican los datos, reduciendo la dimensión a 55
Datosreducidosencoder = encode(autoenc,reducedData);

R = decode(autoenc,Datosreducidosencoder);

%se reconstruye el pca
reconstructedData=(R'*transMatRed)';

figure;
digitdisp(reconstructedData(:,4));
title('PCA(300) + Autoenc (55)');
disp('MSE PCA(300) + Autoenc (55)');
mseError = mse(data-reconstructedData)
