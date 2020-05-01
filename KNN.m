clc
clear all
close all

batch=8000;
ncompca=250;
flanders=3;

load('Trainnumbers.mat')

% if exist('reducedData','var') == 0
%     load('reducedData.mat')
% end

[Trainnumbers.image, porcentaje]=function_pca(Trainnumbers,ncompca);

%% KNN
knn=fitcknn(Trainnumbers.image(:,1:batch)',Trainnumbers.label(:,1:batch),'NumNeighbors',flanders);
prediction=knn.predict(Trainnumbers.image(:,batch+1:10000)');

acierto=1-length(find(prediction'~=Trainnumbers.label(:,batch+1:10000)))/length(prediction);


