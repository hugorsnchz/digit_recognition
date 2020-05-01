clc
clear all
close all
load Trainnumbers.mat
%% KNN
knn=fitcknn(Trainnumbers.image',Trainnumbers.label,'NumNeighbors',3);
prediction=knn.predict(Trainnumbers.image');

error_knn=length(find(prediction'~=Trainnumbers.label));