clear all
close all

ncompca=175;
load('Trainnumbers.mat')
load('Test_numbers_HW1.mat');
flanders=3;

[Trainnumbers.reduced, porcentaje1]=function_pca(Trainnumbers,ncompca);
[Test_numbers.reduced, porcentaje2]=function_pca(Test_numbers,ncompca);

knn=fitcknn(Trainnumbers.reduced',Trainnumbers.label,'NumNeighbors',flanders);
output=knn.predict(Test_numbers.reduced');
dataset=Test_numbers;
