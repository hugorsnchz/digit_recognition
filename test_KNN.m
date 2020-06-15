clear all
close all

ncompca=175;
load('Trainnumbers.mat')
load('Test_numbers_HW1.mat');
flanders=3;

[Trainnumbers.reduced, transMatRed]=function_pca2(Trainnumbers,ncompca);
Test_numbers.reduced = (Test_numbers.image'*transMatRed')';

knn=fitcknn(Trainnumbers.reduced',Trainnumbers.label,'NumNeighbors',flanders);
output=(knn.predict(Test_numbers.reduced'))';
dataset=Test_numbers;
