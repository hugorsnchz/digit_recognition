clear all
close all

ncompca=51;
batch=3000;

tipo='linear';
load('Trainnumbers.mat')
load('Test_numbers_HW1.mat');

[Trainnumbers.reduced, porcentaje1]=function_pca(Trainnumbers,ncompca);
[Test_numbers.reduced, porcentaje2]=function_pca(Test_numbers,ncompca);

output=classify(Test_numbers.reduced',Trainnumbers.reduced(:,1:batch)',Trainnumbers.label(:,1:batch),tipo);
dataset=Test_numbers;
