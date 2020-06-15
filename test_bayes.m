clear all
close all

ncompca=51;

tipo='quadratic';
load('Trainnumbers.mat')
load('Test_numbers_HW1.mat');

[Trainnumbers.reduced, transMatRed]=function_pca2(Trainnumbers,ncompca);
Test_numbers.reduced = (Test_numbers.image'*transMatRed')';

output=classify(Test_numbers.reduced',Trainnumbers.reduced',Trainnumbers.label,tipo);
dataset=Test_numbers;
