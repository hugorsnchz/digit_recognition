clc
clear all
close all

%% Bayesiano

batch=8000;
ncompca=51;

load('Trainnumbers.mat')

[Trainnumbers.image, porcentaje]=function_pca(Trainnumbers,ncompca);

%tipo='linear';
tipo='quadratic';

output=classify(Trainnumbers.image(:,batch+1:10000)',Trainnumbers.image(:,1:batch)',Trainnumbers.label(:,1:batch),tipo);
acierto=1-length(find(output'~=Trainnumbers.label(:,batch+1:10000)))/length(Trainnumbers.label(:,batch+1:10000));

cm = confusionmat(Trainnumbers.label(:,batch+1:10000),output');
numeros=[0 1 2 3 4 5 6 7 8 9];

cc=confusionchart(cm,numeros);

cc.Title = 'Digit Classification Using Bayessian Classifier';
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'column-normalized';

