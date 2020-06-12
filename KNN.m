clc
clear all
close all

batch=8000;
ncompca=175;%250
flanders=3;

load('Trainnumbers.mat')

[Trainnumbers.image, porcentaje]=function_pca(Trainnumbers,ncompca);

%% KNN

knn=fitcknn(Trainnumbers.image(:,1:batch)',Trainnumbers.label(:,1:batch),'NumNeighbors',flanders);
prediction=knn.predict(Trainnumbers.image(:,batch+1:10000)');

acierto=1-length(find(prediction'~=Trainnumbers.label(:,batch+1:10000)))/length(prediction);

cm = confusionmat(Trainnumbers.label(:,batch+1:10000),prediction');
numeros=[0 1 2 3 4 5 6 7 8 9];

cc=confusionchart(cm,numeros);

cc.Title = 'Digit Classification Using KNN Classifier';
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'column-normalized';



