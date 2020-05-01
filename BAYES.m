clc
clear all
close all

%% Bayesiano

batch=9000;
ncompca=200;

load('Trainnumbers.mat')

% if exist('reducedData','var') == 0
%     load('reducedData.mat')
% end

[Trainnumbers.image, porcentaje]=function_pca(Trainnumbers,ncompca);

tipo='linear';
%tipo='quadratic';

output=classify(Trainnumbers.image(:,batch+1:10000)',Trainnumbers.image(:,1:batch)',Trainnumbers.label(:,1:batch),tipo);
acierto=1-length(find(output'~=Trainnumbers.label(:,batch+1:10000)))/length(Trainnumbers.label(:,batch+1:10000));
