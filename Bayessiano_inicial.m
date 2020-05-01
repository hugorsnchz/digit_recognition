clc
clear all
close all
load Trainnumbers.mat

%% Bayesiano

tipo='linear';
%tipo='quadratic'
bay=classify(Trainnumbers.image',Trainnumbers.image',Trainnumbers.label,tipo);

error_bay=length(find(bay'~=Trainnumbers.label));
