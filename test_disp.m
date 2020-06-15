plote = 5;
d = 4000;

figure('units','normalized','outerposition',[0 0 1 1]);

for i=1:plote*plote
    
    subplot(plote,plote,i), digitdisp(dataset.image(:,i+d));
    str = sprintf('%i',output(i+d));
    title(str)
end