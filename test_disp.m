n = 5*5;
plote = sqrt(n);
d = 1000;

figure('units','normalized','outerposition',[0 0 1 1]);

for i=1:n
    
subplot(plote,plote,i), digitdisp(dataset.image(:,i+d));
str = sprintf('%i',output(i+d));
title(str)

end