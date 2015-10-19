clear;
clc;
close all;
%%
Data=readtable('../Data/SampleData1.csv');

Y=Data.Y;
X=table2array(Data);

%%
[result, detail]=PLSR_ANFIS(X, ...
                            Y, ...
                            {'gbellmf', 'gaussmf'}, ...
                            [2 2]);
                          
%%
[modYP,YP,ErrP]=apply_PLSR_ANFIS(X,result);

%%
figure
plot(Y,YP,'.');
hold on
plot(Y,modYP,'r.');

minV=min([Y;YP;modYP]);
maxV=max([Y;YP;modYP]);
axis square;
xlim([minV maxV]);
xlim([minV maxV]);
line(xlim,ylim,'Color','k','LineWidth',2)
RMSE_YP=sqrt(mean((Y-YP).^2));
RMSE_modYP=sqrt(mean((Y-modYP).^2));
legend(sprintf('YP (%0.2f)',RMSE_YP), ...
       sprintf('2Phase YP (%0.2f)',RMSE_modYP), ...
       'Location','NorthWest')
xlabel('Measured');
ylabel('Predicted');

figure,
fHandle=bar(abs([Y-YP, Y-modYP]),'stacked','barWidth',1);
axis tight
xlabel('Sample');
ylabel('Absolute Error')
legend('YP','2Phase YP')
