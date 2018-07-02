% emergency
figure
% no pretraining
plots = [2,3,6];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'r')
hold on;
% pretraining
plots = [8,11,12];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'b')
hold on;
xlim([0.01 1])
ylim([0 0.15])
legend('No Pre-Training','Pre-Training','Location','northwest')
xlabel('Fitness Observed')
ylabel('Normalized Counts')
grid on
cropplot(gcf,gca)


% power saving (note that two of them didn't have newtec lock for half the
% pass...do we want to throw out that data?)
figure
%no pretraining
plots = [4,5,7];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'r')
hold on;
% pretraining
plots = [9,10,13,16];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'b')
hold on;
xlim([0.01 1])
ylim([0 0.15])
%legend('Excellent','Great','Good','Okay/Poor','Location','northwest')
legend('No Pre-Training','Pre-Training','Location','northwest')
xlabel('Fitness Observed')
ylabel('Normalized Counts')
grid on
cropplot(gcf,gca)

% spectral efficiency
figure
% no pretraining
plots = [14,15,17];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'r')
hold on;
% pretraining
plots = [18,19,20];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'b')
hold on;
xlim([0.01 1])
ylim([0 0.15])
legend('No Pre-Training','Pre-Training','Location','northwest')
xlabel('Fitness Observed')
ylabel('Normalized Counts')
grid on
cropplot(gcf,gca)