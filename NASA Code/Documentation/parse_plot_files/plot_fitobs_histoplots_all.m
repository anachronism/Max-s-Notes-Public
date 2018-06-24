% emergency
figure
% excellent
plots = 2;
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'r')
hold on;
% great
plots = [8,12];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'b')
hold on;
% good
plots = [6];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'g')
hold on;
% okay/poor
plots = [3,11];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'k')
hold on;
xlim([0.01 1])
ylim([0 0.3])
legend('Excellent','Great','Good','Okay/Poor','Location','northwest')
xlabel('Fitness Observed')
ylabel('Normalized Counts')
grid on
cropplot(gcf,gca)


% power saving (note that two of them didn't have newtec lock for half the
% pass...do we want to throw out that data?)
figure
% excellent
% plots = 10;
% bars = linspace(0,1,100);
% allcnts=zeros(1,length(bars));
% for i=plots
%     [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
%     allcnts = allcnts+cnts;
% end
% plot(cntrs,allcnts/sum(allcnts),'r')
% hold on;
% great
plots = [4];%,13];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'b')
hold on;
% good
plots = [5,7,16];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'g')
hold on;
% okay/poor
plots = [9];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'k')
hold on;
xlim([0.01 1])
ylim([0 0.3])
%legend('Excellent','Great','Good','Okay/Poor','Location','northwest')
legend('Great','Good','Okay/Poor','Location','northeast')
xlabel('Fitness Observed')
ylabel('Normalized Counts')
grid on
cropplot(gcf,gca)

% spectral efficiency
figure
plots = 19;
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'r')
hold on;
% great
plots = [15,18];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'b')
hold on;
% good
plots = [14,17];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'g')
hold on;
% okay/poor
plots = [20];
bars = linspace(0,1,100);
allcnts=zeros(1,length(bars));
for i=plots
    [cnts,cntrs] = hist(all_workspaces(i).FitObs.*all_workspaces(i).NewtecLck,bars);
    allcnts = allcnts+cnts;
end
plot(cntrs,allcnts/sum(allcnts),'k')
hold on;
xlim([0.01 1])
ylim([0 0.3])
legend('Excellent','Great','Good','Okay/Poor','Location','northwest')
xlabel('Fitness Observed')
ylabel('Normalized Counts')
grid on
cropplot(gcf,gca)