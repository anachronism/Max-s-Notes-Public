clear all
close all
clc
COMPUTE_IDEAL_FITNESS = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Log Files
fid = fopen('logging.txt','r');
A=textscan(fid,'%s','Delimiter','\n');
fclose(fid)

IterationA = strfind(A{1,1}, '::Iteration');
Iteration = find(not(cellfun('isempty', IterationA)));
B=zeros(length(Iteration),1);
for i=1:length(Iteration)
    B(i) = str2double(A{1,1}{Iteration(i),1}(14:end));
end
Iteration=[];
Iteration=B;

StartTimeA = strfind(A{1,1}, '::Start Time');
StartTime = find(not(cellfun('isempty', StartTimeA)));
B=cell(length(StartTime),1);
for i=1:length(StartTime)
    B{i} = A{1,1}{StartTime(i),1}(15:end);
end
StartTime=[];
StartTime=B;

ActionTypeA = strfind(A{1,1}, '::Action Type');
ActionType = find(not(cellfun('isempty', ActionTypeA)));
B=cell(length(ActionType),1);
for i=1:length(ActionType)
    B{i} = A{1,1}{ActionType(i),1}(16:end);
    if(strcmp(strtrim(B{i}),'Exploiting'))
        C(i) = 1;
    else
        C(i) = 0;
    end
end
ActionType=[];
ActionType=C;

ActionChosenTimeA = strfind(A{1,1}, '::Action Chosen');
ActionChosenTime = find(not(cellfun('isempty', ActionChosenTimeA)));
B=cell(length(ActionChosenTime),1);
for i=1:length(ActionChosenTime)
    B{i} = A{1,1}{ActionChosenTime(i),1}(18:end);
end
ActionChosenTime=[];
ActionChosenTime=B;

ActionIDA = strfind(A{1,1}, '::Action ID');
ActionID = find(not(cellfun('isempty', ActionIDA)));
B=zeros(length(ActionID),1);
for i=1:length(ActionID)
    B(i) = str2double(A{1,1}{ActionID(i),1}(14:end));
end
ActionID=[];
ActionID=B;

ActionParamsA = strfind(A{1,1}, '::Action Params');
ActionParams = find(not(cellfun('isempty', ActionParamsA)));
B=zeros(length(ActionParams),6);
for i=1:length(ActionParams)
    B(i,:) = str2num(A{1,1}{ActionParams(i),1}(18:end));
end
ActionParams=[];
ActionParams=B;

ActionSentTimeA = strfind(A{1,1}, '::Action Sent');
ActionSentTime = find(not(cellfun('isempty', ActionSentTimeA)));
B=cell(length(ActionSentTime),1);
for i=1:length(ActionSentTime)
    B{i} = A{1,1}{ActionSentTime(i),1}(16:end);
end
ActionSentTime=[];
ActionSentTime=B;

ActionRecdTimeA = strfind(A{1,1}, '::Action Received');
ActionRecdTime = find(not(cellfun('isempty', ActionRecdTimeA)));
B=cell(length(ActionRecdTime),1);
for i=1:length(ActionRecdTime)
    B{i} = A{1,1}{ActionRecdTime(i),1}(20:end);
end
ActionRecdTime=[];
ActionRecdTime=B;

RecvLockA = strfind(A{1,1}, '::Receive Lock');
RecvLock = find(not(cellfun('isempty', RecvLockA)));
B=zeros(length(RecvLock),1);
for i=1:length(RecvLock)
    B(i) = str2double(A{1,1}{RecvLock(i),1}(17:end));
end
RecvLock=[];
RecvLock=B;

MeasRecvdA = strfind(A{1,1}, '::Measurement Received');
MeasRecvd = find(not(cellfun('isempty', MeasRecvdA)));
B=zeros(length(MeasRecvd),6);
for i=1:length(MeasRecvd)
    B(i,:) = str2num(A{1,1}{MeasRecvd(i),1}(25:end));
end
MeasRecvd=[];
MeasRecvd=B;

ObjecFitA = strfind(A{1,1}, '::Objective Fitnesses Observed');
ObjecFit = find(not(cellfun('isempty', ObjecFitA)));
B=zeros(length(ObjecFit),6);
for i=1:length(ObjecFit)
    B(i,:) = str2num(A{1,1}{ObjecFit(i),1}(33:end));
end
ObjecFit=[];
ObjecFit=B;

FitObsA = strfind(A{1,1}, '::Fitness Observed');
FitObs = find(not(cellfun('isempty', FitObsA)));
B=zeros(length(FitObs),1);
for i=1:length(FitObs)
    B(i) = str2double(A{1,1}{FitObs(i),1}(21:end));
end
FitObs=[];
FitObs=B;

TrainingA = strfind(A{1,1}, '::Training');
Training = find(not(cellfun('isempty', TrainingA)));
B=cell(length(Training),1);
C=zeros(length(Training),1);
for i=1:length(Training)
    B{i} = A{1,1}{Training(i),1}(13:end);
    if(strcmp(B{i},'Yes'))
        C(i) = 1;
    else
        C(i) = 0;
    end
end
Training=[];
Training=C;

EndTimeA = strfind(A{1,1}, '::End Time');
EndTime = find(not(cellfun('isempty', EndTimeA)));
B=cell(length(EndTime),1);
for i=1:length(EndTime)
    B{i} = A{1,1}{EndTime(i),1}(13:end);
end
EndTime=[];
EndTime=B;

ModListA = strfind(A{1,1}, '::cogEngParams.nnAppSpec_modList:');
ModList = find(not(cellfun('isempty', ModListA)));
ModList = str2num(A{1,1}{ModList,1}(34:end));

CodListA = strfind(A{1,1}, '::cogEngParams.nnAppSpec_codList:');
CodList = find(not(cellfun('isempty', CodListA)));
CodList = str2num(A{1,1}{CodList,1}(34:end));

RollOffListA = strfind(A{1,1}, '::cogEngParams.nnAppSpec_rollOffList:');
RollOffList = find(not(cellfun('isempty', RollOffListA)));
RollOffList = str2num(A{1,1}{RollOffList,1}(38:end));

SymbolRateListA = strfind(A{1,1}, '::cogEngParams.nnAppSpec_symbolRateList:');
SymbolRateList = find(not(cellfun('isempty', SymbolRateListA)));
SymbolRateList = str2num(A{1,1}{SymbolRateList,1}(41:end));

TransmitPowerListA = strfind(A{1,1}, '::cogEngParams.nnAppSpec_transmitPowerList:');
TransmitPowerList = find(not(cellfun('isempty', TransmitPowerListA)));
TransmitPowerList = str2num(A{1,1}{TransmitPowerList,1}(44:end));

ModCodListA = strfind(A{1,1}, '::cogEngParams.nnAppSpec_modCodList:');
ModCodList = find(not(cellfun('isempty', ModCodListA)));
ModCodList = str2num(A{1,1}{ModCodList,1}(37:end));

ActionTableStartIdx = strfind(A{1,1}, '::Action List:');
ActionTableStartIdx = find(not(cellfun('isempty', ActionTableStartIdx)))+1;
ActionTableEndIdx = strfind(A{1,1}, '::Action Idxs:');
ActionTableEndIdx = find(not(cellfun('isempty', ActionTableEndIdx)))-3;
ActionTable = zeros(ActionTableEndIdx-ActionTableStartIdx+1,length(str2num(A{1,1}{ActionTableStartIdx,1}(1:end))));
for i=1:size(ActionTable,1)
    ActionTable(i,:) = str2num(A{1,1}{ActionTableStartIdx+i-1,1});
end

WeightVectorA = strfind(A{1,1}, '::cogEngParams.cogeng_fitnessWeights:');
WeightVector = find(not(cellfun('isempty', WeightVectorA)));
WeightVector = str2num(A{1,1}{WeightVector,1}(38:end));

% Open NewTec Log
bypassLogName = dir('bypass_log*');
fid = fopen(bypassLogName.name,'r');
A=textscan(fid,'%s','Delimiter','\n');
fclose(fid)

FrameTimeA = strfind(A{1,1}, 'TIMESTAMP=');
FrameTime = find(not(cellfun('isempty', FrameTimeA)));
B=cell(length(FrameTime),1);
for i=1:length(FrameTime)
    B{i} = A{1,1}{FrameTime(i),1}((FrameTimeA{FrameTime(i),1}-1)+11:(FrameTimeA{FrameTime(i),1}-1)+22);
end
FrameTime=[];
FrameTime=B;

% Open FER Curves
fid = fopen('ferCurves.txt','r');
A=textscan(fid,'%s','Delimiter','\n');
fclose(fid)

FERCurves = cell(size(A,1),2);
for i=1:size(A{1,1},1)
    t = str2num(A{1,1}{i,1});
    FERCurves(i,1) = {t(1:2:end)};
    FERCurves(i,2) = {t(2:2:end)};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Post Processing

% create MODCOD vs. Time
ModcodChosen = zeros(size(ActionParams,1),1);
for i=1:size(ActionParams,1)
    ModcodChosen(i) = ModCodList((abs(ModList-ActionParams(i,5))<0.0001) & ((abs(CodList-ActionParams(i,6))<0.0001)));
end

if(COMPUTE_IDEAL_FITNESS)
    % Generate Theoretical Best Action
    OptimalActionID = zeros(size(ActionParams,1),1);
    OptimalFitnessObserved = zeros(size(ActionParams,1),1);
    OptimalFitnessParams = zeros(size(ActionParams,1),size(ObjecFit,2));
    for i=1:size(ActionParams,1)
        [aID, bfObs, cfitP] = findTheoreticalBestAction(ActionTable, MeasRecvd(i,1), FERCurves, WeightVector, ModList, CodList, ModCodList);
        OptimalActionID(i) = aID;
        OptimalFitnessObserved(i) = bfObs;
        OptimalFitnessParams(i) = cfitP;
        if(mod(i,1000)==0)
            disp(i)
        end
    end
end

% Convert date vectors into seconds. (doesn't work if spans multiple days)
StartTimeSec = zeros(length(StartTime),1);
ActionChosenTimeSec = zeros(length(StartTime),1);
ActionSentTimeSec = zeros(length(StartTime),1);
ActionRecdTimeSec = zeros(length(StartTime),1);
EndTimeSec = zeros(length(StartTime),1);
for i=1:length(StartTime)
    StartTimeSec(i) = (datenum(StartTime{i,1}(13:end)) - floor(datenum(StartTime{i,1}(13:end))))*24*3600;
    ActionChosenTimeSec(i) = (datenum(ActionChosenTime{i,1}(13:end)) - floor(datenum(ActionChosenTime{i,1}(13:end))))*24*3600;
    ActionSentTimeSec(i) = (datenum(ActionSentTime{i,1}(13:end)) - floor(datenum(ActionSentTime{i,1}(13:end))))*24*3600;
    ActionRecdTimeSec(i) = (datenum(ActionRecdTime{i,1}(13:end)) - floor(datenum(ActionRecdTime{i,1}(13:end))))*24*3600;
    EndTimeSec(i) = (datenum(EndTime{i,1}(13:end)) - floor(datenum(EndTime{i,1}(13:end))))*24*3600;
end
FrameTimeSec = (datenum(FrameTime) - floor(datenum(FrameTime)))*24*3600;
INIT_TIME_SEC=StartTimeSec(1);

% use a sliding window of 40 ms and find when we received packets and when
% we didn't.
NewtecLck = zeros(length(ActionSentTimeSec),1);
for i=1:length(ActionSentTimeSec)
    A = find((FrameTimeSec < ActionSentTimeSec(i)) & (FrameTimeSec > ActionSentTimeSec(i)-0.040));
    if(~isempty(A))
        NewtecLck(i) = 1;
    end
end

save('parsedLogsWorkspace.mat')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
% Plot SNR Profile, ViaSat Lock
figure
plot(ActionRecdTimeSec-INIT_TIME_SEC,MeasRecvd(:,1),'b',ActionRecdTimeSec(RecvLock==0)-INIT_TIME_SEC,MeasRecvd(RecvLock==0,1),'*r',...
    'MarkerSize',1)
ylabel('EsN0 Profile (dB)')
xlabel('Time (sec)')
legend('EsN0 Profile','Unreliable Measurement')
% Plot FitObserved
figure
plot(ActionRecdTimeSec-INIT_TIME_SEC,FitObs.*NewtecLck,'b*','MarkerSize',3)
ylabel('Fit Observed')
xlabel('Time (sec)')
% Plot Histogram of Fit Observed (Log)
figure
[cnts,cntrs] = hist(FitObs.*NewtecLck,50);
bar(cntrs,cnts)
set(gca,'YScale','log')
xlabel('Fit Observed')
ylabel('Counts (Log Scale)')
% Plot Histogram of Fit Observed (Linear)
figure
[cnts,cntrs] = hist(FitObs.*NewtecLck,50);
bar(cntrs,cnts)
xlabel('Fit Observed')
ylabel('Counts')
% Plot Receiver Lock Analysis
figure
% 1: Newtec locked.
% -1: ViaSat locked
% -0.5: ViaSat locked, Newtec not locked.
% 0.5: Newtec locked, ViaSat not locked.
X = FitObs.*NewtecLck - FitObs.*RecvLock;
plot(ActionRecdTimeSec(NewtecLck==1)-INIT_TIME_SEC, 1*NewtecLck(NewtecLck==1),'g*', ...
     ActionRecdTimeSec(X>0) -INIT_TIME_SEC, 0.5*NewtecLck(X>0),  'b*', ...
     ActionRecdTimeSec(X<0) -INIT_TIME_SEC, -0.5*RecvLock(X<0),  'r*', ...
     ActionRecdTimeSec(RecvLock==1) -INIT_TIME_SEC, -1*RecvLock(RecvLock==1),'k*')
set(gca,'YTickLabel',[])
xlabel('Time (s)')
legend('Newtec Locked','Newtec Locked AND ViaSat NOT Locked', ...
       'ViaSat Locked AND Newtec NOT Locked', 'ViaSat Locked','Location','east')
title('Analysis of Receiver Lock')
% Plot MODCOD, TX Power, Rolloff vs. Time
figure
subplot(3,1,1)
plot(ActionSentTimeSec-INIT_TIME_SEC,ModcodChosen, 'b')
ylabel('MODCOD')
xlabel('Time (sec)')
subplot(3,1,2)
plot(ActionSentTimeSec-INIT_TIME_SEC,ActionParams(:,2), 'g')
ylabel('TX Power (dB)')
xlabel('Time (sec)')
subplot(3,1,3)
plot(ActionSentTimeSec-INIT_TIME_SEC,ActionParams(:,4), 'r')
ylabel('Roll Off')
xlabel('Time (sec)')
% Plot Action Update Rate Histo
figure
Y=(1./(ActionSentTimeSec(2:end)-ActionSentTimeSec(1:end-1)));
[cnts1,cntrs1] = hist(Y(ActionType(2:end)==1),linspace(min(Y),max(Y),100));
[cnts0,cntrs0] = hist(Y(ActionType(2:end)==0),linspace(min(Y),max(Y),100));
bar(cntrs1,cnts1,1,'FaceColor','g')
hold on;
bar(cntrs0,cnts0,1,'FaceColor','r')
hold off;
xlabel('Update Rate Jitter (Hz)')
ylabel('Counts')
legend('Exploit','Explore','Location','northwest')
% Plot Action Update Rate Histo (log scale)
figure
Y=(1./(ActionSentTimeSec(2:end)-ActionSentTimeSec(1:end-1)));
[cnts1,cntrs1] = hist(Y(ActionType(2:end)==1),linspace(min(Y),max(Y),100));
[cnts0,cntrs0] = hist(Y(ActionType(2:end)==0),linspace(min(Y),max(Y),100));
bar(cntrs1,cnts1,1,'FaceColor','g')
hold on;
bar(cntrs0,cnts0,1,'FaceColor','r')
hold off;
set(gca,'YScale','log')
xlabel('Update Rate Jitter (Hz)')
ylabel('Counts (Log Scale)')
legend('Exploit','Explore','Location','northwest')
% Plot Action Update Time Series
figure
X=ActionSentTimeSec(2:end);
Y=(1./(ActionSentTimeSec(2:end)-ActionSentTimeSec(1:end-1)));
plot(X(ActionType(2:end)==1)-INIT_TIME_SEC,Y(ActionType(2:end)==1),'b*',...
     X(ActionType(2:end)==0)-INIT_TIME_SEC,Y(ActionType(2:end)==0),'r*',...
     'MarkerSize',3)
xlabel('Time (sec)')
ylabel('Update Rate (Hz)')
legend('Exploit','Explore','Location','east')

% MASTER PLOT (Time Series)
figure
subplot(6,1,1)
plot(ActionRecdTimeSec-INIT_TIME_SEC,MeasRecvd(:,1),'b',ActionRecdTimeSec(RecvLock==0)-INIT_TIME_SEC,MeasRecvd(RecvLock==0,1),'*r',...
    'MarkerSize',1)
ylabel('EsN0 Profile (dB)')
xlabel('Time (sec)')
legend('EsN0 Profile','Unreliable Measurement','Location','northeast')
subplot(6,1,2)
plot(ActionRecdTimeSec-INIT_TIME_SEC,FitObs.*NewtecLck,'k*','MarkerSize',3)
ylabel('Fit Observed')
xlabel('Time (sec)')
subplot(6,1,3)
plot(ActionSentTimeSec-INIT_TIME_SEC,ModcodChosen, 'b')
ylabel('MODCOD')
xlabel('Time (sec)')
subplot(6,1,4)
plot(ActionSentTimeSec-INIT_TIME_SEC,ActionParams(:,2), 'g')
ylabel('TX Power (dB)')
xlabel('Time (sec)')
subplot(6,1,5)
plot(ActionSentTimeSec-INIT_TIME_SEC,ActionParams(:,4), 'r')
ylabel('Roll Off')
xlabel('Time (sec)')
subplot(6,1,6)
X=ActionSentTimeSec(2:end);
Y=(1./(ActionSentTimeSec(2:end)-ActionSentTimeSec(1:end-1)));
plot(X(ActionType(2:end)==1)-INIT_TIME_SEC,Y(ActionType(2:end)==1),'b*',...
     X(ActionType(2:end)==0)-INIT_TIME_SEC,Y(ActionType(2:end)==0),'r*',...
     'MarkerSize',3)
xlabel('Time (sec)')
ylabel('Update Rate (Hz)')
legend('Exploit','Explore','Location','southeast')
saveas(gcf,'MasterTimeSeriesPlots.fig')

% Master Plot (Histograms and Lock Analysis)
figure
subplot(2,3,4)
[cnts,cntrs] = hist(FitObs.*NewtecLck,50);
bar(cntrs,cnts)
set(gca,'YScale','log')
xlabel('Fit Observed')
ylabel('Counts (Log Scale)')
subplot(2,3,1)
[cnts,cntrs] = hist(FitObs.*NewtecLck,50);
bar(cntrs,cnts)
xlabel('Fit Observed')
ylabel('Counts')
subplot(2,3,2)
Y=(1./(ActionSentTimeSec(2:end)-ActionSentTimeSec(1:end-1)));
[cnts1,cntrs1] = hist(Y(ActionType(2:end)==1),linspace(min(Y),max(Y),100));
[cnts0,cntrs0] = hist(Y(ActionType(2:end)==0),linspace(min(Y),max(Y),100));
bar(cntrs1,cnts1,1,'FaceColor','g')
hold on;
bar(cntrs0,cnts0,1,'FaceColor','r')
hold off;
xlabel('Update Rate Jitter (Hz)')
ylabel('Counts')
legend('Exploit','Explore','Location','northwest')
subplot(2,3,5)
Y=(1./(ActionSentTimeSec(2:end)-ActionSentTimeSec(1:end-1)));
[cnts1,cntrs1] = hist(Y(ActionType(2:end)==1),linspace(min(Y),max(Y),100));
[cnts0,cntrs0] = hist(Y(ActionType(2:end)==0),linspace(min(Y),max(Y),100));
bar(cntrs1,cnts1,1,'FaceColor','g')
hold on;
bar(cntrs0,cnts0,1,'FaceColor','r')
hold off;
set(gca,'YScale','log')
xlabel('Update Rate Jitter (Hz)')
ylabel('Counts (Log Scale)')
legend('Exploit','Explore','Location','northwest')
subplot(2,3,3)
% 1: Newtec locked.
% -1: ViaSat locked
% -0.5: ViaSat locked, Newtec not locked.
% 0.5: Newtec locked, ViaSat not locked.
X = FitObs.*NewtecLck - FitObs.*RecvLock;
plot(ActionRecdTimeSec(NewtecLck==1)-INIT_TIME_SEC, 1*NewtecLck(NewtecLck==1),'g*', ...
     ActionRecdTimeSec(X>0) -INIT_TIME_SEC, 0.5*NewtecLck(X>0),  'b*', ...
     ActionRecdTimeSec(X<0) -INIT_TIME_SEC, -0.5*RecvLock(X<0),  'r*', ...
     ActionRecdTimeSec(RecvLock==1) -INIT_TIME_SEC, -1*RecvLock(RecvLock==1),'k*')
set(gca,'YTickLabel',[])
xlabel('Time (s)')
legend('Newtec Locked','Newtec Locked AND ViaSat NOT Locked', ...
       'ViaSat Locked AND Newtec NOT Locked', 'ViaSat Locked','Location','east')
title('Analysis of Receiver Lock')
subplot(2,3,6)
plot(ActionRecdTimeSec(Training==1)-INIT_TIME_SEC,Training(Training==1),'*r', ...
     ActionRecdTimeSec(Training==0)-INIT_TIME_SEC,Training(Training==0),'*g')
xlabel('Time (sec)')
legend('Training','Not Training','Location','east')
set(gca,'YTickLabel',[])
saveas(gcf,'MasterHistogramPlots.fig')


% Plot FitObserved vs Ideal
if(COMPUTE_IDEAL_FITNESS)
    figure
    subplot(2,1,1)
    plot(ActionRecdTimeSec-INIT_TIME_SEC,MeasRecvd(:,1),'b',ActionRecdTimeSec(RecvLock==0)-INIT_TIME_SEC,MeasRecvd(RecvLock==0,1),'*r',...
        'MarkerSize',1)
    ylabel('EsN0 Profile (dB)')
    xlabel('Time (sec)')
    legend('EsN0 Profile','Unreliable Measurement','Location','northeast')
    subplot(2,1,2)
    plot(ActionRecdTimeSec-INIT_TIME_SEC,FitObs,'ro', ...
         ActionRecdTimeSec-INIT_TIME_SEC,FitObs.*NewtecLck,'k*', ...
         ActionRecdTimeSec-INIT_TIME_SEC,OptimalFitnessObserved,'g', ...
         'MarkerSize',3)
    ylabel('Fit Observed')
    xlabel('Time (sec)')
    legend('Fitness Observed','Fitness After Post Processing','Ideal Fitness','Location','east')
end

% MASTER PLOT (Time Series)
figure
subplot(4,1,1)
plot(ActionRecdTimeSec-INIT_TIME_SEC,FitObs.*NewtecLck)
ylabel('Fitness Observed')
xlabel('Time (sec)')
subplot(4,1,2)
plot(ActionRecdTimeSec-INIT_TIME_SEC,ObjecFit(:,1).*NewtecLck,'b',ActionRecdTimeSec-INIT_TIME_SEC,ObjecFit(:,2).*NewtecLck,'r')
ylabel('SubFit Observed')
xlabel('Time (sec)')
legend('Throughput','FER','Location','northeast')
subplot(4,1,3)
plot(ActionRecdTimeSec-INIT_TIME_SEC,ObjecFit(:,3).*NewtecLck,'b',ActionRecdTimeSec-INIT_TIME_SEC,ObjecFit(:,4).*NewtecLck,'r')
ylabel('SubFit Observed')
xlabel('Time (sec)')
legend('Bandwidth','Spectral Efficiency','Location','northeast')
subplot(4,1,4)
plot(ActionRecdTimeSec-INIT_TIME_SEC,ObjecFit(:,5).*NewtecLck,'b',ActionRecdTimeSec-INIT_TIME_SEC,ObjecFit(:,6).*NewtecLck,'r')
ylabel('SubFit Observed')
xlabel('Time (sec)')
legend('TX Power Efficiency','Power Consumption','Location','northeast')
saveas(gcf,'FitObservedTimeSeriesPlots.fig')

% VIRTUAL EXPLORATION ANALYSIS
figure
Y=FitObs.*NewtecLck;
subplot(2,1,1)
plot(ActionRecdTimeSec(ActionType(1:end)==0)-INIT_TIME_SEC,Y(ActionType(1:end)==0),'r*',ActionRecdTimeSec-INIT_TIME_SEC,Y,'b')
xlabel('Time (s)')
ylabel('Fitness Observed')
legend('Exploration Fitness','Fitness Observed','Location','northeast')
subplot(2,1,2)
%[cnts1,cntrs1] = hist(Y(ActionType(1:end)==1),linspace(min(Y),max(Y),100));
[cnts0,cntrs0] = hist(Y(ActionType(1:end)==0),linspace(min(Y),max(Y),100));
%bar(cntrs1,cnts1,1,'FaceColor','g')
%hold on;
bar(cntrs0,cnts0,1,'FaceColor','r')
hold off;
set(gca,'YScale','log')
set(gca,'Xlim',[0 1])
xlabel('Fit Observed')
ylabel('Counts (Log Scale)')
%legend('Exploit','Explore','Location','northwest')
saveas(gcf,'VirtualExplorationAnalysis.fig')

