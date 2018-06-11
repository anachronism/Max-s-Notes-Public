tic
%Approach 2 Jornal paper

clearvars
close all; clc

% INPUT TIME SERIES
load('recv_amp_rain.mat') %Fixed ground receiver + Rain
TOTAL=recv_amp_rain; %To be corrupted by measurement noise are input to the IMM filters. 

% load('recv_amp_LMS.mat')% LMS (Mobile ground receiver) + clear sky
% TOTAL=recv_amp_LMS;

% load('recv_amp_LMS_rain.mat')% LMS (Mobile ground receiver) + rain
% TOTAL=recv_amp_LMS_rain;


exp_power=0;  %Clear sky power amplitude expected level (dB)


%Free space noise power is assumed to be constant, no need to decrease 'DC power level' from received power level.

end_T=50;%Training dataset size

d_t_k_ahead=5;%K-steps ahead

%Adds Measurement Noise Vk
rng default
rng('shuffle') 
R=0.1;%Measurement noise sdt. dev. [assumed zero mean]
TOTAL=TOTAL+(R.*randn(1,length(TOTAL)));

%For outage series [REMOVE]
outage_=zeros(1,length(TOTAL));
outage_flag=0;%flag to save values before outages

%Delays [FUTURE USE]
delay_Rx=0;% # of delayed samples at receiver
delay_Tx=0;% # of delayed samples at transceiver
pred_=1;%pred_=0 Prediction OFF; pr=1 Prediction ON;

%[REMOVE]
dim_F_1=4;
dim_F_2=4;

% 2-state semi-Markovian probabilities 
pi_=[0.9 0.1; 0.1 0.9];%never changes

%% Training Q value sets for individual inner filters; must be at least two orders of magnitude different
qf_1=[0 10.^[-10 -3]];
qf_2=[10^-10 10^-1 1 10];%no zeros here

% Matrix combinator; builds all possible matrices using combinations of q values
comb_Q1=combinator(size(qf_1,2),3,'p','r'); %opposite diagonal should be equal to assure symmetry
comb_Q2=combinator(size(qf_2,2),3,'p','r');
% comb_Q1Q2=combinator((size(comb_Q1,1)),2,'p');%qf_1 and qf_2 must be of the same size!!! [USED FOR APPROACH 1]

comb_Q1=[comb_Q1(:,1) comb_Q1(:,2) comb_Q1(:,2) comb_Q1(:,3)];%Building symmetrical matrices; duplicating columns 2 and 3 to assure symmetry
comb_Q2=[comb_Q2(:,1) comb_Q2(:,2) comb_Q2(:,2) comb_Q2(:,3)];

%% Filter_1 Individual  [TRAINING KF]
for ii=1:size(comb_Q1,1)%All combinations of Q1
    Q_1_F1=reshape(qf_1(comb_Q1(ii,:)),2,2)';
    %Check for positive definite
    if all(eig(Q_1_F1) > 0) 
    
    %KF parameters    
    %Prediction horizon k
    pred_hor=delay_Rx+1+delay_Tx;
    
    NN=10; %Variance R measurement window [REMOVE]
    N=3;%size of sliding window [REMOVE]
    
    P_1=1000*eye(2);%Cov matrix of state-estimation error   
    X_1=[exp_power 0]';   
    d_t=1;
    F_1=[1 d_t; 0 1];   
    H_1=[1 0];    
    I=eye(2); 
    
    F_1_k_ahead=[1 d_t_k_ahead; 0 1];    
    
    %KF iteration    
    for jji = 1:1:end_T %Iterates over the entire SNR profile 
        if jji>=delay_Rx+1
            %Parallel Filtering
            X_new=[TOTAL(jji-delay_Rx) 0]';%Gets new measurement
            
            %Filter 1
            X_1=F_1*X_1;
            P_1=F_1*P_1*F_1'+Q_1_F1;
            
            K_1=P_1*H_1'*((H_1*P_1*H_1'+R)\1);
            e_1=((H_1*F_1*X_new)-(H_1*X_1));
            X_1=X_1+(K_1*e_1);
            P_1=(I-K_1*H_1)*P_1;  
        end
        pv_F1(jji-delay_Rx)=X_1(1); %1-step ahead
        pred_k_ahead_F1=F_1_k_ahead*X_1; %Projection K steps ahead
        pv_k_ahead_F1(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead_F1(1); %predicted value k_steps ahead
    end
    error=pv_F1(1:length(pv_F1))-TOTAL(1:length(pv_F1));%Residuals 1-step ahead
    sum_abs_error_F1(ii)=sum((abs(error)).^2);% sum of squared errors 1-step ahead
    error=pv_k_ahead_F1(d_t_k_ahead+1:length(pv_F1))-TOTAL(d_t_k_ahead+1:length(pv_F1));% residual k-steps ahead
    sum_abs_error_ahead_F1(ii)=sum((abs(error)).^2);% sum of squared errors k-steps ahead
    
    end
end
%Learning Filter_1  [condition to find Q1 using min squared error k-steps ahead]
[C1,ind_min]=sort(sum_abs_error_ahead_F1,2);%Uses min error k-steps ahead
non_zero=find(C1);%Avoids using indexes from non-positive Q matrices
min_error_k_steps=C1(non_zero(1));
i=ind_min(non_zero(1));%1st non-zero element index
%Validation Filter_1 [SANITY CHECK using validation data; additional level of certainty; validate using small portion of validation data]
d_t=1;
F_1=[1 d_t; 0 1];
H_1=[1 0];
I=eye(2);
F_1_k_ahead=[1 d_t_k_ahead; 0 1];
Q_1_val_F1=reshape(qf_1(comb_Q1(i,:)),2,2)';
P_1=1000*eye(2);   
X_1=[exp_power 0]';
for jji = end_T+1:1:511
    if jji>=delay_Rx+1
        %Parallel Filtering
        X_new=[TOTAL(jji-delay_Rx) 0]';%Gets new measurement
        
        %Filter 1
        X_1=F_1*X_1;
        P_1=F_1*P_1*F_1'+Q_1_val_F1;
        
        K_1=P_1*H_1'*((H_1*P_1*H_1'+R)\1);
        e_1=((H_1*F_1*X_new)-(H_1*X_1));
        X_1=X_1+(K_1*e_1);
        P_1=(I-K_1*H_1)*P_1;
    end
    pv_F1_val(jji-delay_Rx)=X_1(1); %1-step ahead
    pred_k_ahead_F1_val=F_1_k_ahead*X_1;
    pv_k_ahead_F1_val(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead_F1_val(1);%predicted value k_steps ahead using best Q_1 and Q_2
end
clear error
error=pv_F1_val(end_T+1:length(pv_F1_val))-TOTAL(end_T+1:length(pv_F1_val));
sum_abs_error_F1_val=sum((abs(error)).^2);
clear error
error=pv_k_ahead_F1_val(end_T+1+d_t_k_ahead:length(pv_F1_val))-TOTAL(end_T+1+d_t_k_ahead:length(pv_F1_val));
sum_abs_error_ahead_F1_val=sum((abs(error)).^2)

%% Filter_2 Individual [TRAINING KF]
for ii=1:size(comb_Q2,1)%Individual Filter 1 (all combinations of Q1)
    Q_2_F2=reshape(qf_2(comb_Q2(ii,:)),2,2)';
    %Check for positive definite
    if all(eig(Q_2_F2) > 0) 
    
    %KF parameters    
    %Prediction horizon k
    pred_hor=delay_Rx+1+delay_Tx;
    
    NN=10; %Variance R measurement window [REMOVE]
    N=3;%size of sliding window    [REMOVE]
    
    P_2=1000*eye(2);%Cov matrix of state-estimation error   
    X_2=[exp_power 0]';   
    d_t=1;
    F_2=[1 d_t; 0 1];   
    H_2=[1 0];    
    I=eye(2); 
    F_1_k_ahead=[1 d_t_k_ahead; 0 1];    
    
    %KF iteration    
    for jji = 1:1:end_T %Iterates over the entire SNR profile      
        if jji>=delay_Rx+1
            %Parallel Filtering
            X_new=[TOTAL(jji-delay_Rx) 0]';%Gets new measurement
            
            %Filter 2
            X_2=F_2*X_2;
            P_2=F_2*P_2*F_2'+Q_2_F2;
            
            K_2=P_2*H_2'*((H_2*P_2*H_2'+R)\1);
            e_2=((H_2*F_2*X_new)-(H_2*X_2));
            X_2=X_2+(K_2*e_2);
            P_2=(I-K_2*H_2)*P_2; 
        end
        pv_F2(jji-delay_Rx)=X_2(1); %1-step ahead
        pred_k_ahead_F2=F_1_k_ahead*X_2;
        pv_k_ahead_F2(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead_F2(1);%predicted value k_steps ahead using best Q_1 and Q_2
    end
    error=pv_F2(1:length(pv_F2))-TOTAL(1:length(pv_F2));
    sum_abs_error_F2(ii)=sum((abs(error)).^2);
    error=pv_k_ahead_F2(d_t_k_ahead+1:length(pv_F2))-TOTAL(d_t_k_ahead+1:length(pv_F2));
    sum_abs_error_ahead_F2(ii)=sum((abs(error)).^2);
    end  
end
%Learning Filter_2 [condition to find Q2 using min squared error k-steps ahead]
[C1,ind_min]=sort(sum_abs_error_ahead_F2,2);%Uses min error K-steps ahead
non_zero=find(C1);%Avoids using indexes from non-positive Q matrices
min_error_k_steps=C1(non_zero(1));
i=ind_min(non_zero(1));%1st non-zero element index
%Validation Filter_2  [SANITY CHECK using validation data; additional level of certainty; validate using small portion of validation data]
d_t=1;
F_2=[1 d_t; 0 1];
H_2=[1 0];
I=eye(2);
F_1_k_ahead=[1 d_t_k_ahead; 0 1];
Q_2_val_F2=reshape(qf_2(comb_Q2(i,:)),2,2)';
P_2=1000*eye(2);   
X_2=[exp_power 0]';
for jji = end_T+1:1:511
    if jji>=delay_Rx+1
        %Parallel Filtering
        X_new=[TOTAL(jji-delay_Rx) 0]';%Gets new measurement
        
        %Filter 2
        X_2=F_2*X_2;
        P_2=F_2*P_2*F_2'+Q_2_val_F2;
        
        K_2=P_2*H_2'*((H_2*P_2*H_2'+R)\1);
        e_2=((H_2*F_2*X_new)-(H_2*X_2));
        X_2=X_2+(K_2*e_2);
        P_2=(I-K_2*H_2)*P_2;
    end
    pv_F2_val(jji-delay_Rx)=X_2(1); %1-step ahead
    pred_k_ahead_F2_val=F_1_k_ahead*X_2;
    pv_k_ahead_F2_val(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead_F2_val(1);%predicted value k_steps ahead using best Q_2
end
clear error
error=pv_F2_val(end_T+1:length(pv_F2_val))-TOTAL(end_T+1:length(pv_F2_val)); %residual 1-step ahead
sum_abs_error_F2_val=sum((abs(error)).^2);%sum of squared errors(residuals) 1-step ahead
clear error
error=pv_k_ahead_F2_val(end_T+1+d_t_k_ahead:length(pv_F2_val))-TOTAL(end_T+1+d_t_k_ahead:length(pv_F2_val));%residuals k-steps ahead
sum_abs_error_ahead_F2_val=sum((abs(error)).^2)%sum of squared errors(residuals) k-steps ahead
 
%% Validation IMM [MAIN]
d_t=1;
F_1=[1 d_t; 0 1];
F_2=F_1;
H_1=[1 0];
H_2=H_1;
I=eye(2);
F_1_k_ahead=[1 d_t_k_ahead; 0 1];

Q_1_val=Q_1_val_F1;
Q_2_val=Q_2_val_F2;

% Reset matrices
    P_1=1000*eye(2);%Cov matrix of state-estimation error
    P_2=1000*eye(2);
    
    X_1=[exp_power 0]';
    X_2=[exp_power 0]';    
    
    mu_=[0.5;0.5];

%KF iteration
for jji = end_T+1:1:511 %Iterates over the entire validation data/SNR profile
    
    if jji>=delay_Rx+1
        
        %Interaction/mixing
        mu_h(1)=sum(pi_(1,:)'.*mu_);%Mode prob for the 2 modes
        mu_h(2)=sum(pi_(2,:)'.*mu_);
        
        mu_ij(1,1)=(pi_(1,1).*mu_(1))./mu_h(1);%Mixing prob
        mu_ij(1,2)=(pi_(1,2).*mu_(1))./mu_h(2);%Mixing prob
        mu_ij(2,1)=(pi_(2,1).*mu_(2))./mu_h(1);%Mixing prob
        mu_ij(2,2)=(pi_(2,2).*mu_(2))./mu_h(2);%Mixing prob
        
        X_1mix=((X_1*mu_ij(1,1))+(X_2*mu_ij(2,1)));%Mixing X's from all filters
        X_2mix=((X_1*mu_ij(1,2))+(X_2*mu_ij(2,2)));
        
        P_1mix=((P_1+((X_1mix-X_1)*(X_1mix-X_1)'))*mu_ij(1,1))+((P_2+((X_1mix-X_2)*(X_1mix-X_2)'))*mu_ij(2,1));
        P_2mix=((P_1+((X_2mix-X_1)*(X_2mix-X_1)'))*mu_ij(1,2))+((P_2+((X_2mix-X_2)*(X_2mix-X_2)'))*mu_ij(2,2));
        
        P_1_outage=P_1;%saves previous P1 in case of outage
        P_2_outage=P_2;%saves previous P2 in case of outage
        mu_outage=mu_;
        %Parallel Filtering
        X_new=[TOTAL(jji-delay_Rx) 0]';%Gets new measurement
        
        %Inner Filter 1
        X_1=F_1*X_1mix;
        P_1=F_1*P_1mix*F_1'+Q_1_val;
        
        K_1=P_1*H_1'*((H_1*P_1*H_1'+R)\1);
        e_1=((H_1*F_1*X_new)-(H_1*X_1));
        X_1=X_1+(K_1*e_1);
        P_1=(I-K_1*H_1)*P_1;
        
        pv1_IMM(jji-delay_Rx)=X_1(1); %prediction 1-step ahead
        
        %Inner Filter 2
        X_2=F_2*X_2mix;
        P_2=F_2*P_2mix*F_2'+Q_2_val;
        
        K_2=P_2*H_2'*((H_2*P_2*H_2'+R)\1);
        e_2=((H_2*F_2*X_new)-(H_2*X_2));
        X_2=X_2+(K_2*e_2);
        P_2=(I-K_2*H_2)*P_2;
        
        pv2_IMM(jji-delay_Rx)=X_2(1); %prediction 1-step ahead
        
        
        %Update likelihoods
        L_1=((abs(2*pi*(H_1*P_1*H_1'+R)))\1)*exp(-0.5*e_1'*((H_1*P_1*H_1'+R)\1)*e_1);
        L_2=((abs(2*pi*(H_2*P_2*H_2'+R)))\1)*exp(-0.5*e_2'*((H_2*P_2*H_2'+R)\1)*e_2);
        %Update model probs
        mu_(1)=(mu_h(1)*L_1)/((mu_h(1)*L_1)+(mu_h(2)*L_2));
        mu_(2)=(mu_h(2)*L_2)/((mu_h(1)*L_1)+(mu_h(2)*L_2));
        
        mu_old(:,:,jji)=mu_; %Saves previous mode probs
        %Combination
        X=((X_1*mu_(1))+(X_2*mu_(2)));
        P=((P_1+((X-X_1)*(X-X_1)'))*mu_(1))+((P_2+((X-X_2)*(X-X_2)'))*mu_ij(2));
        
        pv(jji-delay_Rx)=X(1);%IMM output 1-step ahead
        
        if isnan(X(1))%Outage actions for LMS channels (clear sky or rain)  [SAME ACTIONS FOR OUTAGE IN SPACE]                   
            pv(jji-delay_Rx)=-100;%any very small value
            pv_k_ahead(jji-delay_Rx+d_t_k_ahead)=-100;%any very small value
            %Outage time series
            outage_(jji)=1;
            if outage_flag==0
                %Saves the last useble value to be used when system recovers from outage 
                X_1(1)=pv1_IMM(jji-1-delay_Rx);
                X_2(1)=pv2_IMM(jji-1-delay_Rx);
                P_1=P_1_outage;
                P_2=P_2_outage;
                mu_=mu_outage;
                %Flag system outage
                outage_flag=1;
                %In case of consecutive outages
                P_1_outage=P_1;
                P_2_outage=P_2;
                mu_outage=mu_;
            elseif outage_flag==1                
                P_1=P_1_outage;
                P_2=P_2_outage;
                mu_=mu_outage;                
            end
            continue
        end
        
        outage_flag=0;%resets flag; assures that the saved values are those only before the outage
        
        pred_k_ahead=F_1_k_ahead*X; %IMM output prediciton k-steps ahead
        pv_k_ahead(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead(1);%predicted value k_steps ahead using best Q_1 and Q_2
    end
end
error_1_val=pv((end_T+1-delay_Rx):511)-TOTAL((end_T+1-delay_Rx):511);%IMM residual 1-step ahead
sum_abs_error_IMM_1step_val=sum((abs(error_1_val)).^2)  %IMM sum of squared residuals 1-step 

error_2_val=pv_k_ahead((end_T+1-delay_Rx+d_t_k_ahead):511)-TOTAL((end_T+1-delay_Rx+d_t_k_ahead):511);%IMM residual k-steps ahead
sum_abs_error_IMM_5step_val=sum((abs(error_2_val)).^2)%IMM sum of squared residuals k-steps



%% Validation two independent filters (for comparison of results only) [REMOVE]
d_t=1;
F_1=[1 d_t; 0 1];
F_2=F_1;
H_1=[1 0];
H_2=H_1;
I=eye(2);
F_1_k_ahead=[1 d_t_k_ahead; 0 1];

    P_1=1000*eye(2);%Cov matrix of state-estimation error
    P_2=1000*eye(2);
    
    X_1=[exp_power 0]';
    X_2=[exp_power 0]';    
    
%KF iteration
for jji = end_T+1:1:511 %Iterates over the entire SNR profile
    
    if jji>=delay_Rx+1      
        %Parallel Filtering
        X_new=[TOTAL(jji-delay_Rx) 0]';%Gets new measurement
        
        % Inner Filter 1
        X_1=F_1*X_1;
        P_1=F_1*P_1*F_1'+Q_1_val;
        
        K_1=P_1*H_1'*((H_1*P_1*H_1'+R)\1);
        e_1=((H_1*F_1*X_new)-(H_1*X_1));
        X_1=X_1+(K_1*e_1);
        P_1=(I-K_1*H_1)*P_1;
        
        pv1(jji-delay_Rx)=X_1(1);
        
        % Inner Filter 2
        X_2=F_2*X_2;
        P_2=F_2*P_2*F_2'+Q_2_val;
        
        K_2=P_2*H_2'*((H_2*P_2*H_2'+R)\1);
        e_2=((H_2*F_2*X_new)-(H_2*X_2));
        X_2=X_2+(K_2*e_2);
        P_2=(I-K_2*H_2)*P_2;
        
        pv2(jji-delay_Rx)=X_2(1); 
        
        pred_k_ahead_1=F_1_k_ahead*X_1;
        pv_k_ahead_1_comp(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead_1(1);%predicted value k_steps ahead using best Q_1 and Q_2
        
        pred_k_ahead_2=F_1_k_ahead*X_2;
        pv_k_ahead_2_comp(jji-delay_Rx+d_t_k_ahead)=pred_k_ahead_2(1);%predicted value k_steps ahead using best Q_1 and Q_2
    end
end
clear error_1_val
error_1_val=pv_k_ahead_1_comp((end_T+1-delay_Rx+d_t_k_ahead):511)-TOTAL((end_T+1-delay_Rx+d_t_k_ahead):511);
sum_abs_error_1_val_comp=sum((abs(error_1_val)).^2)
clear error_2_val
error_2_val=pv_k_ahead_2_comp((end_T+1-delay_Rx+d_t_k_ahead):511)-TOTAL((end_T+1-delay_Rx+d_t_k_ahead):511);
sum_abs_error_2_val_comp=sum((abs(error_2_val)).^2)
toc

%% Plots [REMOVE]

figure(1)
%1-step ahead
h1=subplot(3,1,1)
plot(TOTAL(180:320))
hold on
plot(pv(180:320))
hold on
plot(pv1(180:320))
hold on
plot(pv2(180:320))
% xlabel('Time (secs)')
set(gca,'XTick',[]);
ylabel('Amplitude (dB)')
grid on
legend('SNR','IMM 1 step','Filter1 1 step','Filter2 1 step')
set(gca,'Fontsize',15);
p1 = get(gcf,'Position');
set(gcf,'Position',[p1(1) p1(2) 1600 1000])

%5-steps ahead
h2=subplot(3,1,2)
plot(TOTAL)
hold on
temp_=zeros(1,511);
temp_(end_T+1:511)=pv_k_ahead(end_T+1:511);%IMM 5-step
plot(temp_)
hold on
temp_=zeros(1,511);
temp_(end_T+1:511)=pv_k_ahead_1_comp(end_T+1:511);
plot(temp_)
hold on
temp_=zeros(1,511);
temp_(end_T+1:511)=pv_k_ahead_2_comp(end_T+1:511);
plot(temp_)
% xlabel('Time (secs)')
set(gca,'XTick',[]);
ylabel('Amplitude (dB)')
grid on
legend('SNR','IMM 5 steps','Filter1 5 steps','Filter2 5 steps')
set(gca,'Fontsize',15);
p2 = get(gcf,'Position');
set(gcf,'Position',[p2(1) p2(2) 1600 1000])

%IMM_mu
h3=subplot(3,1,3)
mu_1=squeeze(mu_old(1,1,1:1:end))';
mu_2=squeeze(mu_old(2,1,1:1:end))';
state_=zeros(1,length(mu_1));
xx=1:1:length(mu_1);
for ij = 1:length(mu_1)%Discretizing the states
    if mu_1(ij)>=0.5
        state_(ij)=1;
    elseif mu_2(ij)>=0.5
        state_(ij)=2;
    end
end
[hAx,hLine1,hLine2] =plotyy(xx,state_,[xx',xx'],[mu_1',mu_2']);
grid(hAx(1),'on')
legend('State','mu Filter1','mu Filter2')
xlabel(hAx(1),'Time (secs)') % label x-axis
ylabel(hAx(1),'State') % label left y-axis
ylabel(hAx(2),'Probability') % label right y-axis
set(hAx,'FontSize',15)
p3 = get(gcf,'Position');
set(gcf,'Position',[p3(1) p3(2) 1600 1000]) 