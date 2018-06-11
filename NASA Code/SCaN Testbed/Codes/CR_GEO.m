rng('default')
clear all
close all; clc



for mission_num=1:5
clearvars -except mission_num

load('recv_amp_rain.mat')
recv_amp_rain=[recv_amp_rain transpose(zeros(540-length(recv_amp_rain),1))];

multi_iterations=1001;
h_hist=zeros(multi_iterations,20);
f_observed_all=zeros(multi_iterations,540);

for iii=1:multi_iterations
  %GEO
  %TOTAL=ones(1,508);
  TOTAL=ones(1,540);

%% Initializing variables

%Adaptation parameters
%Modulation order (QAM)
M_=[4 16 64];
%FEC rate
n_=[15 7];%encoding
k_=[11 4];%encoding
FEC=[1 2];%Encoding modes (1=(15/11), 2=(7/4))

frame_size=2048*8;% in bits
R_=[4:8:68]; % number of packets
P_=[0,20];% Additional Transmission power
Eb_min=(10^(min(P_)/10))/(frame_size*min(R_));
Eb_max=(10^(max(P_)/10))/(frame_size*max(R_));

%Range of Eb (Energy per bit) values
Eb_interval=(Eb_max-Eb_min)/10;%10 values
Eb_=[Eb_min:Eb_interval:Eb_max-Eb_interval];


tr=0;% threshold

episilon=ones(1,length(TOTAL));%epsilon=Exploration probability
bonus=zeros(3,1);

%normalization
T_max=frame_size*max(R_);% bits
P_min_lin=(frame_size*min(R_)*Eb_min); % dB
P_min=10*log10(P_min_lin);% Minimum additional power

%Designer paremeters
BER_min=1e-3;%
BW=512e3; % Hz. 
SNR_std=9 ;% SNR

%U matrix - all combinations between R_,Es_,M_,n_,k_
U=(combvec(R_,Eb_,M_,FEC))';
        
s0=1;% Initialization control to start as random exploration

%Communication mission phases/scenarios 
mission =...
   [0.1  0.6  0.2  0.1;     %Launch/re-entry (1)
    0.6  0.3  0.05 0.05;    %Multimedia      (2)
    0.2  0.1  0.6  0.1;     %Power saving    (3)
    0.25 0.25 0.25 0.25;    %Normal          (4)
    0.1  0.2  0.1  0.6];    %Cooperation     (5)
   %X X X X                 %Emergency       (6)

w=mission(mission_num,:); %Select communication mission.

%Q-values matrix   
q_rows=((1-tr)/0.01);
q_cols=length(U);
Q=zeros(q_rows,q_cols);

alpha=1.+Q;%Learning rate matrix

n=2;%Initial exploration probability divider
Qmax=0;

%%  MAIN SIMULATION ITERATION

reset_hop=0;%reset hop
Qmax2_ii=0; %index of new Qmax
episilon_reset_lim=4e-3; %reset episilon when episilon < episilon_reset_lim
fib=zeros(1,1000+2);

for jji = 1:1:540
        
    %Choose action u (index ii)
    
    %Current max Q
    Qmax=(max(find((sum(Q,2))>0)));%current max Q value
    [Qmax,Qmax_ii]=max(Q(Qmax,:));%current max Q location
    
    %Compute epsilon(jji) %e-greedy function of time
    e_sum=sum(Q,1);
    e_sum(find(e_sum>0))=1;%logic marker to indicate which actions were already tried
        %Decrease exploration rate/increase exploitation
        if jji>2 && episilon(jji-1)> episilon_reset_lim
            episilon(jji)=episilon(jji)/n; % n is # os times system exploited config. After X exploitation, needs to be reset            
            n=n+1;
            n_fib=n_fib+1;
        else
            n=1;%RESET epsilon  
            n_fib=1;
        end

    hop=floor(exp(n_fib));
    
    %epsilon=Exploration probability
    if s0==0       
        e_prob=rand; %pick Explore or Exploit
        
        %Explore
        if e_prob<=episilon(jji)
            %"Random exploration"
            if e_prob>=(episilon(jji))/2 || isempty(find(e_sum<1))
                ii=0;
                while ii<=0 || ii>length(U)%Prevents selecting actions outside range of U
                    ii=ceil(rand*length(U));
                end
            
            %"Guided exploration"
            else
                %Explore nearby
                if e_prob>=(episilon(jji))/8
                    
                    %Constraint hop to U matrix borders
                    if hop>Qmax_ii
                        hop=Qmax_ii-1;
                    elseif hop +Qmax_ii > length(U)
                        hop=length(U)-Qmax_ii;
                    end
                    
                    ii=0;
                    while ii<=0 || ii>length(U) %Prevents selecting actions outside range of U
                        ii=Qmax_ii+(randi([-hop,hop]));
                    end
                %Explore among actions never tried before
                else                    
                    e_sum2=find(e_sum<1);
                    ii=0;
                    while ii<=0 || ii>length(U)
                        ii=e_sum2(ceil(rand*length(e_sum2))); %Prevents selecting actions outside range of U
                    end
                end
            end
        
            
        %Exploit    
        else 
            %Find max state/row with non-zero Q-value
            max_state=max(find((sum(Q,2))>0));            
            [~,ii]=max(Q(max_state,:));%Action/column with max Q-value for the highest state/performance achieved at the moment
            
            if ii==Qmax2_ii && reset_hop==1%new Qmax is being used
                n_fib=1;%reset hop for new local search
                reset_hop=0;%reset hop flag
            end       
        end
    else
        %only applies for s0=1 mode
        s0=0;%reset s0
        ii=ceil(rand*length(U));%First time always explore randomly
    end

    action(jji)=ii;%Chosen action
    if iii==1001
        action(jji)=jji;ii=jji; %Chosen action (brute force)
    end
    
    %Parameters for chosen action
    R=U(ii,1);
    Eb=U(ii,2);
    M=U(ii,3); k=log2(M);    
    n_cod=n_(U(ii,4));
    k_cod=k_(U(ii,4));
    
%% Tx:
    %Transmitted power
    P_lin=frame_size*R*Eb;% Tx power
    P(jji)=10*log10(P_lin);% dB 
    %Bandwidth
    W(jji)=R/k;% Hz
    %Throughput
    T(jji)=frame_size*R;% bits/sec
    
    %Increase Es
    if P(jji)~=0        
        SNR=TOTAL(jji)+P(jji)+recv_amp_rain(jji);
    else
        SNR=TOTAL(jji);
    end        
    
%%Rx:
    
    %BER estimation            
    Measured_EbNo(jji)=SNR-10*log10(k*(k_cod/n_cod)); % Removed -3dB for matching between True and Estimated BER
        %BER measured
       
if Measured_EbNo(jji)<0
    BER_est(jji)=1;
else
        BER_est(jji)=((((sqrt(M))-1)/((sqrt(M))*(log2(sqrt(M)))))*(erfc(sqrt((3*(log2(M))*Measured_EbNo(jji))/(2*(M-1))))))+((((sqrt(M))-2)/((sqrt(M))*(log2(sqrt(M)))))*(erfc(3*(sqrt((3*(log2(M))*Measured_EbNo(jji))/(2*(M-1)))))));
end  
       
    %f_observed reference parameters
    f_max_T=T(jji)/T_max;    
    if f_max_T>=1
        bonus(1)=f_max_T;
        f_max_T=1;
    end
    
    f_min_BER=BER_min/BER_est(jji);
    if f_min_BER>=1
        bonus(2)=f_min_BER; 
        f_min_BER=1;
    end
    
    f_min_P=P_min_lin/P_lin;
    if f_min_P>=1
        bonus(3)=f_min_P; 
        f_min_P=1;        
    end
      
    if W<=BW%Bandwidth 
        f_const_W=0;
    else
        f_const_W=-1;
    end 
        
    %Observed state
    f_observed(jji)=(w(1)*f_max_T)+(w(2)*f_min_BER)+(w(3)*f_min_P)+(w(4)*f_const_W);       
    
    %Analysis of fitness function
    if f_observed(jji)>tr 
        %ij=single((round(f_observed(jji),2)-tr)*100)+1;
        ij=single((round(f_observed(jji)*100)/100-tr)*100)+1;
        
        %Reward 
        f_reward=f_observed(jji)-tr; 
        
        if (any(Q(:,ii)>0))==1
             Q(:,ii)=zeros(size(Q,1),1);
             alpha(:,ii)=ones(size(Q,1),1);
        end
            
        %Q-function matrix update (size restricted to states above tr only)
        Q(ij,ii)=Q(ij,ii)+ (alpha(ij,ii)*f_reward);
        
        %Checks Q bigger than current Qmax and if it has a different action ID
        if jji>1 && Q(ij,ii)> Qmax && Qmax_ii~=ii
            Qmax2_ii=ii;%saves index of new Qmax found
            reset_hop=1;%flag for hop reset when Qmax2_ii is used
        end
      
        if alpha(ij,ii)<=1e-3
            alpha(ij,ii)=1e-3; 
        else
            alpha(ij,ii)=alpha(ij,ii)/2;
        end
                
        bonus=zeros(3,1);
    else
        f_reward=0; 
    end
    
    disp(iii)

end %End of Main 

edges=[0:0.05:1];
if iii==1001
   %h_hist_bf=histcounts(f_observed,edges,'Normalization','probability');%brute force
   h_hist_bf=histc(f_observed,edges(2:end));
end

f_observed_all(iii,:)=f_observed;

%h_hist(iii,:)=histcounts(f_observed,edges,'Normalization','probability');
h_hist(iii,:)=histc(f_observed,edges(2:end));


end %Multiple iterations

%Post-process multiple-iterations
mean_f_observed_main=mean(f_observed_all(1:1000,:),1);
mean_f_observed_bf=mean(f_observed_all(1001,:),1);

%Error bars
mean_hist=mean(h_hist(1:1000,1:20),1);
std_dev_hist=std(h_hist(1:1000,1:20),1);

quantile_main = quantile(mean_f_observed_main,[0.25 0.50 0.75 0.975]);
save(char(strcat({'Quantile_main_Mission_'},num2str(mission_num),{'.mat'})),'quantile_main');

quantile_bf =quantile(mean_f_observed_bf,[0.25 0.50 0.75 0.975]);
save(char(strcat({'Quantile_bf_Mission_'},num2str(mission_num),{'.mat'})),'quantile_bf');

%Integrals
int_main=sum([0.025:0.05:.975].*mean_hist);
save(char(strcat({'Integral_main_Mission_'},num2str(mission_num),{'.mat'})),'int_main');

int_bf=sum([0.025:0.05:.975].*h_hist_bf);
save(char(strcat({'Integral_bf_Mission_'},num2str(mission_num),{'.mat'})),'int_bf');


x_error=[2.27/2:1:21];
z=[h_hist_bf;mean_hist];


barh(z',1)
axis([0,1,0,20])
set(gca,'YTick',linspace(1,20,20))
set(gca,'YTickLabel',{[0.025:0.05:.975]})
hold on
%h2=herrorbar(mean_hist,x_error,std_dev_hist./(sqrt(multi_iterations)));
grid on
set(gca,'FontSize',10)
xlabel('Normalized time duration')
ylabel('Normalized multi-objective performance')
title(strcat({'Mission '},num2str(mission_num)))
fig_name=char(strcat({'hor_Mission_'},num2str(mission_num),{'.fig'}));
savefig(fig_name)
close('all')
end