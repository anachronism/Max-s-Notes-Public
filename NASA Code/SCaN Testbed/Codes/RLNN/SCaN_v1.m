tic

rng('default')
clear all
close all; clc

episode_dur=21600;


for mission_num=4 %[1:5]
  clearvars -except mission_num episode_dur 


  %% Build NN model
  
%Custom NN
net=network(1,3,[0;0;0], [1 ; 0 ; 0], [ 0 0 0; 1 0 0; 0 1 0], [0 0 1]);
%NN input size
net.inputs{1}.size=6; % RL measurements
%NN input range values
net.inputs{1}.range = [-1 1; -1 1; -1 1; -1 1; -1 1; -1 1];

%NN train function
net.trainFcn = 'trainlm';
%NN dataset division function (training, validation, test)
net.divideFcn='dividerand'; % 70%,15%,15% default


%NN output functions (help nntransfer)
net.layers{1}.size = 6;
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.size = 6;
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'purelin';

%Early stop conditions
net.trainParam.max_fail=20;
net.trainParam.min_grad=1e-12;

%Number of parallel NN
numNN=100;
NN = cell(1,numNN);
perfs=zeros(1,numNN);

%Percentage to exclude from training buffer for next retraining
nn_delete=0.5; %after training, delete 50% of oldest actions and retrain only after these percentage of training data is replaced by new training data

% Percentage of training data used for training (remaining training is used for parellel testing 
nn_parallel_train=0.9; %(90% for training and 10% for parellel testing)


%Flag
NN_train=0; %checks if NN was trained and controls when to train it again
  
%% RL iterations

  for iii=1:1
    %% Load Channel --> 0=GEO; 1=LEO
    cn=0;

    if cn==1
      % Fixed - LEO Channel [CLEAR SKY or RAIN]
      load ('L_fs.mat') %(LEO time series) Clear sky SNR profile at fixed ground receiver
      TOTAL=(L_fs-max(L_fs))*-1;

    else
      % Fixed - GEO Channel [CLEAR SKY or RAIN]
        TOTAL=9*ones(1,episode_dur); %GEO clear sky SNR profile >>> 1000 seconds of constant 9 dB SNR profile
    end

    %% Initializing variables

    %Adaptation parameters
    
    %Modulation order (QAM)
    M_=[4 16 64];
    %FEC rate
    n_=[15 7];%encoding
    k_=[11 4];%encoding
    enc_rate=[15/11 7/4];%Encoding modes (1=(15/11), 2=(7/4))

    %Data rate
    %CCSDS Proximity-1 Space Link Protocol frame size (2048*8) bits
    %Rate (R) ranges from 4*(2048*8):2*(4*(2048*8)):17*(4*(2048*8)) so that R is bigger than R=[56e3:128e3:1e6];
    %Rate (R) is changes by varying # of packets sent [4:8:68]*(2048*8)

    frame_size_=[2048*8 2048*2*8 2048*3*8 2048*4*8];% in bits
    R_=[4:8:68]; % number of packets sent per second controls link data rate
    P_=[0,20];% Additional [min, max] transmission power [dB] provided by the output amplifier dBW (to be added to SNR as input of AWGN for simulation)
    % Convert min,max transmission power to Eb [min,max]
    Eb_min=(10^(min(P_)/10))/(max(frame_size_)*max(R_));
    Eb_max=(10^(max(P_)/10))/(min(frame_size_)*min(R_));
    %Transmit power: 10*log10(frame_size*max(R_)*Eb_max)

    %Range of Eb (Energy per bit) values
    Eb_interval=(Eb_max-Eb_min)/100; %100 values
    Eb_=[Eb_min:Eb_interval:Eb_max-Eb_interval]; %Possible Eb values


    %   -->AI parameters<--
    tr=0;% RL state reward threshold
    episilon=ones(1,length(TOTAL));%epsilon=Exploration probability

    %Values used for normalization of monitored parameters
    T_max=max(frame_size_)*max(R_);% throughput [bits]
    P_min_lin=(min(frame_size_)*min(R_)*Eb_min); % min additional linear power
    P_min=10*log10(P_min_lin);% min additional power [dB]

    %Designer paremeters: comms target performance
    BER_min=1e-3;% max BER
    BW=512e3; % Constant bandwidth [Hz]
    SNR_std=9 ;% Expected SNR value for AWGN channel under clear sky conditions [dB]

    %U matrix - all parameter combinations between R_,Es_,M_,n_,k_,frame_size
    U=(combvec(R_,Eb_,M_,enc_rate,frame_size_))';
    
    %Add additional features for NN training
    U=[U log2(U(:,3))];
    
    %Get Actions normalization function (rows are features). Used to build normalized training set
    [~,ps] = mapminmax(U');

    s0=1;% Initialization control to start as random exploration

    %Communication mission phases/scenarios
    %     [Thrp, BER, Pwr, W];
    mission =...
    [0.1  0.6  0.2  0.1;    %Launch/re-entry (1)
    0.6  0.3  0.05 0.05;    %Multimedia      (2)
    0.2  0.1  0.6  0.1;     %Power saving    (3)
    0.25 0.25 0.25 0.25;    %Normal          (4)
    0.1  0.2  0.1  0.6];    %Cooperation     (5)    
    

    w=mission(mission_num,:); %Select communication mission.

    %Q-values matrix        
    q_rows=((1-tr)/0.005);%generates 200 rows (up to 200 performance levels)100=100%, 200=200% of fitness function [more than 200%, i.e., 2x the maximum multi-objective performance expected, are allocated at 200th row]
    q_cols=length(U);
    Q=zeros(q_rows,q_cols);

    alpha=1.+Q;%Learning rate matrix 

    n=2;%Initial exploration probability divider
    Qmax=0;

    %%  MAIN SIMULATION ITERATION
    
    reset_hop=0; %reset hop flag
    Qmax2_ii=0;  %index of new Qmax
    episilon_reset_lim=4e-3; %reset episilon when episilon < episilon_reset_lim
    fib=zeros(1,1000+2);

    %Log observables
    log_f_max_T=zeros(1,episode_dur);
    log_f_min_BER=zeros(1,episode_dur);
    log_f_min_P=zeros(1,episode_dur);
    log_f_const_W=zeros(1,episode_dur);
    

    %Initializing NN training set matrix

    history_size=100; % <<<<< NN training window with UNIQUE actions (and its respective performance) 
    
    hist_count=0;
    ind_mean=0;
    temp_hist_=zeros(history_size,3); %Sliding window of last action-parformance data [acdionID f_observed time_stamp]
    hist_=zeros(history_size,size(U,2)+1); %[action_parameters f_observed]
    f_observed_predic=zeros(1,size(U,1));
    action_pred=zeros(1,episode_dur);
    
    perf_th=0.65; % performance threshold for RLNN action exploration decision making
    rejection_rate=1; %rejection percentage (rate) of exploring actions which performance predictions fall below the performance_threshold value
        
    rej_counter=0;
    rej_accept=0;
    
    train_ind=[];% indices when NN training took place
        
    explore_control=0; %used in while loop during explore_mode=1 ONLY (starts as 0 always to be run once)
    
    for jji = 1:1:episode_dur %time series length      
        %% AI functions 
      % [For implementation/production >> include parameter change monitoring feature]
      % If (R_max, Es_max, M_max, n_max, k_max) changes
      %   Updates matrix U in case adaptable parameters changed
      %   Update reference parameters
      %   Update Q-values matrix size (append zeros)

      %Choose action u (index ii)
      %Current max Q
      Qmax=(max(find((sum(Q,2))>0)));%current max Q value
      [Qmax,Qmax_ii]=max(Q(Qmax,:));%current max Q location

      %Compute epsilon(jji) %e-greedy function of time
      %After x % of possible actions have been tried out, start to decrease exploration probability (increase Exploitation)
      e_sum=sum(Q,1);
      e_sum(find(e_sum>0))=1;%logic marker to indicate which actions were already tried

      %Decrease exploration rate/increase exploitation
      if jji>2 && episilon(jji-1)> episilon_reset_lim
        episilon(jji)=episilon(jji)/n; % n is # os times system exploited config. After X exploitation, needs to be reset. >>> What is the best value of X ???

        n=n+1;
        n_fib=n_fib+1;
      else
        n=1;%RESET epsilon
        n_fib=1;
      end

      %Compute alpha(jji) %learning rate proportional to exploration value (which is function of overall observed function value).

      hop=floor(exp(n_fib));

      %(Exploit or Explore) epsilon=Exploration probability
      if s0==0
        e_prob=rand; %Randomly (uniformly) pick Explore or Exploit

        %Explore
        if e_prob<=episilon(jji)
            
            while explore_control==0
                
                %"Random exploration" 1/2% of time
                if e_prob>=(episilon(jji))/2 || isempty(find(e_sum<1)) %(all actions were already tried)
                    ii=0;
                    while ii<=0 || ii>length(U)%Prevents selecting actions outside range of U
                        ii=ceil(rand*length(U));
                    end
                    
                    %"Guided exploration" 1/2% of time
                else
                    %Explore nearby 1/4% of time
                    if e_prob>=(episilon(jji))/4
                                                
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
                        
                        %Explore among actions never tried before during 1/4 of time
                    else
                        e_sum2=find(e_sum<1);
                        ii=0;
                        while ii<=0 || ii>length(U)
                            ii=e_sum2(ceil(rand*length(e_sum2))); %Prevents selecting actions outside range of U
                        end
                    end
                end
                if NN_train==0 %exit while everytime NN is not trained yet
                    explore_control=1; %exit while
                end
                
                
                %NN Prediction
                %Predictions are done every iteration ONLY after NN is trained and ONLY if exploring
                if NN_train==1
                    input_norm = mapminmax('apply',U(ii,:)',ps);%Applying normalization function ps to new training input
                    if ind_NN~=0 % use NN that resulted in min MSE
                        f_predic=NN{ind_NN}(input_norm);
                    else % use all NN and get the average
                        y_pred=0;
                        for i_NN=1:numNN
                            y_pred=y_pred+NN{i_NN}(input_norm);
                        end
                        f_predic=y_pred/numNN;
                    end%                     
                    
                    %Logic to decide on actions predicted by NN
                    if f_predic >= perf_th
                        explore_control=1; %exit while    
                        
                    else
                        if rand>=rejection_rate %allows bad performances to be learnt only for a time percentage higher than rejection_rate%
                            explore_control=1; %exit while                            
                            rej_accept=rej_accept+1;
                            
                        end
                        rej_counter=rej_counter+1; %all below the threshold performance predictions of non-executed actions, after NN training
                        f_observed_predic(jji)=f_predic;
                        e_prob=episilon(jji)*rand; %draws new probability number for new exploration attempt 
%                         
                    end
                end
            end %while
            explore_control=0; % allows exploration in next iteration (depending on the value drawn)
            
            
            
            %Exploit
        else
            %Find max state/row with non-zero Q-value
            max_state=max(find((sum(Q,2))>0));
            [~,ii]=max(Q(max_state,:)); %Action/column with max Q-value for the highest state/performance achieved at the moment
            
            if ii==Qmax2_ii && reset_hop==1%new Qmax is being used
                n_fib=1;%reset hop for new local search
                reset_hop=0;%reset hop flag
            end
        end

      %only applies for s0=1 mode
      else
        s0=0;%reset s0
        ii=ceil(rand*length(U));%First time always explore randomly
      end

      action(jji)=ii;%Chosen action time-series

      if iii==1001
        % Brute force mode
        ii=jji; %Chosen action time-series
        action(jji)=jji;
      end


      %Parameters for chosen action
      R=U(ii,1);
      Eb=U(ii,2);
      M=U(ii,3); k=log2(M);
      enc_rate_=U(ii,4);
      frame_size=U(ii,5);

      %% Measured at Tx (sent to Rx as telemetry data):

      %BEFORE BEING AFFECTED BY CHANNEL DYNAMICS

      %Transmitted power
      P_lin=frame_size*R*Eb;% Tx power in [linear]
      P(jji)=10*log10(P_lin);% [dB] (added to SNR for simulation of AWGN channel)

      %Bandwidth
      W(jji)=R/k;% [Hz]

      %Throughput
      T(jji)=frame_size*R;% [bits/sec]

      %Amplifier - Increase Es (energy per symbol)
      if P(jji)~=0
        %y=y.*(10^(Es/20));%Considering modulator output is 1 linear/0 dBW.
        SNR=TOTAL(jji)+P(jji);
      else
        SNR=TOTAL(jji);
      end

      %% AI at Rx:
      %Measured at Rx:

      %BER estimation
      %Actual SNR=Additional + Channel SNR
      %1) EbNo: SNR measurements at Rx, assumed without noise, for BER estimation (after FEC)
      Measured_EbNo(jji)=10^((SNR-10*log10(k*(enc_rate_)))/10); 
      %BER measured
      %2) BER: [using Eq. 17 from CHO AND YOON: ON THE GENERAL BER EXPRESSION IEEE Trans. on Comms, 2002]
      if Measured_EbNo(jji)<0
        BER_est(jji)=1;
      else
        BER_est(jji)=((((sqrt(M))-1)/((sqrt(M))*(log2(sqrt(M)))))*(erfc(sqrt((3*(log2(M))*Measured_EbNo(jji))/(2*(M-1))))))+((((sqrt(M))-2)/((sqrt(M))*(log2(sqrt(M)))))*(erfc(3*(sqrt((3*(log2(M))*Measured_EbNo(jji))/(2*(M-1)))))));
      end

      %(The following are sent out by the Tx as telemetry data:)
      %-Transmitted power computed at Tx.
      %-Bandwidth computed at Tx and not affected by channel (Assumed no presence of interferes).

      %Multi-objective fitness function (f_observed reference parameters)
      f_max_T=T(jji)/T_max;

      f_min_BER_true=(100-(-1*(log10(BER_min/BER_est(jji))))*(-100/(log10(BER_min))))/100; %Function value range from 1 to BER_min scaled to 0 to 1.

      %f_min_BER for reward (Q-function) and NN training_set
      if f_min_BER_true>=1
          f_min_BER=1;
      else
          f_min_BER=f_min_BER_true;
      end
      
      f_min_P=P_min_lin/P_lin;

      if W<=BW%Bandwidth
        % f_const_W=0;
        f_const_W=1; %No penalty if W is smaller than target BW (cannot cause interference)
      else
        f_const_W=1-((W-BW)/BW); %If W is more than double the target BW
        if f_const_W<0
          f_const_W=0;
        end
      end

      %Observed state: fitness function
      f_observed(jji)=(w(1)*f_max_T)+(w(2)*f_min_BER)+(w(3)*f_min_P)+(w(4)*f_const_W);
      
%       %NN target
%       nn_target=[nn_target; [f_max_T f_min_BER f_min_P f_const_W]];
      
% %       %Creating dataset examples for NN training
% %       f_observed=f_observed';
% %       save('train_data.mat','nn_input','f_observed')
% %       f_observed=f_observed';

      %Logging function measurebles
      log_f_max_T(jji)=f_max_T;
      log_f_min_BER(jji)=f_min_BER;
      log_f_min_P(jji)=f_min_P;
      log_f_const_W(jji)=f_const_W;      


      %Analysis of fitness function
      if f_observed(jji)>tr 
        ij=single((round(f_observed(jji),2)-tr)*100)+1;

        if ij>= 200 %all performance levels equal to or higher than 100 is located at 100 column on Q-table
            ij=200;
        end
        %Reward function 
        f_reward=f_observed(jji)-tr;  %positive linear function starting from tr

        %Find previous non-zero state for this same action
        if (any(Q(:,ii)>0))==1
          Q(:,ii)=zeros(size(Q,1),1);%set all Q-values for current action ii to zero
          alpha(:,ii)=ones(size(Q,1),1);%Also resets the learning rate vector for that action
        end
        
        %Update Q-values
        Q(ij,ii)=(Q(ij,ii)*(1-alpha(ij,ii)))+ (alpha(ij,ii)*f_reward);       
        
        %Check if new Q is bigger than current Qmax and if it has a different action ID
        if jji>1 && Q(ij,ii)> Qmax && Qmax_ii~=ii
          Qmax2_ii=ii;%saves index of new Qmax found
          reset_hop=1;%flag for hop reset when Qmax2_ii is used
        end



        %Learning rate decreases by half for each element, after each time it is used.
        if alpha(ij,ii)<=1e-3
          alpha(ij,ii)=1e-3; %Minimum learning rate. Stops decreasing the learning rate to avoid infinite division of very small numbers.a
        else
          alpha(ij,ii)=alpha(ij,ii)/2;
        end

        bonus=zeros(3,1);
      else
        f_reward=0; % Does not give a reward if the state achieved is not higher than the threshold
      end
  
    
      %History sliding window 
      if isempty(find(temp_hist_(:,1)==ii)) %chosen action not present in sliding window
          if hist_count<history_size %if sliding window not full yet
              hist_count=hist_count+1; %populate
              ind_update=(temp_hist_(:,3)>=1);%index for update
              temp_hist_(hist_count,:)= [ii f_reward 1]; % [actions f_observed time_stamp]
              temp_hist_(ind_update,3)=temp_hist_(ind_update,3)+1; %update all action IDs   
          else %if sliding window already full >>> replace
              [~,jj]=max(temp_hist_(:,3)); %find action with oldest/highest time_stamp              
              temp_hist_(jj,:)= [ii f_reward 0]; %replace it
              temp_hist_(:,3)=temp_hist_(:,3)+1; %update all action IDs   
          end
      else %chosen action is already on sliding window          
          if action(jji-1)~=ii %checks if current action is different to last one (if equal, do nothing)
              ind_mean=find(temp_hist_(:,1)==ii);
              ind_update=(temp_hist_(:,3)>0) & (temp_hist_(:,3)<temp_hist_(ind_mean,3));%index for update
              temp_hist_(ind_mean,:)= [ii (f_reward+temp_hist_(ind_mean,2))/2 1]; %take the mean  
              temp_hist_(ind_update,3)=temp_hist_(ind_update,3)+1; %update all action IDs      

          end
      end
      
      
% %       %Train NN
% %       if hist_count==history_size % enables training after sliding window is full
% %           train_ind=[train_ind jji]; % indices when NN training took place
% %           
% %           % Populate with acitons (only after training window was built)
% %           hist_(:,1:size(U,2))=U((temp_hist_(:,1)),:);
% %           % Populate f_observed for non-repeating actions
% %           hist_(:,end)=temp_hist_(:,2);
% %           
% %           %Applying normalization function ps to new training input
% %           pnewn = mapminmax('apply',hist_(:,1:size(U,2))',ps);
% %           examples_input= [pnewn'];
% %           examples_target= [hist_(:,end)];
% % % % % %           save('train_data.mat','examples_input','examples_target')
% %           
% %           %Reset sliding window
% %           %Delete x percentage of oldest actions
% %           temp_hist_=sortrows(temp_hist_,3);
% %           temp_hist_((history_size-(nn_delete*history_size)):end,:)=0;
% %           hist_count=(history_size-(nn_delete*history_size))-1; %resets hist_count
% %           
% %           %Execute training NN
% %           
% %           %Splits training dataset into 90% training and 10% parallel testing          
% %           Q1 = floor(history_size*nn_parallel_train);
% %           Q2 = history_size-Q1;
% %           ind = randperm(history_size);
% %           ind1 = ind(1:Q1);
% %           ind2 = ind(Q1+(1:Q2));
% %           %Training
% %           x1 = examples_input(ind1,:)';
% %           y1 = examples_target(ind1,:)';
% %           %Parallel testing (%Guarantees all nets use same test dataset)
% %           x2 = examples_input(ind2,:)';
% %           y2 = examples_target(ind2,:)';
% %           
% %           y_total=0;
% %           for n_i=1:numNN
% %               NN{n_i}=train(net,x1,y1);              
% %               y_val=NN{n_i}(x2);
% %               perfs(n_i)=mean((y_val-y2).^2);
% %               y_total=y_total+y_val;
% %           end
% %           %Get min among all NN
% %           y_min_perfs=min(perfs);
% %           %Get average of all NN
% %           y_mean=y_total/numNN;
% %           y_mean_perf=mean((y_mean-y2).^2);
% %           
% %           ind_NN=0;
% %           %Chose which NN to use for prediction
% %           if y_min_perfs<y_mean_perf
% %               %Get index of NN that resulted in min MSE
% %               [~,ind_NN]=min(perfs);%Use this NN for predictions                            
% %           %else
% %               %Do predictions using all NN and average output
% %           end
% %           
% %           NN_train=1;
% %       end

    end %End of Main iteration
    


  end %Multiple iterations

end

%% Evaluate performance

sprintf('All f_observed less than threshold >> %d',length(find(f_observed<=perf_th)))

sprintf('Prediction < threshold after NN training >> %d', length(find(f_observed_predic(train_ind(1):end)))) %predicted to be less than threshold
sprintf('Prediction < threshold after NN training, but was accepted >> %d',rej_accept)  %predicted to be less than threshold, but that was accepted for training reasons
sprintf('f_observed < threshold after NN training >> %d',length(find(f_observed(train_ind(1):end)<perf_th))) % data points less than threshold, after first training

%performance sum
sum(f_observed) %all
sprintf('Sum after NN training >> %d',sum(f_observed(train_ind(1):end)))

plot(f_observed)
hold on
plot(f_observed_predic)

toc