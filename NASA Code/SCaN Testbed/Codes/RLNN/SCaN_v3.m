rng('shuffle')
% clear all
close all; clc

for iterations=1:1
    iterations
    episode_dur=21600; %only for analysis
    
    for mission_num=4 %[1:5] Loop over different mission profiles/fitness functions
        clearvars -except mission_num episode_dur iterations numIterations f_observed_save f_observed2_save
        f_observed=[];
        f_observed2=[];
        
        load('ber_functions_3') %loads DVB-S2 (Long frames) BER curve functions for online BER estimation using SNR measurements
        
        %% Build NN structures
        
        %--------------------------------------
        %NN Explore (NN1)
        net=network(1,3,[0;0;0], [1 ; 0 ; 0], [ 0 0 0; 1 0 0; 0 1 0], [0 0 1]);
        
        %NN input size
        net.inputs{1}.size=1;
        %NN input range values
        net.inputs{1}.range = [-1 1; -1 1; -1 1; -1 1; -1 1; -1 1; -1 1];
        
        %NN train function
        net.trainFcn = 'trainlm';
        %NN dataset division function (training, validation, test)
        net.divideFcn='dividerand'; % 70%,15%,15% default
        
        % NN output functions (help nntransfer)
        
        % NN1 Layers
        net.layers{1}.size = 7;
        net.layers{1}.transferFcn = 'logsig';
        net.layers{2}.size = 50;
        net.layers{2}.transferFcn = 'logsig';
        net.layers{3}.size = 1;
        net.layers{3}.transferFcn = 'purelin';
        
        %Early stop conditions
        net.trainParam.max_fail=20;
        net.trainParam.min_grad=1e-12;
        
        %Number of parallel NN
        numNN=20;
        NN = cell(1,numNN);
        
        %Flags [do not change]
        NN_train=0; %checks if NN was trained and controls when to train it again
        NN_train_2=0;
        NN_train_exploit=0;
        
        %--------------------------------------
        %NN Exploit (NN2)
        
        net_exploit=network(1,2,[0;0], [1 ; 0], [ 0 0 ; 1 0], [0 1]);
        
        %NN input size
        net_exploit.inputs{1}.size=1;
        %NN input range values
        net_exploit.inputs{1}.range = [-1 1; -1 1; -1 1; -1 1; -1 1; -1 1; -1 1];
        
        %NN train function
        net_exploit.trainFcn = 'trainlm';
        %NN dataset division function (training, validation, test)
        net_exploit.divideFcn='dividerand'; % 70%,15%,15% default
        
        % NN output functions (help nntransfer)
        
        %NN2 Layers
        net_exploit.layers{1}.size = 20;
        net_exploit.layers{1}.transferFcn = 'logsig';
        net_exploit.layers{2}.size = 1;
        net_exploit.layers{2}.transferFcn = 'purelin';
        
        %Early stop conditions
        net_exploit.trainParam.max_fail=20;
        net_exploit.trainParam.min_grad=1e-12;
        
        %%% TEST
        tmp_newWeights = getwb(net_exploit);%%%
        tmp_newWeights = sqrt(2/20) * randn(size(tmp_newWeights)); % 2/(20+1)
        net_exploit = setwb(net_exploit,tmp_newWeights);
        %Number of parallel NN
        
        numNN_exploit=10;
        NN_exploit = cell(6,numNN_exploit);
        %Flag
        NN_train_exploit=0; %checks if NN was trained and controls when to train it again
        
        max_f_observed=0;
        
        %% RL iterations
        
        for iii=1:1
            % Load Channel --> 0=GEO; 1=LEO
            cn=1;
            
            if cn==1
                % Fixed - LEO Channel [CLEAR SKY or RAIN]
                load ('L_fs.mat') %(LEO time series) Clear sky SNR profile at fixed ground receiver
                TOTAL=(L_fs-max(L_fs))*-1;
                
                
%                 load('esno_curves.mat');
%                 % 5 doesn't work, size = 75..
%                 tmpTotal =time_series(2).esno_viasat; %%% TODO: verify this is what should be done 
%                 %%% TODO: Incorporate logic to deal with bad connections.
%                 TOTAL = (tmpTotal - max(tmpTotal))* -1;
            else
                % Fixed - GEO Channel [CLEAR SKY or RAIN]
                TOTAL=6*ones(1,episode_dur); %GEO clear sky SNR profile >>> 1000 seconds of constant 9 dB SNR profile
            end
            %Upsample attenuation time series to 10Hz
            for i=1:1:length(TOTAL)
                TOTAL2(10*i-9:10*i)=TOTAL(i);
            end
            
            %% Initializing variables
            
            %Adaptation parameters
            
            % IN CASE **ANY** PARAMETER CHANGE ITS RANGE [MIN, MAX] the
            % NORMALIZATION 'ps' function MUST BE LEARNED again !!!
            
            mod_list = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32];
            cod_list = [1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10, 3/5, 2/3, 3/4, 5/6, 8/9, 9/10, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10, 3/4, 4/5, 5/6, 8/9, 9/10];
            
            M_=[4 8 16 32];
            k_=log2(M_);
            
            BW_max=5e6; %MHz --> Max single-frequency SCaN testbed transponder bandwidth
            BW_min=0.5e6; %MHz
            
            roll_off=[0.2 0.3 0.35]; %Squared-root raised-cosine filter roll-off factor
            
            Rs_max=BW_max/(1+max(roll_off)); %Max Symbol rate such that BW_max value is not compromised
            Rs_min=BW_min/(1+min(roll_off));
            
            T_max= Rs_max * log2(max(mod_list)) * max(cod_list); %Throughput in bits/sec
            T_min= Rs_min * log2(min(mod_list)) * min(cod_list);
            
            frame_size_=64800;% in bits --->Long-frame DVB-S2
            
            % Ranges of adaptable parameters for value scalling of NN inputs
            modcod_=1:length(mod_list); %Mod + Cod
            Rs_=Rs_min:0.1*1e6:Rs_max;    %Symbol rate range
            Es_=0:10; %Additional Es/No [dB] to boost signal before channel;
            
            %Values used for normalization of monitored parameters
            Es_min_lin=10^(min(Es_)/10);
            Es_max_lin=10^(max(Es_)/10);
            
            %Consumed Power
            P_consu_min_lin=(Es_min_lin*Rs_min); %linear
            P_consu_max_lin=(Es_max_lin*Rs_max);
            
            %Spectral efficiency
            spect_eff_max=max(log2(mod_list))*max(cod_list)/(1+min(roll_off));
            spect_eff_min=min(log2(mod_list))*min(cod_list)/(1+max(roll_off));
            
            %Consumed power efficiency
            pwr_eff_max=(max(log2(mod_list))*max(cod_list))/(Es_min_lin*Rs_min);
            pwr_eff_min=(min(log2(mod_list))*min(cod_list))/(Es_max_lin*Rs_max);
            
            %SNR
            max_SNR=12.9263; % [dB] Maximum link margin achieved during clear sky conditions for current predicted orbit (obtained from link budget)
            max_SNR_lin=10^(max_SNR/10);
            
            %Designer paremeters: comms target performance BER value
            BER_max=1e-12;% max BER
            ber_dB_max=-10.*log10(BER_max);
            ber_dB_min=-10.*log10(1);
            
            BER_min=1e-6;% min BER
            
            BW=1e6; % Constant bandwidth [Hz]
            
            %   -->AI parameters<--
            
            %RL parameters
            tr=0;% RL state reward threshold
            % [no (gamma) discount factor ->REMOVED TEMPORAL DIFFERENCE]
            episilon(1)=1;
            
            %U matrix - all parameter combinations between Rs_,Es_,modcod_,roll_off
            U=(combvec(Rs_,Es_,modcod_,roll_off))';
            
            %Add additional features for NN training
            U=[U zeros(size(U,1),1) zeros(size(U,1),1)];
            U(:,5)=mod_list(U(:,3));
            U(:,6)=cod_list(U(:,3));
            modcod_map=U(:,3);
            U(:,3)=log2(U(:,5)); %replaces modcod mapping by mod_order
            
            %Final U structure for NN training [Rs_, Es_, log2(M), roll_off, M, (n/k)]
            
            %Get Actions normalization function (rows are features). Used to build normalized training set
            [~,ps] = mapminmax(U');
            
            %Scaled action matrix U_out, range [0, 1]
            U_out=U;
            for s=1:size(U,2)
                U_out(:,s)=(U_out(:,s)-min(U_out(:,s)))/(max(U_out(:,s))-min(U_out(:,s))); % range [0,1]
            end
            
            %List to classify predicted modcod (used at NN2 output)
            a_=unique(U_out(:,3));
            b_=unique(U_out(:,5));
            x=0;
            for i=1:4
                for j=1:4
                    x=x+1;
                    sum_(x)=a_(i)+b_(j); %[0 0.14 0.33 0.44 0.47 0.66 0.77 0.8 1 1 1.1 1.14 1.33 1.44 1.66 2] Classification targets are: [0 .4762 1.0952 2]
                end
            end
            
            s0=1;% Initialization control to start as random exploration until NN gets trained
            
            %Communication mission phases/scenarios
            %     [Thrp, BER, BW, Spec_eff, Pwr_eff, Pwr ];
            mission =...
                [0.2  0.4  0.1   0.1   0.1   0.1;     %Launch/re-entry (1)
                 0.5  0.3  0.05  0.05  0.05  0.05;    %Multimedia      (2)
                 0.05 0.05 0.05  0.05  0.3   0.5;     %Power saving    (3)
                 1/6  1/6  1/6   1/6   1/6   1/6;     %Normal          (4)
                 0.05 0.05 0.4   0.4   0.05  0.05;    %Cooperation     (5)
                 0.1  0.8  0.025 0.025 0.025 0.025];  %Emergency       (6)        
            
            w=mission(mission_num,:); %Select communication mission.
                        
            n=2; %Initial exploration probability divider
            
            %NO Q-table anymore! :)
            
            %%  MAIN SIMULATION ITERATION
            
            
            episilon_reset_lim=4e-3; %reset episilon when episilon < episilon_reset_lim
            
            %Log observables (for performance analysis only)
            log_f_max_T=zeros(1,episode_dur);
            log_f_min_BER=zeros(1,episode_dur);
            log_f_min_P=zeros(1,episode_dur);
            log_f_const_W=zeros(1,episode_dur);
            
            %Normalizing all NN exploration inputs (actions)
            input_explore_2 = mapminmax('apply',U',ps); %Applying normalization function ps to new training input
            
            %Percentage to exclude from training buffer for next retraining
            nn_delete=1;%1/4; %after training, delete 50% of oldest actions and retrain only after these percentage of training data is replaced by new training data
            
            history_size=200; % <<<<< NN training window with UNIQUE actions (and its respective performance) >>> designer parameter ANALYZE IMPACT ON PERFORMANCE!!!!!
%             history_size = 75;
%             history_size = 50;
%             history_size = 100;
            % Percentage of training data used for training [for online learning 100% history data can be used] (remaining training data is used for parellel testing)
            nn_parallel_train=1; %(90% for training and 10% for parellel testing)
            
            hist_count=0;
            ind_mean=0;
            
            temp_hist_=zeros(history_size,10); %Sliding window of last action-parformance data [acdionID f_observed time_stamp]
            hist_=zeros(history_size,size(U,2)+1); %[action_parameters f_observed]
            
            action_pred=zeros(1,episode_dur);
            action_discrete=zeros(1,episode_dur);
            
            %NN1 threshold prediction values for virtual exploration
            perc_max_explore=.90; % threshold percentage of current achievable maximum predicted performance
            rejection_rate=.95; %rejection percentage (rate) of exploring actions which performance predictions fall below the performance_threshold value
            
            train_ind=[];% indices when NN training took place
            
            track_exploit_err=0;
            
            explore_control=0; %used in while loop during explore_mode=1 ONLY (starts as 0 always to be run once)
            
            first_explore=0;
            
            e_p=[]; %exploit performance (holds last exploitation performance value; if new/current exploitation action resuls in poorer performance than previous, roll-back last exploitation action
            last_e_a=[-1 -1 -1 -1 -1 -1];%NaN NaN NaN NaN NaN NaN];
            
            elapsed_time=0; %time counter [seconds]
            jji=0; %main discrete time index/packet counter
            frame_dur=[];
            
            ind3=0;
            
            %% Main processing loop basedd on time duration (tracked by elapsed time, computing packet duration)
            while elapsed_time<512%length(time_series(1).esno_viasat)%50%512
                jji=jji+1; %increments packet counter
                
                % AI functions
                % [For implementation include parameter change monitoring feature]
                % If [Rs_, Es_, log2(M), roll_off, M, (n/k)] (either max, min, or any value within its ranges) changes, do the following:
                %   -Updates matrix U in case adaptable parameters changed
                %   -Update reference parameters for scaling
                
                %Compute epsilon(jji) %e-greedy function of time
                %Decrease exploration rate/increase exploitation
                if jji>=history_size + 1 && episilon(jji-1)> episilon_reset_lim
                    episilon(jji)=1/n; % n is # os times system exploited config. After episilon(jji-1)> episilon_reset_lim, reset exploration probability.
                    n=n+1;
                else
                    episilon(jji)=1;
                    n=1;%reset epsilon
                end
                
                
                %-->> (Exploit or Explore) epsilon=Exploration probability <<--%
                if s0==0
                    e_prob=rand; %Randomly (uniformly) pick Explore or Exploit
                    
                    % [Explore]
                    if e_prob<=episilon(jji)
                        expl=0; %flag
                        
                        %-----NN1 Prediction--->Exploration
                        %Predictions are done every iteration ONLY after NN is trained and ONLY if exploring
                        if NN_train==1
                            %Inputs
                            input_explore=[input_explore_2; (ones(1,length(U)))*(((measured_SNR_lin/max_SNR_lin)-0.5)./0.5)]; %scaling to range [-1, 1]
                            
                            %Predictions
                            parfor i_NN=1:numNN
                                y_pred(i_NN,:)=NN{i_NN}(input_explore); %Explore NN prediction
                            end
                            f_predic=mean(y_pred); %Ensamble average prediction
                            
                            %Logic to decide on actions predicted by NN
                            perf_th=max(f_predic)*perc_max_explore; %current performance threshold
                            
                            %get indexes of f_predic >= perf_th
                            bigger_f_predic=find(f_predic>=perf_th);
                            %get indexes of f_predic < perf_th
                            smaller_f_predic=find(f_predic<perf_th);
                            
                            NN_train_2=0;
                            
                            if first_explore==1%right after training the action with max predicted performance is explored first (could guide Exploitation NN input)
                                ii=find(f_predic==max(f_predic));
                                ii=datasample(ii,1);
                                first_explore=0;
                            else
                                if rand>=rejection_rate
                                    % pick random action with performance < perf_th
                                    if not(isempty(smaller_f_predic))
                                        ii=datasample(smaller_f_predic,1);
                                    else
                                        ii=datasample(bigger_f_predic,1);
                                    end
                                else
                                    %pick random action with performance > perf_th
                                    if not(isempty(bigger_f_predic))
                                        ii=datasample(bigger_f_predic,1);
                                    else
                                        ii=datasample(smaller_f_predic,1);
                                    end
                                end
                            end
                        end
                        track_exploit_err=0; %reset when explore
                        
                        
                        % [Exploit]
                    else
                        expl=1; %flag
                        
                        %-----NN2 Prediction--->Exploitation
                        %Predictions are done every iteration ONLY after NN is trained and ONLY if exploring
                        if NN_train_exploit==1 %Use NN predictions if NN has been already trained
                            
                            %Inputs
                            input_norm=[input_norm2 (measured_SNR_lin/max_SNR_lin)]';
                            input_norm=(input_norm-0.5)./0.5; %scaling to range [-1, 1]
                            
                            %Predictions
                            for n_i=1:numNN_exploit
                                parfor n_j=1:6
                                    norm_action_pred(n_i,n_j)=NN_exploit{n_j,n_i}(input_norm); %Exploit NN prediction
                                end
                            end
                            norm_action=mean(norm_action_pred);%Ensamble average prediction
                            
                            %Classify modcod
                            [~,modcod_class]=min(abs(sum_(1,1:5:16)-(norm_action(3)+norm_action(5))));
                            
                            %Denormalize predicted Action
                            for s=1:size(U,2)
                                action_pred(s,jji)=(norm_action(s)*(max(U(:,s))-min(U(:,s))))+min(U(:,s)); %only for simulation analysis
                                
                                %Classify denormalized values into executable action parameters [Action structure [Rs_, Es_, log2(M), roll_off, M, (n/k)] for NN training]
                                %Switcting between vectors (edges)
                                if s==1
                                    edges=Rs_;
                                elseif s==2
                                    edges=Es_;
                                elseif s==3
                                    edges=2:5; %log2(M)
                                elseif s==4
                                    edges=roll_off;
                                elseif s==5
                                    edges=M_;
                                elseif s==6 %classify encoding rate based on the modulation order classified previously
                                    if (action_discrete(5,jji))==4
                                        edges=[1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6 8/9 9/10];
                                    elseif (action_discrete(5,jji))==8
                                        edges=[3/5 2/3 3/4 5/6 8/9 9/10];
                                    elseif (action_discrete(5,jji))==16
                                        edges=[2/3 3/4 4/5 5/6 8/9 9/10];
                                    elseif (action_discrete(5,jji))==32
                                        edges=[3/4 4/5 5/6 8/9 9/10];
                                    end
                                end
                                
                                if s==3 || s==5
                                    min_ind=modcod_class;
                                else
                                    [~,min_ind]=min(abs(action_pred(s,jji)-edges)); %clasification using minimum distance
                                end
                                action_discrete(s,jji)=edges(min_ind); %Discretized action values (just to comply with action table based on hardware capabilities)
                            end
                            
                            [~,ii]=ismember(action_discrete(:,jji)', U, 'rows'); %finds the action ID of normalized predicted action within U
                            
                            
                        else %If NN failed to be trained
                            % Exploit actions from history in descending order of performance
                        end
                    end
                    
                    %only applies for s0=1 mode (while NN have not been trained yet)
                else
                    expl=0; %flag
                    %s0 gets reset after NN is trained
                    ii=ceil(rand*length(U));%First time always explore randomly
                end
                
                action(jji)=ii; %Chosen action id time-series; for analysis only
                
                %Parameters for chosen action --> %U structure [Rs_, P_, log2(M), roll_off, M, (n/k)] for NN training
                Rs=U(ii,1);
                Es_add=U(ii,2);
                k=U(ii,3);
                r_off=U(ii,4);
                M=U(ii,5);
                rate=U(ii,6);
                
                %Action time series
                act_modcod_map(jji)=modcod_map(ii);
                act_Rs(jji)=U(ii,1);
                act_Es_add(jji)=U(ii,2);
                act_k(jji)=U(ii,3);
                act_r_off(jji)=r_off;
                act_M(jji)=U(ii,5);
                act_rate(jji)=U(ii,6);
                
                
                frame_dur(jji)=frame_size_/Rs; %Frame duration in secs
                elapsed_time=sum(frame_dur);
                
                if ceil(elapsed_time/(512/5120))> length(TOTAL2) %for analysis only (stops script if there is no more data on synthetic snr time-series)
                    break
                end
                
                
                %% Measured at Tx (sent to Rx as telemetry data):
                
                %BEFORE BEING AFFECTED BY CHANNEL DYNAMICS
                
                %Transmitted power
                measured_Es(jji)=Es_add;
                %Amplifier - Increase Es (energy per symbol)
                measured_SNR(jji)=TOTAL2(ceil(elapsed_time/(512/5120)));
                measured_SNR_lin=10^(measured_SNR(jji)/10);
                measured_SNR_lin_norm(jji)=measured_SNR_lin/max_SNR_lin;
                EsNo=measured_SNR(jji)+Es_add;
                
                %Consumed additional power [dB]
                measured_P_consu(jji)=10*log10((10^(Es_add/10))*Rs);
                measured_P_consu_lin=10^(measured_P_consu(jji)/10);
                
                %Power efficiency Rb/P_consu  [bits/sec/Watts]
                measured_Pwr_eff(jji)=(k*rate)/((10^(Es_add/10))*Rs);
                
                %Bandwidth
                measured_W(jji)=Rs*(1+r_off);% [Hz]
                
                %Throughput
                measured_T(jji)=Rs*k*rate;% [bits/sec]
                
                
                
                %% AI at Rx:
                %Measured at Rx (AFTER BEING AFFECTED BY CHANNEL DYNAMICS):
                
                %  This study uses SNR profiles. A real-world experiment is required to evaluate its performance while using real SNR measurements.
                
                modcod_n=modcod_map(ii); %retrieves modcod ID that maps into BER_curve function
                eval(sprintf('ber_func = modcod_%d;', modcod_n)) %retrieves the proper BER_curve function
                if ber_func(EsNo)<0
                    measured_BER_est(jji)=1e-12; %assigns very low value (non-zero)
                elseif ber_func(EsNo)==0
                    measured_BER_est(jji)=1e-12; %assigns very low value (non-zero)
                elseif ber_func(EsNo)>1
                    measured_BER_est(jji)=1; %holds value at max = 1
                else
                    measured_BER_est(jji)=ber_func(EsNo); %assigns BER value as predicted by function
                end
                
                %Spectral efficiency
                measured_spec_eff(jji)=k*rate/(1+r_off);
                
                %(The following are sent out by the Tx as telemetry data:)
                %     -Transmitted power computed at Tx + additional power used above/below link budget estimated for clear sky operations
                %     -Bandwidth computed at Tx and not affected by channel (Assumed no presence of interferes).
                %     -Roll-off factor
                
                %Multi-objective fitness function (f_observed reference parameters)
                
                % Throughput
                f_max_T=measured_T(jji)/T_max;
                
                %BER
                f_min_BER_true=(100-(-1*(log10(BER_min/measured_BER_est(jji))))*(-100/(log10(BER_min))))/100; %Function value range from 1 to BER_min scaled to 0 to 1.
                if f_min_BER_true>=1
                    f_min_BER=1;
                else
                    f_min_BER=f_min_BER_true;
                end
                
                % Additional power
                f_min_P=Es_min_lin/(10^(Es_add/10));
                
                %Bandwidth
                if measured_W(jji)<=BW %Bandwidth
                    f_const_W=1; %No penalty if W is smaller than target BW (cannot cause interference)
                else
                    f_const_W=1-((measured_W(jji)-BW)/BW); %If W is more than double the target BW
                    if f_const_W<0
                        f_const_W=0;
                    end
                end
                
                %Spectral efficiency
                spec_eff_max=log2((mod_list(end)))*(cod_list(end))/(1+min(roll_off));
                f_spec_eff=measured_spec_eff(jji)/spec_eff_max;
                
                
                %Observed state: fitness function
                f_observed2(jji,:) = [((measured_T(jji)-T_min)./(T_max-T_min)) ((-10.*log10(measured_BER_est(jji))-ber_dB_min)./(ber_dB_max-ber_dB_min)) ((measured_W(jji)-BW_min)./(BW_max-BW_min)) ((measured_spec_eff(jji)-spect_eff_min)./(spect_eff_max-spect_eff_min)) ((log10(measured_Pwr_eff(jji))-log10(pwr_eff_min))./(log10(pwr_eff_max)-log10(pwr_eff_min))) 1-((measured_P_consu_lin-P_consu_min_lin)./(P_consu_max_lin-P_consu_min_lin)) (measured_SNR_lin./max_SNR_lin)]; %range [0,1]
                f_observed(jji) = f_observed2(jji,1:end-1)*w'; %range [0,1] (SNR is not part of optimization goal)
                
                %Adapt/updates NN2 input whenever exploration finds a better performance
                if  s0==1 && hist_count==0
                    e_p=f_observed(jji);
                elseif s0==1 && hist_count>0
                    if f_observed(jji)>e_p
                        e_p=f_observed(jji);                        
                    end
                end
                
                if f_observed(jji)>max_f_observed
                    max_f_observed=f_observed(jji); %global best known performance so far
                    if expl==0
                        input_norm2=f_observed2(jji,1:end-1);
                    end
                else
                    if expl==1 % [Exploit]
                        %fprintf('%f,%f \n',f_observed(jji),e_p);
                        if f_observed(jji)<e_p
                            if e_p-f_observed(jji)>0.1 && (sum(input_norm2==last_e_a)==length(input_norm2)) %RESET "More efficient Recover Mode". Threshold value is a designer parameter (0.5 for specific missions, 0.1 for general)
                                fprintf('f1');
                                s0=1; %enters exploration mode
                                %reset NN history
                                hist_count=0;
                                temp_hist_=zeros(history_size,10);
                                jji_reset(jji)=jji;
                                max_f_observed=0;                                
                            elseif f_observed(jji)<e_p*0.9 %Quick "Recover Mode" using performances from the buffer. Triggers when 90% below previous exploration level
                                fprintf('f2');
                                hist2=sortrows(temp_hist_,2);   
                                ind3=ind3+1;
                                if ind3==history_size
                                    ind3=1;
                                end
                                nn2=hist2(end-ind3,4:9);
                                input_norm2=[(nn2(:,1)-T_min)./(T_max-T_min) (-10.*log10(nn2(:,2))-ber_dB_min)./(ber_dB_max-ber_dB_min) (nn2(:,3)-BW_min)./(BW_max-BW_min) (nn2(:,4)-spect_eff_min)./(spect_eff_max-spect_eff_min) (log10(nn2(:,5))-log10(pwr_eff_min))./(log10(pwr_eff_max)-log10(pwr_eff_min)) 1-((nn2(:,6)-P_consu_min_lin)./(P_consu_max_lin-P_consu_min_lin))];
                            elseif f_observed(jji)>e_p*0.9 && ind3>0 %Accepts new exploitation performance 90% above last exploitation threshold 
                                fprintf('f3 time:%f,f_observed:%f e_p:%f \n',elapsed_time,f_observed(jji),e_p);
                                e_p=f_observed(jji);
                                last_e_a=input_norm2;
                            else
                                fprintf('f4');
                                input_norm2=last_e_a; %if exploiting and current exploitation performance is worse than previous exploitation performance; roll-back NN2 input
                            end
                        else        
                            fprintf('f5 time:%f,f_observed:%f e_p:%f \n',elapsed_time,f_observed(jji),e_p);
                            e_p=f_observed(jji); %tracks last exploitation performance
                            last_e_a=input_norm2; %tracks last exploitation NN2 input                            
                        end
                    end
                end
                
                %Logging function measurebles; for analysis only
                log_f_max_T(jji)=f_max_T;
                log_f_min_BER(jji)=f_min_BER;
                log_f_min_P(jji)=f_min_P;
                log_f_const_W(jji)=f_const_W;
                
                
                %-->> History sliding window (shared among Explore and Exploit NN's) <<--
                if isempty(find(temp_hist_(:,1)==ii)) %chosen action not present in sliding window
                    hist_count=hist_count+1; %populate
                    fprintf('%d ',hist_count);%%%
                    if hist_count<=history_size %if sliding window not full yet
                        ind_update=(temp_hist_(:,3)>=1);%index for update
                        temp_hist_(hist_count,:)= [ii f_observed(jji) 1 measured_T(jji) measured_BER_est(jji) measured_W(jji) measured_spec_eff(jji) measured_Pwr_eff(jji) measured_P_consu_lin measured_SNR_lin]; % [actions f_observed time_stamp]
                        temp_hist_(ind_update,3)=temp_hist_(ind_update,3)+1; %update all action IDs
                    else %if sliding window already full >>> replace
                        [~,jj]=max(temp_hist_(:,3)); %find action with oldest/highest time_stamp
                        temp_hist_(jj,:)= [ii f_observed(jji) 0 measured_T(jji) measured_BER_est(jji) measured_W(jji) measured_spec_eff(jji) measured_Pwr_eff(jji) measured_P_consu_lin measured_SNR_lin]; %replace it
                        temp_hist_(:,3)=temp_hist_(:,3)+1; %update all action IDs
                    end
                else %chosen action is already on sliding window, update it with most recent performance
                    ind_mean=find(temp_hist_(:,1)==ii);
                    ind_update=(temp_hist_(:,3)>0);%index for update
                    temp_hist_(ind_mean,:)= [ii f_observed(jji) 1 measured_T(jji) measured_BER_est(jji) measured_W(jji) measured_spec_eff(jji) measured_Pwr_eff(jji) measured_P_consu_lin measured_SNR_lin];
                    temp_hist_(ind_update,3)=temp_hist_(ind_update,3)+1; %update all action IDs
                end
                
                % ----> Train NN <---
                if hist_count==history_size % enables training after sliding window is full
                    train_ind=[train_ind jji]; % indices when NN training took place
                    
                    %-------------Examples NN_explore-------------
                    % Populate with acitons (only after training window was built)
                    hist_(:,1:size(U,2))=U((temp_hist_(:,1)),:);
                    % Populate f_observed for non-repeating actions
                    hist_(:,end)=temp_hist_(:,2);
                    %Applying normalization function ps to new training input
                    pnewn = mapminmax('apply',hist_(:,1:size(U,2))',ps);
                    
                    %NN1 training inputs
                    examples_input= [pnewn'];
                    examples_input=[examples_input (((temp_hist_(:,end)/max_SNR_lin)-0.5)./0.5)]; %scaled to range [-1,1]
                    
                    %NN1 training outputs
                    examples_target= [hist_(:,end)]; %range [0, 1]
                    
                    %Spliting inputs/outputs dataset (not needed for online operations)
                    %Splits training dataset into 90% training and 10% parallel testing
                    Q1 = floor(history_size*nn_parallel_train);
                    Q2 = history_size-Q1; %not needed for online operations
                    ind = randperm(history_size);
                    ind1 = ind(1:Q1);
                    ind2 = ind(Q1+(1:Q2)); %not needed for online operations
                    %Training
                    x1 = examples_input(ind1,:)';
                    y1 = examples_target(ind1,:)';
                    %Parallel testing (%Guarantees all nets use same test dataset)
                    x2 = examples_input(ind2,:)'; %not needed for online operations
                    y2 = examples_target(ind2,:)';%not needed for online operations
                    
                    %%%
                    %Training NN1
                    tic
                    parfor n_i=1:numNN
                        [NN{n_i},trainRecord{n_i}]=train(net,x1,y1);
                    end
                    elapsedTime = toc;
                    
                    y1_results = NN{1}(x1);
                    err1 = perform(NN{1},y1,y1_results);
                    fprintf('Time to train explore network is %f, error = %f or %f \n', elapsedTime,err1,trainRecord{1}.best_tperf);%trainRecord{1}.best_tperf);
                    %flags0e
                    NN_train=1;
                    NN_train_2=1;
                    first_explore=1;
                    
                    
                    %-------------Examples NN_exploit-------------
                    %NN1 training inputs
                    examples_in_exploit = [(temp_hist_(:,4)-T_min)./(T_max-T_min) (-10.*log10(temp_hist_(:,5))-ber_dB_min)./(ber_dB_max-ber_dB_min) (temp_hist_(:,6)-BW_min)./(BW_max-BW_min) (temp_hist_(:,7)-spect_eff_min)./(spect_eff_max-spect_eff_min) (log10(temp_hist_(:,8))-log10(pwr_eff_min))./(log10(pwr_eff_max)-log10(pwr_eff_min)) 1-((temp_hist_(:,9)-P_consu_min_lin)./(P_consu_max_lin-P_consu_min_lin)) temp_hist_(:,10)./max_SNR_lin]; %range [0,1]                                       
                    examples_in_exploit=(examples_in_exploit-0.5)./0.5; %scaled to range [-1, 1]
                    
                    %NN1 training outputs
                    examples_out_exploit = [U_out(temp_hist_(:,1),:)]; %range [0, 1]
                    
                    %Spliting inputs/outputs dataset (not needed for online operations)
                    %Splits training dataset into 90% training and 10% parallel testing
                    Q1 = floor(history_size*nn_parallel_train);
                    Q2 = history_size-Q1; %not needed for online operations
                    ind = randperm(history_size);
                    ind1 = ind(1:Q1);
                    ind2 = ind(Q1+(1:Q2)); %not needed for online operations
                    %Training
                    x1_exploit = examples_in_exploit(ind1,:)'; %not needed for online operations
                    y1_exploit = examples_out_exploit(ind1,:)'; %not needed for online operations
                    %%%
                    %Training NN2
                    parfor n_i=1:numNN_exploit
                        for n_j=1:6
                            NN_exploit{n_j,n_i}=train(net_exploit,x1_exploit,y1_exploit(n_j,:)); %training per feeature NN
                        end
                    end
                    NN_train_exploit=1;
                    
                    %-------------Reset sliding window------------
                    %Delete x percentage of oldest actions
                    temp_hist_=sortrows(temp_hist_,3);
                    temp_hist_((history_size-(nn_delete*history_size))+1:end,:)=0;
                    hist_count=(history_size-(nn_delete*history_size)); %resets hist_count
                    ind3=1;
                    
                    if s0==1
                        s0=0;%reset s0; allows NN's to be used
                    end
                end
                
            end %End of Main iteration (while)
        end %End of multiple iterations for different channels
    end %End of mission loop
    f_observed_save{iterations} = f_observed;
    f_observed2_save{iterations} = f_observed2;
    %eval(sprintf('save(''sim_%d_m4_resetMODE9.mat'')', iterations)) 
    %eval(sprintf('save(''sim_%d_m4_resetMODE9.mat'')', iterations)) 
end