%% trainrlm:
% Takes a MATLAB network class and a struct containing the different 
% Parameters required to do iteration.
function [net,recurseOut,err_out] = trainrlm(net_old,recurseIn,X,Y)
    
    %net = struct(net_old);
    
    % X and Y can either be a single input-output combo,
    % or a batch of in-out combos. 

    
    % Make sample-by-sample, batch by batch mode.
%     sizeBatch = length(X);
%     if sizeBatch ~= recurseIn.sizeBatch && recurseIn.sizeBatch ~= -1
%         error('Size of input does not match the size of inputs previously passed');
%     end
    repeat = 1;
    % Pass sample through NN.
    results_NN = net_old(X);
    % Get Error. 
    err = perform(net_old,Y,results_NN);%results_NN - Y;
    %%% STILL TO DO % Get gradient along x.
    
    %%% TODO: Figure out how to input the weights to the update.
    % Old weights will be a combination of net_old.iw and NN.lw
    
    old_wb = getwb(net_old);
    
    %grad = defaultderiv('dperf_dwb',net_old,X,Y);
%         grad = defaultderiv('dperf_dwb',net_old,X,Y);
        while repeat == 1
            %     p = 5;%0.1;%l*length(old_wb);%5;%%% CHANGE
            % p = 0.1;
            %%%
            % Call rlm_update / actually do it.
            
            %%%%
            net_tmp = setwb(net_old,old_wb);
            time_tmp = recurseIn.Time;
            rho_tmp = recurseIn.rho;
            new_P = recurseIn.P;
            new_S = recurseIn.S;
            new_wb = old_wb;
            for i = 1:size(X,2)
                grad = defaultderiv('dperf_dwb',net_tmp, X(:,i),Y(:,i));
                [new_S, new_P,new_wb] =  rlm_update(grad,rho_tmp, new_P,new_wb,...
                                                    time_tmp, recurseIn.alpha,err);
                net_tmp = setwb(net_old, new_wb);
                time_tmp = time_tmp + 1;
                
            end
            
%             net_tmp = setwb(net_old,new_wb);
            %%%%
            
            results2_NN = net_tmp(X);
            err_out = perform(net_tmp,Y,results2_NN);
            
            if err_out - err >= 1e-13
               % net = setwb(net_old,old_wb); % If the update increases error, don't actually update.
                recurseIn.rho = recurseIn.rho * 5;%10;
                if recurseIn.rho > 1e11
                    recurseIn.rho = 1e11;
                    net = net_old;
%                     recurseIn.S = new_S;
%                     recurseIn.P = new_P;
                    repeat = 0;
                end
            else%if err_out < 0.01
                net = net_tmp;
                recurseIn.S = new_S;
                recurseIn.P = new_P;
                recurseIn.rho = recurseIn.rho /5;%/ 10;
                if recurseIn.rho < 1e-30
                    recurseIn.rho = 1e-30;
                end
                repeat = 0;
            end
        end
    recurseIn.Time = time_tmp;    
%     fprintf('recurseIn.Time = %f \n', recurseIn.Time);
  %  end
    
    recurseOut = recurseIn;
%     recurseOut.Time = recurseOut.Time + 1;
    
    %net = net_old;
% [NET,TR] = trainlm(NET,X,T)
end




% % % %% trainrlm:
% % % % Takes a MATLAB network class and a struct containing the different 
% % % % Parameters required to do iteration.
% % % function [net,recurseOut,err_out] = trainrlm(net_old,recurseIn,X,Y)
% % %     
% % %     %net = struct(net_old);
% % %     
% % %     % X and Y can either be a single input-output combo,
% % %     % or a batch of in-out combos. 
% % % 
% % %     
% % %     % Make sample-by-sample, batch by batch mode.
% % % %     sizeBatch = length(X);
% % % %     if sizeBatch ~= recurseIn.sizeBatch && recurseIn.sizeBatch ~= -1
% % % %         error('Size of input does not match the size of inputs previously passed');
% % % %     end
% % %     repeat = 1;
% % %     % Pass sample through NN.
% % %     results_NN = net_old(X);
% % %     % Get Error. 
% % %     err = perform(net_old,Y,results_NN);%results_NN - Y;
% % %     %%% STILL TO DO % Get gradient along x.
% % %     
% % %     %%% TODO: Figure out how to input the weights to the update.
% % %     % Old weights will be a combination of net_old.iw and NN.lw
% % %     
% % %     old_wb = getwb(net_old);
% % %     
% % %     %grad = defaultderiv('dperf_dwb',net_old,X,Y);
% % % %         grad = defaultderiv('dperf_dwb',net_old,X,Y);
% % %         while repeat == 1
% % %             %     p = 5;%0.1;%l*length(old_wb);%5;%%% CHANGE
% % %             % p = 0.1;
% % %             %%%
% % %             % Call rlm_update / actually do it.
% % %             
% % %             %%%%
% % %             net_tmp = setwb(net_old,old_wb);
% % %             time_tmp = recurseIn.Time;
% % %             rho_tmp = recurseIn.rho;
% % %             new_P = recurseIn.P;
% % %             new_S = recurseIn.S;
% % %             new_wb = old_wb;
% % %             for i = 1:size(X,2)
% % %                 grad = defaultderiv('dperf_dwb',net_tmp, X(:,i),Y(:,i));
% % %                 [tmp_S, tmp_P,new_wb] =  rlm_update(grad,rho_tmp, new_P,new_wb,...
% % %                                                     time_tmp, recurseIn.alpha,err);
% % %                 net_last = net_tmp;
% % %                 net_tmp = setwb(net_old, new_wb);
% % %                 results2_NN = net_tmp(X(:,i));
% % %                 err_out = perform(net_tmp,Y(:,i),results2_NN);
% % %                 if err_out - err >= 1e-13
% % %                    % net = setwb(net_old,old_wb); % If the update increases error, don't actually update.
% % %                    rho_tmp = rho_tmp * 5;%10;
% % %                     if rho_tmp > 1e11
% % %                         rho_tmp = 1e11;
% % %                         net_tmp = net_last;
% % %                         repeat = 0;
% % %                     end
% % %                 else%if err_out < 0.01
% % %                     
% % %                     new_S=tmp_S;
% % %                     new_P=tmp_P;
% % %                     rho_tmp = rho_tmp /5;%/ 10;
% % %                     if rho_tmp < 1e-30
% % %                         rho_tmp = 1e-30;
% % %                     end
% % %                     repeat = 0;
% % %                 end
% % %                 
% % %                 time_tmp = time_tmp + 1;
% % %             end
% % %             
% % % %             net_tmp = setwb(net_old,new_wb);
% % %             %%%%
% % %             
% % %             results2_NN = net_tmp(X);
% % %             err_out = perform(net_tmp,Y,results2_NN);
% % %             
% % % %             if err_out - err >= 1e-13
% % % %                % net = setwb(net_old,old_wb); % If the update increases error, don't actually update.
% % % %                 recurseIn.rho = recurseIn.rho * 5;%10;
% % % %                 if recurseIn.rho > 1e11
% % % %                     recurseIn.rho = 1e11;
% % % %                     net = net_old;
% % % % %                     recurseIn.S = new_S;
% % % % %                     recurseIn.P = new_P;
% % % %                     repeat = 0;
% % % %                 end
% % % %             else%if err_out < 0.01
% % % %                 net = net_tmp;
% % % %                 recurseIn.S = new_S;
% % % %                 recurseIn.P = new_P;
% % % %                 recurseIn.rho = recurseIn.rho /5;%/ 10;
% % % %                 if recurseIn.rho < 1e-30
% % % %                     recurseIn.rho = 1e-30;
% % % %                 end
% % % %                 repeat = 0;
% % % %             end
% % % %         end
% % %     recurseIn.Time = time_tmp;  
% % %     recurseIn.P = new_P;
% % %     recurseIn.S = new_S;
% % %     recurseIn.rho = rho_tmp;
% % %     net = net_tmp;
% % % %     fprintf('recurseIn.Time = %f \n', recurseIn.Time);
% % %   %  end
% % %     
% % %     recurseOut = recurseIn;
% % % %     recurseOut.Time = recurseOut.Time + 1;
% % %     
% % %     %net = net_old;
% % % % [NET,TR] = trainlm(NET,X,T)
% % % end