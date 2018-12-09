%% trainrlm:
% Takes a MATLAB network class and a struct containing the different 
% Parameters required to do iteration.
function [net,recurseOut,err_out] = trainrlm(net_old,recurseIn,X,Y,Xtest,Ytest)
    
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
    results_NN = net_old(Xtest);
    % Get Error. 
    err = perform(net_old,Ytest,results_NN);%results_NN - Y;
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
                grad = defaultderiv('dperf_dwb',net_tmp, X(:,i),Y(:,i));%defaultderiv('dperf_dwb',net_tmp, X(:,i),Y(:,i));
                [new_S, new_P,new_wb] =  rlm_update(grad,rho_tmp, new_P,new_wb,...
                                                    time_tmp, recurseIn.alpha,err);
                net_tmp = setwb(net_old, new_wb);
                time_tmp = time_tmp + 1;
                
            end
            
%             grad = defaultderiv('dperf_dwb',net_tmp, X(:,i),Y(:,i));%defaultderiv('dperf_dwb',net_tmp, X(:,i),Y(:,i));
%             [new_S, new_P,new_wb] =  rlm_update(grad,rho_tmp, new_P,new_wb,...
%                                                     time_tmp, recurseIn.alpha,err);
%             net_tmp = setwb(net_old, new_wb);
%             time_tmp = time_tmp + 1;
            
%             net_tmp = setwb(net_old,new_wb);
            %%%%
            
            results2_NN = net_tmp(Xtest);
            err_out = perform(net_tmp,Ytest,results2_NN);
            
            if err_out - err >= 1e-13
               % net = setwb(net_old,old_wb); % If the update increases error, don't actually update.
                recurseOut.rho = recurseIn.rho * 5;%10;
                if recurseOut.rho > 1e11
                    recurseOut.rho = 1e11;
                end
                    net = net_old;
%                     recurseIn.S = new_S;
%                     recurseIn.P = new_P;
                    repeat = 0;
            else%if err_out < 0.01
                net = net_tmp;
                recurseOut.S = new_S;
                recurseOut.P = new_P;
                recurseOut.rho = recurseIn.rho /5;%/ 10;
                if recurseOut.rho < 1e-30
                    recurseOut.rho = 1e-30;
                end
                repeat = 0;
            end
        end
    recurseOut.Time = time_tmp; 
    new_rho = recurseOut.rho;
    recurseOut = recurseIn.updateMatrix(grad,time_tmp,new_P,new_S,new_rho,recurseIn.alpha);
%     fprintf('recurseIn.Time = %f \n', recurseIn.Time);
  %  end

%     recurseOut.Time = recurseOut.Time + 1;
    
    %net = net_old;
% [NET,TR] = trainlm(NET,X,T)
end

