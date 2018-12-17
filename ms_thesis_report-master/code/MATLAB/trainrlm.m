%% trainrlm:
% Takes a MATLAB network class and a struct containing the different 
% Parameters required to do iteration.
function [net,recurseOut,err_out] = trainrlm(net_old,recurseIn,X,Y,Xtest,Ytest)
    
    % X and Y can either be a single input-output combo,
    % or a batch of in-out combos. 

    
    % Make sample-by-sample, batch by batch mode.
    repeat = 1;
    % Pass sample through NN.
    results_NN = net_old(Xtest);
    % Get Error. 
    err = perform(net_old,Ytest,results_NN);%results_NN - Y;
    old_wb = getwb(net_old);
        while repeat == 1

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
            results2_NN = net_tmp(Xtest);
            err_out = perform(net_tmp,Ytest,results2_NN);
            
            if err_out - err >= 1e-13
              % If the update increases error, don't actually update.
                recurseOut.rho = recurseIn.rho * 5;%10;
                if recurseOut.rho > 1e11
                    recurseOut.rho = 1e11;
                end
                    net = net_old;
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
end

