function [S_new,P_new,w_new] = rlm_update(grad,p, P_old,w_old,t, alpha,err)
    % 0.97 < alpha < 1.
    % Error is received results compared to forward Prop'd results. Can be
    % done on a sample by sample basis, or on blocks.
    % Gradient would probably be current error minus last error.
    
    % Calculate Omega
    nGrad = length(grad.');
    omegaRow = zeros(nGrad,1);
%     if t >= length(grad.')
%         omegaRow(mod(t,length(grad.'))+1) = 1;
%     end
    omegaRow(mod(t,length(grad.'))+1) = 1;

    Omega = [grad,omegaRow];
    
    % Calculate Lambda
    invLambda = [1,0;0,p];
    Lambda = pinv(invLambda);
    
    % Calculate new weights and intermediate matrices.
    S_new = alpha * Lambda + Omega.'*P_old*Omega;
    P_new = 1/alpha * (P_old - P_old*Omega*pinv(S_new)*Omega.'*P_old);
%     P_new = P_new ./ max(P_new(:));
    
    w_new = w_old + P_new*grad*err;
end