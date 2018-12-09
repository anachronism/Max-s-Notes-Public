function  [netOut,f_measure,g_mean,recall,precision,...
    err] = learn_nse_update(net, data_train, labels_train, data_test, ...
  labels_test)
numClassifiers = 20;%
 % has the 
  if net.initialized == false, net.beta = []; end
  
  mt = size(data_train,2); % number of training examples
  Dt = ones(mt,1)/mt;         % initialize instance weight distribution
  Dt_sampBySamp = Dt;
  if net.initialized==1
    % STEP 1: Compute error of the existing ensemble on new data
    predictions = regress_ensemble(net, data_train, labels_train); %%% CONVERT TO REGRESSION.
    
    Et_sampBySamp = mse_reg(predictions.',labels_train)/mt;
    Et = sum(Et_sampBySamp); %%% THIS IS TO BE UPDATED TO FIT FOR REG.
    %%% get a beta for each net.
    Bt = Et/(1-Et);           % this is suggested in Metin's IEEE Paper
    Bt_sampBySamp = Et_sampBySamp./(1-Et_sampBySamp);
    if Bt==0, Bt = 1/mt; end; % clip 
    
    % update and normalize the instance weights
    Wt = 1/mt * Bt;
    Dt = Wt/sum(Wt);
    Wt_sampBySamp = 1/mt * Bt_sampBySamp;
    Dt_sampBySamp = Wt_sampBySamp/sum(Wt_sampBySamp);
%     Dt(predictions==labels_train) = Dt(predictions==labels_train) * Bt; 
%     Dt = Dt/sum(Dt);
  end
  
  % STEP 3: New classifier
  if size(net.classifiers,2) < numClassifiers
      net.classifiers{end + 1} = train(...
        net.base_classifier, ...
        data_train, ...
        labels_train);
  else %%% TODO: Proper pruning instead of oldest-out
      net.classifiers{mod(size(net.classifiers,2),20) + 1} = train(...
        net.base_classifier, ...
        data_train, ...
        labels_train);
  end
  
  % STEP 4: Evaluate all existing classifiers on new data
  t = size(net.classifiers,2);
  y = decision_ensemble(net, data_train, labels_train, t); %%% VERIFY IS WHAT"S WANTED
 %   y = regress_ensemble(net, data_train, labels_train);%, t);
  for k = 1:min(net.t,numClassifiers) %%% DOESNT WORK WITH CYCLICAL.
%     epsilon_tk  = sum(Dt.*mse_reg(y(:,k),labels_train)/mt^2); %%% CONVERT TO REGRESSION ERROR. CHECK TO SEE IF WEIGHT IS OK.
    epsilon_tk = sum(Dt_sampBySamp.*mse_reg(y(:,k),labels_train.')/mt);
    if (k<net.t)&&(epsilon_tk>0.5) 
      epsilon_tk = 0.5;
    elseif (k==net.t)&&(epsilon_tk>0.5)
      % try generate a new classifier 
      net.classifiers{k} = train(...
        net.base_classifier, ...  
        data_train, ...
        labels_train);
      epsilon_tk  = sum(Dt(y(:, k) ~= labels_train));
      epsilon_tk(epsilon_tk > 0.5) = 0.5;   % we tried; clip the loss 
    end
    net.beta(net.t,k) = epsilon_tk / (1-epsilon_tk);
  end
  
  % compute the classifier weights
  if net.t==1
    if net.beta(net.t,net.t)<net.threshold
      net.beta(net.t,net.t) = net.threshold;
    end
    net.w(net.t,net.t) = log(1/net.beta(net.t,net.t));
  else
    for k = 1:min(net.t,numClassifiers) %%% MAKE SURE THIS IS WHAT"S WANTED>
      b = t - k - net.b; %%% check
      if net.t <= numClassifiers
        omega = 1:(net.t - k + 1);
      else
          omega = 1:numClassifiers;
      end
      omega = 1./(1+exp(-net.a*(omega-b)));
      omega = (omega/sum(omega))';
      if net.t <= numClassifiers
        beta_hat = sum(omega.*(net.beta(k:net.t,k))); %%% FIX TO WORK WITH THE MODULO ASPECT.
      else
          net.beta(end-numClassifiers+1:end,k)
          beta_hat = sum(omega .* net.beta(end-numClassifiers+1:end,k));
      end
      if beta_hat < net.threshold
        beta_hat = net.threshold;
      end
      net.w(net.t,k) = log(1/beta_hat);
    end
  end
  
  % STEP 7: classifier voting weights
  net.classifierweights{end+1} = net.w(end,:);
  
%   predictions = decision_ensemble(net, data_test, labels_test,t);
%   [predictions,posterior] = classify_ensemble(net, data_test, labels_test);
  %errs(ell) = sum(predictions ~= labels_test_t)/numel(labels_test_t);
  
  f_measure = 0;
  g_mean = 0;
  recall = 0;
  precision = 0;
  err = 0;
%    [f_measure,g_mean,recall,precision,...
%     err] = stats(labels_test, predictions, net.mclass);
  
  net.initialized = 1;
  net.t = net.t + 1;
  netOut = net;
