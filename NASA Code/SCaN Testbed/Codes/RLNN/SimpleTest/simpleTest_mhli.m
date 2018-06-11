    close all;
    [x,t] = crab_dataset;
    
    net = simpleTest_setup();

% NN output functions (help nntransfer)

    % NN1 Layers
    net.layers{1}.size = 13;%7;
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.size = 50;
    net.layers{2}.transferFcn = 'logsig';
    net.layers{3}.size = 2;
    net.layers{3}.transferFcn = 'purelin';

    %Early stop conditions
    net.trainParam.max_fail=20;
    net.trainParam.min_grad=1e-12;
    
    [net,tr] = train(net,x,t);
    figure();
    plotperform(tr);
    
    
    
    testX = x(:,tr.testInd);
    testT = t(:,tr.testInd);

    testY = net(testX);
    testIndices = vec2ind(testY);
    figure();
    plotconfusion(testT,testY);