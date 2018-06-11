function net = simpleTest_setup()

    net=network(1,3,[0;0;0], [1 ; 0 ; 0], [ 0 0 0; 1 0 0; 0 1 0], [0 0 1]);
    %NN input size
    net.inputs{1}.size=6;%1;
    %NN input range values
    %net.inputs{1}.range = [-1 1; -1 1; -1 1; -1 1; -1 1; -1 1; -1 1];
    net.inputs{1}.range = [-1 1; -1 1; -1 1; -1 1; -1 1; -1 1; ];
    %NN train function
    net.trainFcn = 'trainlm';
     %NN dataset division function (training, validation, test)
    net.divideFcn='dividerand'; % 70%,15%,15% default
    net.performFcn = 'mse';
    % net.performFcn = 'crossentropy';    

end