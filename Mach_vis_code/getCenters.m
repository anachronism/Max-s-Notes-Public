function meanArray = getCenters(data,labels,dim)
    %Split up data into labels.
    x = repmat((labels == 1).',9,1);
    data_f1 = data .* x;
    data_f1( :, ~any(data_f1,1) ) = [];  %remove zero cols
    mean_f1 = mean(data_f1.').';
    
    
    x = repmat((labels == 2).',9,1);
    data_f2 = data.*x;
    data_f2( :, ~any(data_f2,1) ) = [];  %remove zero cols
    mean_f2 = mean(data_f2.').';
    
    x = repmat((labels == 3).',9,1);
    data_f3 = data.*x;
    data_f3( :, ~any(data_f3,1) ) = [];  %remove zero cols
    mean_f3 = mean(data_f3.').';
    
    x = repmat((labels == 4).',9,1);
    data_f4 = data.*x;
    data_f4( :, ~any(data_f4,1) ) = [];  %remove zero cols
    mean_f4 = mean(data_f4.').';
    
    x = repmat((labels == 5).',9,1);
    data_f5 = data.*x;
    data_f5( :, ~any(data_f5,1) ) = [];  %remove zero cols
    mean_f5 = mean(data_f5.').';
    
    x = repmat((labels == 6).',9,1);
    data_f6 = data.*x;
    data_f6( :, ~any(data_f6,1) ) = [];  %remove zero cols
    mean_f6 = mean(data_f6.').';
    
    x = repmat((labels == 7).',9,1);
    data_f7 = data.*x;
    data_f7( :, ~any(data_f7,1) ) = [];  %remove zero cols
    mean_f7 = mean(data_f7.').';
    
    x = repmat((labels == 8).',9,1);
    data_f8 = data.*x;
    data_f8( :, ~any(data_f8,1) ) = [];  %remove zero cols
    mean_f8 = mean(data_f8.').';
    
    x = repmat((labels == 9).',9,1);
    data_f9 = data.*x;
    data_f9( :, ~any(data_f9,1) ) = [];  %remove zero cols
    mean_f9 = mean(data_f9.').';
    
    x = repmat((labels == 10).',9,1);
    data_f10 = data.*x;
    data_f10( :, ~any(data_f10,1) ) = [];  %remove zero cols
    mean_f10 = mean(data_f10.').';

    meanArray = [mean_f1,mean_f2,mean_f3,mean_f4,mean_f5 ...
                mean_f6,mean_f7,mean_f8,mean_f9,mean_f10];
end