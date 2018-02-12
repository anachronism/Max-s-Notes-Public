% Return the index and distance value of the current results.
function [index,distance] = nnL2_bagOfWords(compare,dict)
    compare = compare.';
    for i = 1:size(dict,2)
        l2_norm(i) = sum((compare - dict(i,:)).^2);
    end
    
    [distance,index] = min(l2_norm);

end