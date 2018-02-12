function out = cell2feat(cellArray)
    n = length(cellArray);
    out = zeros(n, 2500);
    
    for i = 1:n
        out(i,:) = reshape(cellArray{i},[1,2500]);
    end
end