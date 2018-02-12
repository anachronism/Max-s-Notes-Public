% A bit redundant, but makes it standalone.
function norm = normImages(im)
    norm = cell(length(im),1);
    for i = 1:length(im)
        tmp = reshape(im{i}, [1,2500]);
        norm{i} = (tmp - mean(tmp))/std(tmp);
        norm{i} = reshape(norm{i},[50,50]);
    end
end