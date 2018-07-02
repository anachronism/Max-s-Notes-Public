clear all
close all
toplvl = dir;

all_workspaces = [];
for i=1:length(toplvl)
    if(~toplvl(i).isdir || ...
       strcmp(toplvl(i).name,'.') || ...
       strcmp(toplvl(i).name,'..') || ...
       strcmp(toplvl(i).name,'rlnn4'))
        continue
    else
        cd(toplvl(i).name);
        sublvl = dir;
        for j=1:length(sublvl)
            if(~sublvl(j).isdir || ...
                strcmp(sublvl(j).name,'.') || ...
                strcmp(sublvl(j).name,'..') || ...
                strcmp(sublvl(j).name,'rlnn4'))
                continue
            else
                cd(sublvl(j).name)
                
                S = load('parsedLogsWorkspace.mat','FitObs','NewtecLck');
                all_workspaces=[all_workspaces S] ;
                cd('..')
            end
        end
        cd('..')
    end
end
save('all_workspaces_fitobserved.mat')
