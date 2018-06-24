plots = [2,15,6,9];
% excellent
for i=1:length(plots)
    figure
    a=plots(i);
    plot(all_workspaces(a).ActionRecdTimeSec-all_workspaces(a).INIT_TIME_SEC, ... 
        all_workspaces(a).MeasRecvd(:,1),'b', ... 
        all_workspaces(a).ActionRecdTimeSec(all_workspaces(a).RecvLock==0)-all_workspaces(a).INIT_TIME_SEC, ...
        all_workspaces(a).MeasRecvd(all_workspaces(a).RecvLock==0,1),'*r',...
        'MarkerSize',1)
    ylabel('E_s/N_0 Profile (dB)')
    xlabel('Time (s)')
    legend('E_s/N_0 Profile','Unreliable Measurement')
    ylim([-5 25])
    set(gcf,'Position',[2000 2000 1120 210])
    ylim([-4 20])
    grid on
    cropplot(gcf,gca)
    
    if i==1
        print('excellent_esno_profile.pdf','-dpdf','-opengl')
    elseif i==2
        print('great_esno_profile.pdf','-dpdf','-opengl')
    elseif i==3
        print('good_esno_profile.pdf','-dpdf','-opengl')
    elseif i==4
        print('poor_esno_profile.pdf','-dpdf','-opengl')
    end
end