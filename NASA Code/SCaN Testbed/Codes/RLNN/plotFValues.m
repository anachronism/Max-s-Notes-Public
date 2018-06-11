figure()
for i = 1:5
    subplot(1,5,i)
    plot(f_observed_save{i});
    ylim([0 1]);
end