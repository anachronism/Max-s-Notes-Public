function mse_out = mse_reg(actual,label)
mse_out = (actual - label).^2; %%% TODO: Check what D is in case.
end