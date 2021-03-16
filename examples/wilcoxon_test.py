#%%
import numpy as np
import scipy.stats
import math
# paired Wilcoxon signed rank test
kls = np.loadtxt('saved_data/2608_experiment/alpha_1_scale_02_kls.txt')
baseline_kls = np.loadtxt('saved_data/2908_train_acc_prob/alpha_1/kls.txt')
abs_diffs = np.abs(kls - baseline_kls)
sgn_diffs = np.sign(kls - baseline_kls)
sorted_ids = np.argsort(abs_diffs)
ranks = np.arange(1, 61)
sorted_sgns = sgn_diffs[sorted_ids]
W = np.sum(ranks * sorted_sgns)
print("W: ", W)
sigma_W = math.sqrt( ( 60 * (60 + 1) * (2*60 + 1) )/6.0  )
print("z score: ", W / sigma_W)
print("cdf(z_score): ", scipy.stats.norm().cdf(W/sigma_W))
# %%
