import numpy as np
import diptest
import matplotlib.pyplot as plt

# generate some bimodal random draws
N = 10000
hN = N // 2
x = np.empty(N, dtype=np.float64)
x[:hN] = np.random.normal(-1, 1.0, hN)
x[hN:] = np.random.normal(1.5, 1.0, hN)

plt.hist(x, bins=100)
plt.show()
# only the dip statistic
dip = diptest.dipstat(x)

# both the dip statistic and p-value
dip, pval = diptest.diptest(x)
print(dip, pval)