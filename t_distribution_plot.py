import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)


# Serbestlik derecesi 1 olan (degress of freedom)
veri1t = stats.t.rvs(loc=0, df=1, size=15)
veri2t = stats.t.rvs(loc=0, df=2, size=15)
veri3t = stats.t.rvs(loc=0, df=5, size=15)
veri4t = stats.t.rvs(loc=0, df=20, size=15)

plt.xlim(-5, 5)
sns.displot(veri1t, color="green", kde=True)
sns.displot(veri2t, color="red", kde=True)
sns.displot(veri3t, color="blue", kde=True)
sns.displot(veri4t, color="black", kde=True)
plt.show()
