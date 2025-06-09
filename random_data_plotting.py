import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

a = np.random.normal(45, 1, 10000)

# matplotlib ile grafik

# plt.hist(a, bins=100)
# plt.show()

# seaborn ile grafik

# sns.displot(a)
# sns.displot(a, color="g", kde=True)
sns.kdeplot(a)
plt.show()
