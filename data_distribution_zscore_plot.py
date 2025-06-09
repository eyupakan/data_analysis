import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# seed yapısı her program çalıştığında farklı random veriseti vermek yerine sabit bir veriseti üretir.

np.random.seed(0)

# Ortalama 30, standart sapma 2 olan normal dağılımlı 15.000 adet veri üretiliyor
data = np.random.normal(30, 2, 15000)

# manuel zscore
# data_z = (data - np.mean(data)) / np.std(data)

data_z = stats.zscore(data)

# orijinal veri için grafik oluşturma

# sns.displot(data, kde=True)
# plt.title("Veri Dağılım Grafiği", fontsize=14, loc="center", c="green")
# plt.xlabel("veriler", fontsize=13, c="red")
# plt.ylabel("frekans", fontsize=13, c="blue")
# plt.axvline(x=np.mean(data), linestyle="--", linewidth=2.5, label="ortalama", c="red")
# plt.axvline(
#     x=np.mean(data) - np.std(data),
#     linestyle="--",
#     linewidth=2.5,
#     label="ortalama-1 standart sapma",
#     c="green",
# )
# plt.axvline(
#     x=np.mean(data) + np.std(data),
#     linestyle="--",
#     linewidth=2.5,
#     label="ortalama +1 standart sapma",
#     c="black",
# )


sns.displot(data_z, kde=True)
plt.title("Veri Dağılım Grafiği", fontsize=14, loc="center", c="green")
plt.xlabel("veriler", fontsize=13, c="red")
plt.ylabel("frekans", fontsize=13, c="blue")
plt.axvline(x=np.mean(data_z), linestyle="--", linewidth=2.5, label="ortalama", c="red")
plt.axvline(
    x=np.mean(data_z) - np.std(data_z),
    linestyle="--",
    linewidth=2.5,
    label="1 standart sapma",
    c="green",
)
plt.axvline(
    x=np.mean(data_z) + np.std(data_z),
    linestyle="--",
    linewidth=2.5,
    c="green",
)


plt.legend()
plt.show()
