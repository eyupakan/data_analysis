import numpy as np
from scipy import stats

# 100 ürünün ortalama ağırlığı 1040 standart sapması 25'tir. tüm ürünlerin ortalama ağırlıkları %95 güven aralığında kaçtır?

n = 100
xOrt = 1040
xStandart = 25
guven = 0.95

aralik = stats.norm.interval(confidence=guven, loc=xOrt, scale=xStandart / np.sqrt(n))

print(aralik)
