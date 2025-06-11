# Örneklem sayısı azsa (<30) veya standart sapma bilinmiyorsa kullanılır.

import numpy as np
from scipy import stats

n = 30
xOrt = 140
xSsapma = 25
guven = 0.95
sDerecesi = n - 1

aralik = stats.t.interval(
    confidence=guven, loc=xOrt, df=sDerecesi, scale=xSsapma / np.sqrt(n)
)
print(aralik)
