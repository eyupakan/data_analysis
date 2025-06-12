# Bir fabrikada A ve B ürünlerinin ağırlıklarının varyansları sırasıyla 164gr 216gr. A ürününden 28 adet, B ürününden 30 adet örneklem alındığında A
# ürününün ortalama ağırlığı 32gr, B ürününün ortalama ağırlığı 26gr çıkmıştır.
# Bu verilere göre A ve B ürünlerinin ortalama ağırlıklarının farkını %95 güven aralığıyda bulunuz?

# İki grup (popülasyon) arasında ortalama farkı var mı diye bakmak istediğinde, o farkın tahmini güven aralığını hesaplarız.

import numpy as np
from scipy import stats

"""
na = 28
nb = 30
varA = 164
varB = 216
ortA = 32
ortB = 26
guven = 0.95


aralik = stats.norm.interval(
    confidence=guven, loc=(ortA - ortB), scale=np.sqrt((varA / na) + (varB / nb))
)

"""
# 2 farklı hasta grubu arasında 8 ve 10 bireylerden oluşan örneklemler çekilmiştir. Bu iki grubun bir virüse karşı reaksiyon verme zaman ortalamaları sırasıyla 3 ve 2.7'dir.
# Birleştirilmiş varyans 0,05 olarak hesaplandığına göre bu iki farklı hasta grubunun virüse karşı verdiği reaksiyon zaman farklarını %95 güven ile bulunuz?

na = 8
nb = 10
birVar = 0.05
ortA = 3
ortB = 2.7
guven = 0.95
n3 = (1 / na) + (1 / nb)

aralik = stats.t.interval(
    confidence=guven, df=(na + nb - 2), loc=(ortA - ortB), scale=np.sqrt(n3 * birVar)
)
print(aralik)
