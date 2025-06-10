from scipy.stats import norm

# 4.5'ten küçük olma olasılığı
# olasılık = norm(loc=5.3, scale=1).cdf(4.5)

# 4.5'ten büyük olma olasılığı
# olasılık = 1 - norm(loc=5.3, scale=1).cdf(4.5)


# print(olasılık)


# p(4.5<x<6.5) için

ustSinir = norm(loc=5.3, scale=1).cdf(6.5)
altSinir = norm(loc=5.3, scale=1).cdf(4.5)

olasılık = ustSinir - altSinir
print(olasılık)
