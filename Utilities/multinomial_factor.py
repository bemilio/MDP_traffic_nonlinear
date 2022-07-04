from  math import factorial

# Multinomial factor of degree k over (vector) v
def multinomial_factor(degree, v):
    denominator = 1
    for i in range(v.size()[0]):
        denominator = denominator * factorial(int(v[i].item()))
    numerator = factorial(degree)
    return numerator/denominator