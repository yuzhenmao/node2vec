import numpy as np


def alias_sampling(weights):
    # referred from the original code and https://blog.csdn.net/haolexiao/article/details/65157026:
    num = len(weights)
    prob = np.zeros(num)
    index = np.zeros(num, dtype=np.int)

    smaller = []
    larger = []

    # Normalization
    for ii, weight in enumerate(weights):
        prob[ii] = num * weight
        if prob[ii] < 1.0:
            smaller.append(ii)
        else:
            larger.append(ii)

    # Flatting the distribution
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        index[small] = large
        prob[large] = prob[large] + prob[small] - 1.0
        if prob[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return index, prob
