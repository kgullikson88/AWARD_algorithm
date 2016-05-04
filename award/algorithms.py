import numpy as np
from sklearn.neighbors import NearestNeighbors, DistanceMetric

class DTW_Metric(object):
    """
    Dynamic Time Warping metric
    """
    def __init__(self, metric=None):
        """ Initialize the DTW with the given metric. Default is euclidean """
        if metric is None:
            self.metric = self._euclidian
        else:
            self.metric = metric

    def _euclidian(self, x1, x2):
        return (x1-x2)**2

    def __call__(self, s1, s2, r):
        """
        Get the dynamic time warping distance between time series s1 and s2,
        with a maximum warping of r
        """
        # Make sure r is a valid value
        r = max(r, abs(len(s1) - len(s2)))

        # Initialize dictionary for dynamic programming
        DTW = {}
        for i in range(-1,len(s1)):
            for j in range(-1,len(s2)):
                DTW[(i, j)] = np.inf
        DTW[(-1, -1)] = 0

        # Build up the DTW structure
        for i, s1val in enumerate(s1):
            for j in range(max(0, i-r), min(len(s2), i+r)):
                s2val = s2[j]
                dist = self.metric(s1val, s2val)
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],
                                         DTW[(i, j-1)],
                                         DTW[(i-1, j-1)])
        return np.sqrt(DTW(len(s1)-1, len(s2)-1))



class NaiveRankReduction(object):
    def __init__(self):
        pass

    def _remove_duplicates(self, data, classes):
        raise NotImplementedError

    def fit(self, data, classes):
        data, classes = self._remove_duplicates(data, classes)
