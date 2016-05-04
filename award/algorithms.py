import numpy as np
from sklearn.neighbors import NearestNeighbors, DistanceMetric, BallTree
from numba import jit, jitclass, int_, float_, void

@jit(nopython=True)
def dtw_metric(s1, s2, r):
    """
    Get the dynamic time warping distance between time series s1 and s2,
    with a maximum warping of r
    """
    # Make sure r is valid
    r = max(r, abs(len(s1) - len(s2)))

    # Initialize dictionary for dynamic programming
    DTW = np.ones((len(s1)+1, len(s2)+1)) * np.inf
    DTW[0][0] = 0

    # Build up the DTW structure
    for i, s1val in enumerate(s1):
        for j in range(max(0, i-r), min(len(s2), i+r)):
            s2val = s2[j]
            dist = (s1val - s2val)**2
            DTW[i+1][j+1] = dist + min(DTW[i][j+1],
                                   DTW[i+1][j],
                                   DTW[i][j])
    return np.sqrt(DTW[len(s1)][len(s2)])



class DTW_Metric(object):
    """
    Dynamic Time Warping metric
    """
    #@void(int_)
    def __init__(self, metric=None):
        """ Initialize the DTW with the given metric. Default is euclidean """
        if metric is None:
            self.metric = self._euclidian
        else:
            self.metric = metric

    #@float_(float_, float_)
    def _euclidian(self, x1, x2):
        return (x1-x2)**2

    #@float_(float_[:], float_[:], float_)
    def __call__(self, s1, s2, r=None):
        """
        Get the dynamic time warping distance between time series s1 and s2,
        with a maximum warping of r
        """
        # Make sure r is a valid value
        if r is None:
            r = min(len(s1), len(s2))
        else:
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
        return np.sqrt(DTW[(len(s1)-1, len(s2)-1)])



class NaiveRankReduction(object):
    def __init__(self, metric=None):
        if metric is None:
            self.metric = DTW_Metric()


    def _remove_duplicates(self, data, classes):
        good_indices = np.ones(data.shape[0])
        for i, ts in enumerate(data):
            for j in range(i+1, len(data)):
                if len(ts) == len(data[j]) and all(np.isclose(val1, val2) for val1, val2 in zip(ts, data[j])):
                    good_indices[i] = 0
        idx = good_indices.astype(bool)
        return data[idx], classes[idx]

    def _neighbors(self, data):
        r = data.shape[1]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree',
                                metric='pyfunc', func=dtw_metric,
                                metric_params=dict(r=r), n_jobs=-1)
        nbrs.fit(data)
        distance, indices = nbrs.kneighbors(data, n_neighbors=2)

        return distance[:, 1], indices[:, 1]


    def valid_range(self, N):
        for i in range(N):
            if i not in self.discarded:
                yield i


    def fit(self, data, classes):
        data, classes = self._remove_duplicates(data, classes)
        neighbor_distance, neighbor_index = self._neighbors(data)

        N = len(data)
        match = classes == classes[neighbor_index]
        self.discarded = []
        for loop_num in range(N):
            print(loop_num)
            # Get the rank of each value
            rankings = [0 if match[i] else -np.inf for i in self.valid_range(N)]
            priority = [0.0 for _ in self.valid_range(N)]
            original_index = list(self.valid_range(N))
            for k, i in enumerate(self.valid_range(N)):
                #print(k, i)
                if np.isfinite(rankings[k]):
                    for j in self.valid_range(N):
                        if neighbor_index[j] == i:
                            #print(j)
                            # Time-series i is the nearest neighbor to j
                            rankings[k] += 1 if classes[j] == classes[i] else -2
                            priority[k] += 1.0 / neighbor_distance[j]**2

            # Find the lowest-ranked value
            ranking_order = [idx for (idx, r, p) in
                             sorted(zip(original_index, rankings, priority),
                                    key=lambda x: (x[1], x[2]))]
            self.discarded.append(ranking_order[0])
            print(self.discarded[-1], '\n')
        return self.discarded
