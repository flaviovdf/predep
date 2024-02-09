import numpy as np

from bandwidth_estimator import *
from sklearn.neighbors import KernelDensity
from scipy import integrate
from multiprocessing import Pool
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def integral_kde(kde,bounds, density_function=lambda x: x):
    """
    Calculate the integral of a given kernel density estimate (KDE) over specified bounds.

    Parameters:
        kde (sklearn.neighbors.KernelDensity): The fitted KDE model.
        bounds (list of tuples): The integration bounds for each dimension of the KDE.
        density_function (callable, optional): A function to apply to the KDE values before integrating.

    Returns:
        float: The integral value.
    """

    def funct(*args):
        point = list(args)
        p = np.exp(kde.score_samples([point]))
        return density_function(p)

    integral_value = integrate.nquad(funct, bounds)[0]
    return integral_value

def predep_s(s, bounds_type = "ref", bandwidth="ISJ"):
    # Fit kernel density estimation for s dimension
    bandwidth_s = get_bandwidth(s,bandwidth_type=bandwidth)
    kde_s = KernelDensity(kernel="gaussian", bandwidth=bandwidth_s)
    kde_s.fit(s)

    # Define density function for integration
    density_function = lambda x: x ** 2

    dims = kde_s.n_features_in_
    if bounds_type == "inf":
        bounds = [[-np.inf, np.inf] for _ in range(dims)]
    elif bounds_type == "ref":
        lower_bounds = list(np.min(s, axis=0))
        upper_bounds = list(np.max(s, axis=0))

        bounds = [[lower_bounds[i],upper_bounds[i]] for i in range(len(lower_bounds))]

    # Calculate tau_s using integral_kde function
    return integral_kde(kde_s, density_function=density_function, bounds=bounds)

def calc_predep_s_t(s_t, sample_points, bandwidth):
    if len(s_t) <= s_t.shape[1]:
        return 0

    if sample_points is None:
        sample_points = int(len(s_t) * np.log(len(s_t))) + s_t.shape[1] + 1

    # Randomly sample x1 and x2 from points within the bin
    indices = np.random.randint(0, len(s_t), size=(sample_points, 2))
    x1, x2 = s_t[indices[:, 0]], s_t[indices[:, 1]]

    # Calculate w as the difference between x1 and x2
    w = x1 - x2

    # Fit kernel density estimation for s_t dimension
    bandwidth_s_t = get_bandwidth(w, bandwidth_type=bandwidth)
    kde_s_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth_s_t)
    kde_s_t.fit(w)

    # Calculate t_val by evaluating kernel density estimation for s-t dimension at 0 and multiplying by p
    t_val = np.exp(kde_s_t.score_samples([[0] * s_t.shape[1]]))[0]

    return t_val


def fragment_space(s, t, max_size_frag=1000, n_clusters=None):
    t_o = t.copy()  # Make a copy of t to avoid modifying the original array

    if len(t) > max_size_frag:
        # Randomly sample max_size_frag indices without replacement
        sample = np.random.choice(len(t), max_size_frag, replace=False)
        t = t[sample]

    if n_clusters is None:
        # Calculate the default number of clusters based on the square root of the data size
        n_clusters = int(np.sqrt(len(t)))

    def compute_cluster_distances(t):
        ids = []
        distance = []
        for i in range(t.shape[1]):
            # Compute hierarchical clustering distances for each dimension
            linkage_matrix = sch.linkage(t[:, i:i + 1], method="ward")
            cluster_distances = linkage_matrix[:, 2]
            ids = ids + [i] * len(cluster_distances)
            distance = distance + list(cluster_distances)

        # Sort distances and return the corresponding indices
        index = np.argsort(distance)[::-1]
        ids = np.array(ids)[index]
        return ids

    ids = compute_cluster_distances(t)

    n_cluster_per_dim = [1 for _ in range(t.shape[1])]

    for i in ids:
        # Increment the number of clusters for a dimension while keeping the total below n_clusters
        n_cluster_per_dim[i] += 1
        if np.prod(n_cluster_per_dim) > n_clusters:
            n_cluster_per_dim[i] -= 1

    bounds_all = []
    for i in range(len(n_cluster_per_dim)):
        p_1 = np.sort(t[:, i:i + 1], axis=0)
        # Perform hierarchical clustering to determine cluster boundaries for each dimension
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_cluster_per_dim[i], metric='euclidean', linkage='ward')
        labels = hierarchical_cluster.fit_predict(p_1)
        bounds = [-np.inf]
        p_last = None
        l_last = labels[0]
        for i in range(len(p_1)):
            p = p_1[i]
            l = labels[i]
            if l != l_last:
                # Calculate cluster boundary as the midpoint between two data points
                bounds.append(((p + p_last) / 2)[0])
            p_last = p
            l_last = l
        bounds.append(np.inf)
        bounds_all.append(bounds)

    def get_bin(p, bounds_all):
        bin_result = []

        for i in range(len(p)):
            bounds = bounds_all[i]
            for b in range(1, len(bounds)):
                if p[i] <= bounds[b]:
                    # Assign a bin index based on data point location within cluster boundaries
                    bin_result.append(b - 1)
                    break

        return bin_result

    # Compute the bins for each data point in t_o
    part = np.array([get_bin(p, bounds_all) for p in t_o])

    # Create a list of unique labels (combination of bins)
    labels = np.unique(part, axis=0)

    frag = []
    for i in range(len(labels)):
        label = labels[i]
        # Filter data points based on the assigned label (combination of bins)
        label_indices = np.all(part == label, axis=1)
        points_t = s[label_indices]
        bin_start = None
        bin_end = None
        frag.append([points_t, bin_start, bin_end])

    return frag

def predep_s_t_multi(args):
    fragment,sample_points,bandwidth,kde_t,size,ref_estimator = args

    s_t, bin_start, bin_end = fragment
    t_val = calc_predep_s_t(s_t, sample_points, bandwidth)

    if ref_estimator == "integral":
        ref = integral_kde(kde_t, [[bin_start, bin_end]], density_function=lambda x: x)
    elif ref_estimator == "proportion":
        ref = len(s_t)/size
    elif ref_estimator == "center":

        if len(s_t) == 0:
            ref = 0
        else:
            if bin_start == -np.inf:
                bin_start = np.min(s_t)

            if bin_end == np.inf:
                bin_end = np.max(s_t)

            center = (bin_start + bin_end) / 2

            p = np.exp(kde_t.score_samples([[center]]))[0]
            dist = (bin_end - bin_start)
            ref = p * dist

    else:
        raise Exception("ref estiamtor not defined")

    return t_val * ref

def predep_s_t(s,t, sample_points=None, bandwidth="ISJ", num_threads=1,
               max_size_frag = 1000, n_clusters = None, ref_estimator = "proportion", get_all_data = False):

    # Fit kernel density estimation for t dimension
    bandwidth_t = get_bandwidth(points = t, bandwidth_type=bandwidth)
    kde_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth_t)
    kde_t.fit(t)

    points_frag = fragment_space(s = s, t = t, max_size_frag=max_size_frag, n_clusters=n_clusters)

    args = [[i,sample_points,bandwidth,kde_t, len(s), ref_estimator] for i in points_frag]

    if num_threads <= 0:
        # Use all available CPU cores if num_threads is negative
        num_threads = None

    if num_threads == 1:
        ts = []

        for arg in args:
            ts.append(predep_s_t_multi(arg))

    else:
        with Pool(processes=num_threads) as pool:
            ts = pool.map(predep_s_t_multi, args)

    if get_all_data:
        return ts, points_frag
    else:
        return np.sum(ts)

def predep(s,t, sample_points=None, bandwidth="ISJ", num_threads=1,
               max_size_frag = 1000, n_clusters = None, ref_estimator = "proportion",
           get_all_data = False, get_inter_values = False,space_cal = "bootstrap"):

    if space_cal == "integral":
        p_s = predep_s(s)
    elif space_cal == "bootstrap":
        p_s = calc_predep_s_t(s,None,bandwidth=bandwidth)

    p_s_t = predep_s_t(s,t, sample_points=sample_points, bandwidth=bandwidth, num_threads=num_threads,
               max_size_frag = max_size_frag, n_clusters = n_clusters, ref_estimator = ref_estimator,
                       get_all_data = get_all_data)

    predep = (p_s_t - p_s)/p_s_t

    if get_inter_values:
        return predep,p_s,p_s_t
    else:
        return predep
