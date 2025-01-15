import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform

def linkage(X,
            method='single',
            metric='euclidean',
            *args,
            **kwargs):
    """
    A wrapper around scipy.cluster.hierarchy.linkage that emulates the syntax
    of MATLAB's linkage function.

    Parameters
    ----------
    X : ndarray
        - If X is 2D, interpret as data (n_samples x n_features).
        - If X is 1D, interpret as a condensed distance matrix
          of length n_samples*(n_samples-1)//2.
    method : str, optional
        Linkage method. One of:
          'single', 'complete', 'average', 'weighted', 'centroid',
          'median', 'ward'.
        Default is 'single'.
    metric : str, optional
        Distance metric to use if X is observation data. One of:
          'euclidean', 'squaredeuclidean', 'seuclidean', 'fasteuclidean',
          'fastsquaredeuclidean', 'fastseuclidean', 'mahalanobis', 'cityblock',
          'minkowski', 'chebychev', 'cosine', 'correlation', 'hamming',
          'jaccard', 'spearman'.
        Default is 'euclidean'.

    *args:
        Positional arguments that might appear in MATLAB calls, e.g.
        Z = linkage(X,method,metric,'savememory',value).
        These are ignored or parsed for MATLAB-compatibility only.
    **kwargs:
        Additional keyword arguments, for example:
          - 'V' (std dev array) if metric='seuclidean'
          - 'VI' or 'C' (covariance or inverse covariance) if metric='mahalanobis'
          - 'p' (exponent) if metric='minkowski'
        Or 'savememory' for MATLAB-compatibility.

    Returns
    -------
    Z : ndarray of shape (n-1, 4)
        A linkage matrix encoding the hierarchical clustering. Each row
        represents a merge of two clusters. The columns are:
          1. idx1 : int
          2. idx2 : int
          3. dist : float
          4. size : int
        where 'size' is the total number of original observations in
        the newly formed cluster.

    Notes
    -----
    - If 'centroid', 'median', or 'ward' is chosen, it is assumed that the
      distances are Euclidean. If a condensed distance vector is passed,
      SciPy will attempt to verify it or compute centroid/median/Ward
      distances accordingly.
    - 'fasteuclidean', 'fastsquaredeuclidean', and 'fastseuclidean' are
      mapped internally to their standard counterparts: 'euclidean',
      'squaredeuclidean', and 'seuclidean'.
    - 'savememory' is a no-op in this Python wrapper.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 3)
    >>> Z = linkage(X, 'ward')
    >>> Z
    array([[ 0.        ,  5.        ,  0.07603864,  2.        ],
           [ 2.        ,  7.        ,  0.21194439,  2.        ],
           ...
           [ 8.        , 10.        ,  0.56467721, 10.        ]])
    """

    # --- 1) Parse 'savememory' if present (MATLAB compatibility). ---
    #     We do not do anything with it in SciPy, but let's remove it from kwargs.
    if 'savememory' in kwargs:
        _ = kwargs.pop('savememory', None)  # no-op, just pop it

    # --- 2) Map any "fast*" metric to normal ones for SciPy. ---
    metric_map = {
        'fasteuclidean': 'euclidean',
        'fastsquaredeuclidean': 'squaredeuclidean',
        'fastseuclidean': 'seuclidean'
    }
    if metric in metric_map:
        metric = metric_map[metric]

    # --- 3) If the user provided a condensed distance matrix (1D),
    #         then call scipy_linkage directly. ---
    X = np.asarray(X)
    if X.ndim == 1:
        # We assume X is already a condensed distance matrix
        # SciPy's linkage can handle it directly if method is valid
        Z = scipy_linkage(X, method=method)
        return Z

    # --- 4) If X is a 2D observation matrix, compute distances using pdist. ---
    #     We pass any additional parameters in **kwargs to pdist if relevant.
    #     For example: pdist(X, metric='mahalanobis', VI=..., ...).
    if X.ndim == 2:
        # For 'squaredeuclidean', SciPy expects metric='euclidean' with an extra
        # note that the squared distances will be handled differently.
        # Alternatively, we can compute them ourselves or pass a custom function.
        # The most straightforward approach is:
        if metric == 'squaredeuclidean':
            # SciPy doesn't have a direct 'squaredeuclidean' metric name.
            # We'll compute the Euclidean distances and square them.
            # Then call linkage on that condensed matrix:
            dist_array = pdist(X, metric='euclidean', **kwargs)
            # Square the distances:
            dist_array = dist_array**2
            Z = scipy_linkage(dist_array, method=method)
            return Z
        else:
            # For all other valid metrics, we can pass them directly to pdist.
            dist_array = pdist(X, metric=metric, **kwargs)
            Z = scipy_linkage(dist_array, method=method)
            return Z

    # --- 5) If none of the above, raise an error (e.g., user provided a
    #         3D array or something invalid). ---
    raise ValueError(
        "Input X must be either a 2D observation matrix or a 1D condensed distance "
        "matrix. Got shape: {}".format(X.shape)
    )
