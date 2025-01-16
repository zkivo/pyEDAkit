import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster, inconsistent
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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

##############################################################################
# MATLAB-style cluster function (wrapper around SciPy fcluster)
##############################################################################
def cluster(Z, *args, **kwargs):
    """
    A wrapper around SciPy's fcluster and inconsistent that emulates
    MATLAB's cluster function for hierarchical clustering.

    Syntax:
    --------
    T = cluster(Z,'Cutoff',C)
    T = cluster(Z,'Cutoff',C,'Depth',D)
    T = cluster(Z,'Cutoff',C,'Criterion',criterion)
    T = cluster(Z,'MaxClust',N)

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix (output of
        the `linkage` function). Typically shape (m-1, 4) in SciPy.

    *args:
        Positional arguments used to mimic MATLAB name-value pairs:
         - 'Cutoff', C
         - 'Depth', D
         - 'Criterion', criterion
         - 'MaxClust', N
         Each can also be passed in multiple-value form (e.g., C or N can be arrays).

    **kwargs:
        Same as *args but in key=value style. e.g. cluster(Z, Cutoff=..., Depth=...).

    Returns
    -------
    T : ndarray
        A numeric vector or matrix of cluster assignments for each observation.
        - If C or N is a scalar, T is 1D of length m (where m is #observations).
        - If C or N is a length L array, T is an m-by-L matrix, each column
          containing cluster assignments for the corresponding C or N value.

    Notes
    -----
    - 'inconsistent' is the default criterion for 'Cutoff', with Depth=2 by default.
    - If 'Criterion'='distance', cluster uses a distance threshold in fcluster.
    - If 'Criterion'='inconsistent', cluster uses the "inconsistent" threshold in fcluster.
    - If 'MaxClust' is provided, it uses criterion='maxclust' in fcluster,
      forcing a maximum of N clusters. This uses a distance-based cut.
    - For multiple values of C or N, the output T is a matrix with one column
      per threshold or cluster count.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 3)
    >>> Z = linkage(X, 'ward')
    >>> # Cutoff by distance at 1.0
    >>> T = cluster(Z, 'Cutoff', 1.0, 'Criterion', 'distance')
    >>> # Force exactly 3 clusters
    >>> T2 = cluster(Z, 'MaxClust', 3)
    >>> # Multiple cutoffs
    >>> T3 = cluster(Z, 'Cutoff', [0.7, 1.2, 1.5], 'Criterion', 'distance')
    """
    # Default values
    cutoff = None
    maxclust = None
    criterion = 'inconsistent'  # default in MATLAB
    depth = 2                   # default Depth in MATLAB

    # --- 1) Parse the *args in MATLAB-like name-value pairs ---
    i = 0
    while i < len(args):
        if i+1 >= len(args):
            raise ValueError(f"Parameter '{args[i]}' is missing its value.")
        name = str(args[i]).lower()
        val = args[i+1]
        i += 2

        if name == 'cutoff':
            cutoff = val
        elif name == 'maxclust':
            maxclust = val
        elif name == 'criterion':
            criterion = str(val).lower()
        elif name == 'depth':
            depth = int(val)
        else:
            raise ValueError(f"Unrecognized parameter: '{args[i]}'")

    # --- 2) Parse **kwargs similarly ---
    for k, v in kwargs.items():
        k_lower = k.lower()
        if k_lower == 'cutoff':
            cutoff = v
        elif k_lower == 'maxclust':
            maxclust = v
        elif k_lower == 'criterion':
            criterion = str(v).lower()
        elif k_lower == 'depth':
            depth = int(v)
        else:
            raise ValueError(f"Unrecognized parameter: '{k}'")

    # Convert to NumPy arrays if they are not already
    if cutoff is not None and not np.isscalar(cutoff):
        cutoff = np.asarray(cutoff).ravel()
    if maxclust is not None and not np.isscalar(maxclust):
        maxclust = np.asarray(maxclust).ravel()

    # We need the number of observations, m, to shape the output.
    # SciPy linkage has shape (m-1, 4). So m = Z.shape[0] + 1
    m = Z.shape[0] + 1

    # --- 3) If 'MaxClust' is provided, do an fcluster for each N in maxclust. ---
    if maxclust is not None:
        # If maxclust is scalar, just do one call.
        if np.isscalar(maxclust):
            N = int(maxclust)
            T = fcluster(Z, t=N, criterion='maxclust')
            return T
        else:
            # We have a vector of N values => produce an m-by-l matrix
            T_out = np.zeros((m, len(maxclust)), dtype=int)
            for iN, N in enumerate(maxclust):
                cluster_assign = fcluster(Z, t=N, criterion='maxclust')
                T_out[:, iN] = cluster_assign
            return T_out

    # --- 4) If 'Cutoff' is provided, do an fcluster for each c in cutoff. ---
    if cutoff is not None:
        # If the user did not specify criterion, it remains 'inconsistent' by default.

        # If cutoff is scalar, do one call. If array, do multiple columns.
        if np.isscalar(cutoff):
            c_val = float(cutoff)
            if criterion == 'distance':
                T = fcluster(Z, t=c_val, criterion='distance')
                return T
            elif criterion == 'inconsistent':
                T = fcluster(Z, t=c_val, criterion='inconsistent', depth=depth)
                return T
            else:
                raise ValueError(
                    "Only 'inconsistent' or 'distance' are valid cluster criteria with Cutoff."
                )
        else:
            # cutoff is an array => produce an m-by-l matrix
            T_out = np.zeros((m, len(cutoff)), dtype=int)
            for ic, c_val in enumerate(cutoff):
                if criterion == 'distance':
                    cluster_assign = fcluster(Z, t=float(c_val), criterion='distance')
                elif criterion == 'inconsistent':
                    cluster_assign = fcluster(Z, t=float(c_val),
                                              criterion='inconsistent',
                                              depth=depth)
                else:
                    raise ValueError(
                        "Only 'inconsistent' or 'distance' are valid cluster criteria with Cutoff."
                    )
                T_out[:, ic] = cluster_assign
            return T_out

    # --- 5) If neither 'MaxClust' nor 'Cutoff' is specified, error out. ---
    raise ValueError(
        "No valid clustering instruction found. Use 'Cutoff',C or 'MaxClust',N."
    )


def kmeans(X, k, *args, **kwargs):
    """
    K-means clustering in the style of MATLAB's kmeans function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix. Rows correspond to observations, columns to variables.
    k : int
        Number of clusters.

    *args :
        Positional arguments that follow MATLAB's 'Name',Value syntax, e.g.:
          ('Distance','sqeuclidean','Replicates',5,'MaxIter',200, ...)

    **kwargs :
        Pythonic keyword arguments that likewise follow MATLAB's naming, e.g.:
          Distance='sqeuclidean', Replicates=5, MaxIter=200, etc.

    Returns
    -------
    idx : ndarray of shape (n_samples,)
        Cluster index (label) for each observation (1-based in MATLAB, 0-based
        in scikit-learn, but we add +1 to match MATLAB).
    C : ndarray of shape (k, n_features)
        Final centroid locations.
    sumd : ndarray of shape (k,)
        Within-cluster sum of distances. sumd[j] is the sum of distances
        between all points assigned to cluster j and the centroid of cluster j.
    D : ndarray of shape (n_samples, k)
        Distances from each point (row) to every centroid (column).

    Notes
    -----
    - By default, uses 'Distance' = 'sqeuclidean' (the usual squared Euclidean),
      which aligns with scikit-learn’s KMeans.
    - Supports basic name-value pairs:
         'Distance':   'sqeuclidean' (default), other metrics raise NotImplementedError
         'Start':      'plus' (k-means++), 'sample' (random), or user-provided
                       numeric matrix
         'Replicates': mapped to scikit-learn's n_init
         'MaxIter':    mapped to max_iter
         'Display':    'off','final','iter' -> controls verbosity (0 or 1)
         'EmptyAction','OnlinePhase','Options': accepted but either ignored or raise
                       warnings (since scikit-learn doesn't support them directly)
    - Returns all four outputs. In MATLAB usage, e.g.
         idx = kmeans(...);
         [idx,C] = kmeans(...);
      simply ignore the extra outputs in Python.

    Examples
    --------
    >>> import numpy as np
    >>> # Suppose we have 2D data
    >>> X = np.array([[1,2],[1,4],[1,0],
    ...               [10,2],[10,4],[10,0]])
    >>> # Basic call
    >>> idx, C, sumd, D = kmeans(X, 2)
    >>> print(idx)  # cluster assignments
    >>> print(C)    # final centroids
    >>> print(sumd) # within-cluster sums
    >>> print(D)    # distances from each point to each centroid
    """

    # --- 1) Default parameter values in the spirit of MATLAB ---
    distance = 'sqeuclidean'
    start = 'plus'         # (i.e., 'k-means++')
    replicates = 1
    maxiter = 100
    display = 'off'        # 'off', 'final', or 'iter'
    # Some parameters that we won't fully implement, but parse anyway:
    emptyaction = 'singleton'
    onlinephase = 'off'
    options = None

    # --- 2) Parse *args in (Name, Value) pairs, just like MATLAB. ---
    i = 0
    while i < len(args):
        if i+1 >= len(args):
            raise ValueError(f"Parameter '{args[i]}' has no corresponding value.")
        param_name = str(args[i]).lower()
        param_val = args[i+1]
        i += 2
        if param_name == 'distance':
            distance = param_val
        elif param_name == 'start':
            start = param_val
        elif param_name == 'replicates':
            replicates = param_val
        elif param_name == 'maxiter':
            maxiter = param_val
        elif param_name == 'display':
            display = param_val
        elif param_name == 'emptyaction':
            emptyaction = param_val
        elif param_name == 'onlinephase':
            onlinephase = param_val
        elif param_name == 'options':
            options = param_val
        else:
            raise ValueError(f"Unrecognized parameter name: '{args[i]}'")

    # --- 3) Parse **kwargs in a Pythonic style. ---
    for key, val in kwargs.items():
        key_lower = key.lower()
        if key_lower == 'distance':
            distance = val
        elif key_lower == 'start':
            start = val
        elif key_lower == 'replicates':
            replicates = val
        elif key_lower == 'maxiter':
            maxiter = val
        elif key_lower == 'display':
            display = val
        elif key_lower == 'emptyaction':
            emptyaction = val
        elif key_lower == 'onlinephase':
            onlinephase = val
        elif key_lower == 'options':
            options = val
        else:
            raise ValueError(f"Unrecognized parameter: '{key}'")

    # --- 4) Handle distance. Currently only support 'sqeuclidean'. ---
    if distance.lower() != 'sqeuclidean':
        raise NotImplementedError("Only 'sqeuclidean' distance is supported in this wrapper.")

    # --- 5) Handle Start. ---
    #   'plus'    -> init='k-means++'
    #   'sample'  -> init='random'
    #   'uniform' or 'cluster' -> raise NotImplementedError or partial
    #   numeric   -> user-supplied initial centers => must be shape (k, n_features)
    init_param = 'k-means++'
    if isinstance(start, str):
        s_lower = start.lower()
        if s_lower == 'plus':
            init_param = 'k-means++'
        elif s_lower == 'sample':
            init_param = 'random'
        elif s_lower in ['cluster', 'uniform']:
            raise NotImplementedError(f"Start='{start}' is not supported in this wrapper.")
        else:
            raise ValueError(f"Unrecognized 'Start' option: {start}")
    else:
        # If it's an array or numeric matrix, we treat it as the user specifying
        # initial cluster centroids. Must have shape (k, n_features) or shape (k, n_features, r)
        init_arr = np.asarray(start, dtype=float)
        if init_arr.ndim == 2:
            # shape: (k, p)
            if init_arr.shape[0] != k:
                raise ValueError("The first dimension of Start does not match k.")
            init_param = init_arr
        elif init_arr.ndim == 3:
            # shape: (k, p, r) => implies multiple replicates
            # scikit-learn only takes a single init. We can roll out multiple fits manually...
            # For simplicity, we handle only the first page or raise an error.
            raise NotImplementedError("3D arrays for 'Start' are not supported in this basic wrapper.")
        else:
            raise ValueError("Start array must be 2D or 3D.")

    # --- 6) Map Replicates -> n_init, MaxIter -> max_iter. ---
    n_init_param = replicates
    max_iter_param = maxiter

    # --- 7) Map Display -> verbose. ---
    #   'off'   => 0
    #   'final' => 0 (scikit-learn doesn't have a final summary)
    #   'iter'  => 1
    verbose_param = 0
    disp_lower = display.lower()
    if disp_lower == 'iter':
        verbose_param = 1
    elif disp_lower in ['final', 'off']:
        verbose_param = 0
    else:
        raise ValueError(f"Unrecognized 'Display' option: {display}")

    # --- 8) Warn or ignore unsupported name-value pairs. ---
    if emptyaction.lower() != 'singleton':
        # scikit-learn does not support reassigning empty clusters,
        # so we just warn or raise an error if user sets something else.
        print(f"Warning: 'EmptyAction'='{emptyaction}' is not fully supported. "
              f"Using default scikit-learn behavior (error if a cluster is empty).")
    if onlinephase.lower() != 'off':
        print(f"Warning: 'OnlinePhase'='{onlinephase}' is not implemented. Using standard batch updates.")
    if options is not None:
        print("Warning: 'Options' is not fully supported. Ignoring in this wrapper.")

    # --- 9) Fit the model using scikit-learn's KMeans. ---
    X = np.asarray(X, dtype=float)
    # Handle missing data (NaNs) as MATLAB does (remove rows):
    nan_mask = np.isnan(X).any(axis=1)
    removed_indices = np.where(nan_mask)[0]
    keep_mask = ~nan_mask
    X_valid = X[keep_mask]

    if len(X_valid) == 0:
        raise ValueError("All rows of X contain NaNs, cannot perform k-means.")

    # Create KMeans object
    kmeans_model = KMeans(
        n_clusters=k,
        init=init_param,
        n_init=n_init_param,
        max_iter=max_iter_param,
        verbose=verbose_param,
        tol=1e-4,  # default tolerance
        algorithm='lloyd',  # standard, similar to MATLAB batch updates
        random_state=None  # You could parse from 'options' or user input
    )

    kmeans_model.fit(X_valid)
    labels_valid = kmeans_model.labels_
    centers = kmeans_model.cluster_centers_

    # --- 10) Construct outputs. ---
    # idx: we need a vector of length n_samples. For rows with NaN, MATLAB kmeans returns NaN.
    idx_full = np.full(shape=(X.shape[0],), fill_value=np.nan)
    idx_full[keep_mask] = labels_valid
    # MATLAB labels are 1-based, while scikit-learn's are 0-based:
    idx_full = idx_full + 1  # convert to 1-based

    # C: the final centroid locations. shape (k, p)
    # For rows dropped due to NaNs in X, that doesn't change the centroid array. So it's fine.
    C = centers

    # sumd: Within-cluster sums of distances. We'll compute using the (squared) Euclidean distances
    # consistent with 'sqeuclidean'.
    #   sumd[j] = sum of distances between all points in cluster j and center j
    # scikit-learn's inertia_ is the sum of squared distances to centroids across all clusters.
    # But we want the sum per cluster. We'll do it manually.
    # Distances from each valid point to each centroid:
    D_valid = cdist(X_valid, C, metric='euclidean')**2  # squared Euclidean
    # For each cluster j, sum the distances
    sumd_array = np.zeros(k)
    for j in range(k):
        in_cluster_j = (labels_valid == j)
        sumd_array[j] = D_valid[in_cluster_j, j].sum()

    # D: Distances from each point to each centroid => shape (n_samples, k)
    # For rows with NaNs, we return all NaN
    D_full = np.full(shape=(X.shape[0], k), fill_value=np.nan)
    D_full[keep_mask, :] = D_valid

    # Return the 4 outputs
    return idx_full, C, sumd_array, D_full
