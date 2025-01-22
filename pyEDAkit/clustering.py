import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster, inconsistent
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import cophenet as scipy_cophenet
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import networkx as nx

def kmeans(X, k):
    """
    Perform k-means clustering on data X.

    Parameters
    ----------
    X : ndarray
        The data matrix (n_samples x n_features).
    k : int

    Returns
    -------
    idx : ndarray
        The cluster index of each point (n_samples,).

    centers : ndarray
        The cluster centers (k x n_features).
    """

    kmeans = KMeans(n_clusters=k)
    idx = kmeans.fit_predict(X)
    return idx, kmeans.cluster_centers_

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


def kmeans_matlab(X, k, *args, **kwargs):
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
    >>> idx, C, sumd, D = kmeans_matlab(X, 2)
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


def minspantree(G, *args, **kwargs):
    """
    Python wrapper that emulates MATLAB's minspantree function for an undirected graph.
    Uses NetworkX under the hood.

    Parameters
    ----------
    G : networkx.Graph
        An undirected graph object. The edges can have a 'weight' attribute to indicate
        their cost or distance.

    *args :
        Positional arguments, in MATLAB-style name-value pairs:
        e.g. ('Method','sparse','Root',2,'Type','forest').

    **kwargs :
        Keyword arguments in Pythonic style, but named as in MATLAB:
        e.g. Method='sparse', Root=2, Type='tree'.

    Returns
    -------
    T : networkx.Graph
        The minimum spanning tree (MST), or a forest if 'Type'='forest'.
        This subgraph has the same set of nodes as G, but only edges forming
        the MST are included. (For disconnected graphs with 'Type'='tree',
        T will have only the MST of the component containing 'Root'. For
        'Type'='forest', T will contain the MST edges of all components.)

    pred : dict
        A dictionary of predecessor nodes, keyed by node. `pred[u] = v` if
        v is the parent of u in the MST rooted at 'Root'. By convention,
        `pred[root] = 0`. If 'Type'='tree' and a node is not in the same
        component as 'Root', `pred[node] = np.nan`.
        For 'Type'='forest', nodes in components other than the root are
        still given a valid parent within their own connected component,
        but if you only care about the root's component, you can disregard
        the rest.

    MATLAB Name-Value Parameters
    ----------------------------
    Method : {'dense','sparse'}, default='dense'
        - 'dense'  => Use Prim's algorithm.
        - 'sparse' => Use Kruskal's algorithm.

    Root : int or str, default=1
        The root node. If 'Method'='dense', the MST is grown from this root
        (Prim’s algorithm). If 'Method'='sparse' (Kruskal), the root is only
        used to define the predecessor array.

    Type : {'tree','forest'}, default='tree'
        - 'tree'   => Return MST (or actually a spanning tree) only for the
                      connected component containing 'Root'.
        - 'forest' => Return a minimum spanning forest for **all** components
                      of G.

    Notes
    -----
    - `nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')`
      or `algorithm='prim'` is used behind the scenes.
    - Edges are assumed to have a 'weight' attribute. If not present, default
      weight = 1.
    - For a disconnected graph and Type='tree', T contains only the MST of
      the root’s connected component.
    - The predecessor array is computed by doing a BFS from the root on the
      MST subgraph. Edges are directed away from the root in `pred`.

    Examples
    --------
    >>> # Suppose G is an undirected NetworkX graph with N=5 nodes
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([
    ...     (1,2,4), (1,3,2), (2,3,1), (2,4,5), (3,5,7)
    ... ])
    >>> # Basic usage, returning MST of the connected component containing node=1:
    >>> T, pred = minspantree(G)
    >>> list(T.edges(data=True))
    [(1, 3, {'weight': 2}), (2, 3, {'weight': 1}), (2, 4, {'weight': 5}), (3, 5, {'weight': 7})]
    >>> pred
    {1: 0, 3: 1, 2: 3, 4: 2, 5: 3}
    >>> # Use 'Method'='sparse' => Kruskal, 'Type'='forest'
    >>> T2, pred2 = minspantree(G, 'Method','sparse','Type','forest')
    >>> list(T2.edges(data=True))
    [(1, 3, {'weight': 2}), (2, 3, {'weight': 1}), (2, 4, {'weight': 5}), (3, 5, {'weight': 7})]
    >>> pred2
    {1: 0, 3: 1, 2: 3, 4: 2, 5: 3}
    """

    # ----------------------------------------
    # 1) Default parameters
    # ----------------------------------------
    method = 'dense'  # 'dense' => Prim, 'sparse' => Kruskal
    root   = 1        # default root node
    tree_type = 'tree'  # 'tree' or 'forest'

    # ----------------------------------------
    # 2) Parse MATLAB-style name-value pairs
    # ----------------------------------------
    i = 0
    while i < len(args):
        if i+1 >= len(args):
            raise ValueError(f"Parameter '{args[i]}' is missing a value.")
        p_name = str(args[i]).lower()
        p_val  = args[i+1]
        i += 2
        if p_name == 'method':
            method = p_val
        elif p_name == 'root':
            root = p_val
        elif p_name == 'type':
            tree_type = p_val
        else:
            raise ValueError(f"Unrecognized parameter name: '{args[i]}'")

    # Also parse keyword arguments
    for key, val in kwargs.items():
        k_lower = key.lower()
        if k_lower == 'method':
            method = val
        elif k_lower == 'root':
            root = val
        elif k_lower == 'type':
            tree_type = val
        else:
            raise ValueError(f"Unrecognized parameter: '{key}'")

    # Validate method
    if method not in ('dense','sparse'):
        raise NotImplementedError("Only 'dense' (Prim) or 'sparse' (Kruskal) are supported.")

    # Validate type
    if tree_type not in ('tree','forest'):
        raise NotImplementedError("Only 'tree' or 'forest' are supported for 'Type'.")

    # Ensure G is a NetworkX Graph
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph (undirected) for this wrapper.")

    # If node names are strings in MATLAB, we handle them as well (just pass them directly).
    # NetworkX doesn't mind if root is a string or int as long as it is in G.

    # ----------------------------------------
    # 3) Build MST or forest using NetworkX
    #    - 'dense' => Prim's
    #    - 'sparse' => Kruskal's
    # ----------------------------------------
    algo = 'prim' if method=='dense' else 'kruskal'
    # NetworkX minimum_spanning_tree does it for all components => effectively a forest.
    # We'll get a spanning forest if the graph is disconnected.
    MST_all = nx.minimum_spanning_tree(G, weight='weight', algorithm=algo)

    # ----------------------------------------
    # 4) If user wants Type='tree', we keep only
    #    the component containing 'root'
    #    (like MATLAB's minspantree would do).
    # ----------------------------------------
    if tree_type == 'tree':
        # Extract the connected component containing root
        if root not in MST_all:
            # If root is not in MST, it's possible root not in G at all,
            # or is an isolated node with no edges in G
            raise ValueError(f"Specified root node '{root}' not found in the graph.")
        # We can find all nodes in that connected component
        connected_nodes = nx.node_connected_component(MST_all, root)
        # Subgraph containing only that component
        MST_sub = MST_all.subgraph(connected_nodes).copy()
        T = nx.Graph()
        T.add_nodes_from(MST_sub.nodes(data=True))
        T.add_edges_from(MST_sub.edges(data=True))
    else:
        # 'forest' => keep the entire MST with edges for all components
        T = MST_all.copy()

    # ----------------------------------------
    # 5) Build predecessor array, pred
    #    pred[node] = its parent in BFS from root.
    #    For 'tree' mode, any node not in that comp => pred[node] = NaN
    #    In 'forest' mode, we do a BFS from root, but also BFS from every
    #    other root in the MST if needed.
    # ----------------------------------------
    pred = dict()
    for node in T.nodes():
        pred[node] = np.nan  # initialize

    # If root in T, BFS from 'root' assigns parent.
    # But if 'forest', we also BFS from other components to get their parents.
    # We'll simulate MATLAB’s approach:
    #   - If 'tree': BFS from root only.
    #   - If 'forest': BFS from root, then BFS from an arbitrary node in each other component as well.

    def assign_pred_by_bfs(start_node):
        from collections import deque
        queue = deque([start_node])
        pred[start_node] = 0  # per MATLAB convention for the root
        while queue:
            curr = queue.popleft()
            for nbr in T.neighbors(curr):
                # If pred[nbr] is still nan => unvisited
                if np.isnan(pred[nbr]):
                    pred[nbr] = curr
                    queue.append(nbr)

    if root in T:
        assign_pred_by_bfs(root)

    if tree_type == 'forest':
        # There may be other components. For each connected component, pick any
        # node that is still np.nan and do BFS from it if no parent assigned.
        unvisited = [n for n in T.nodes() if np.isnan(pred[n])]
        while unvisited:
            start = unvisited[0]
            assign_pred_by_bfs(start)
            unvisited = [n for n in T.nodes() if np.isnan(pred[n])]

    # If tree_type='tree', then nodes not in the root's component remain np.nan

    return T, pred


def cophenet(Z, Y):
    """
    Compute the cophenetic correlation coefficient for a hierarchical cluster tree,
    along with (optionally) the cophenetic distances in condensed form, matching
    MATLAB's cophenet behavior.

    Parameters
    ----------
    Z : ndarray of shape (m-1, 4)
        The linkage matrix representing the hierarchical cluster tree. Typically
        the output of `scipy.cluster.hierarchy.linkage` (m observations => m-1 merges).
        In MATLAB terms, Z(:,3) holds the distance at each merge.

    Y : ndarray of shape (m*(m-1)//2,)
        The condensed distance vector used to generate Z (e.g., the output of
        `scipy.spatial.distance.pdist`). This is the same shape as MATLAB's Y from
        `pdist(X)`.

    Returns
    -------
    c : float
        Cophenetic correlation coefficient, measuring how faithfully the dendrogram
        preserves the pairwise distances in Y.
        A value close to 1 indicates a high-quality clustering solution.

    d : ndarray of shape (m*(m-1)//2,)
        The cophenetic distances in condensed form. d[i] is the cophenetic distance
        between the same pair of observations indexed by Y[i]. The cophenetic
        distance between two observations is the linkage distance at which they are
        first merged in the dendrogram.

    Notes
    -----
    - The formula for the cophenetic correlation coefficient c is:

      c = ( sum_{i<j} (Y_ij - y)(Z_ij - z) )
          -------------------------------------------------
          sqrt( sum_{i<j}(Y_ij - y)^2 * sum_{i<j}(Z_ij - z)^2 )

      where:
        * Y_ij is the original distance between observations i and j
        * Z_ij is the cophenetic (dendrogram) distance between i and j
        * y and z are the average of Y and Z (cophenetic distances), respectively.

    - In Python, SciPy’s `cophenet(Z, Y)` computes both c and d, so we simply wrap
      that function. If you only need c, you can ignore d:

        >>> c, _ = cophenet(Z, Y)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist
    >>> from scipy.cluster.hierarchy import linkage
    >>> # Generate sample data (e.g., three clusters in 3D space)
    >>> X = np.vstack([
    ...     np.random.randn(10,3),
    ...     np.random.randn(10,3) + 5,
    ...     np.random.randn(10,3) + 10
    ... ])
    >>> # Compute condensed distances and linkage
    >>> Y = pdist(X)  # shape (m*(m-1)//2,)
    >>> Z = linkage(Y, method='average')
    >>> # Now compute cophenet
    >>> c, d = cophenet(Z, Y)
    >>> print("Cophenetic correlation coefficient:", c)
    >>> print("Cophenetic distances shape:", d.shape)
    """
    c, d = scipy_cophenet(Z, Y)
    return c, d


def silhouette(X, clust, Distance='euclidean', DistParameter=None,
               do_plot=True):
    """
    Silhouette plot and values, mimicking MATLAB's silhouette function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix. Rows correspond to observations, columns
        correspond to features (variables).
    clust : array-like of shape (n_samples,)
        Cluster labels (integers) of each observation in X.
    Distance : str or callable, optional (default='euclidean')
        Distance metric to use. Must be recognized by scikit-learn’s
        silhouette_samples, e.g., 'euclidean', 'manhattan', 'cosine',
        'precomputed'. If given a function handle, we raise
        NotImplementedError in this basic wrapper.
    DistParameter : dict or other, optional
        Additional distance-metric parameter(s). For example, if
        Distance='minkowski', you can pass DistParameter={'p': 3} to use
        Minkowski exponent p=3. If using an unsupported or custom function,
        we raise an error.
    do_plot : bool, optional (default=True)
        Whether to create the silhouette plot. If False, only the silhouette
        values are computed and returned.

    Returns
    -------
    s : ndarray of shape (n_samples,)
        The silhouette value for each observation. Values range from -1 to 1,
        where higher values indicate that the observation is well matched
        to its cluster.
    h : matplotlib.figure.Figure or None
        The figure handle for the silhouette plot if `do_plot=True`.
        If `do_plot=False`, we return None.

    Notes
    -----
    1) This function is a wrapper around sklearn.metrics.silhouette_samples.
    2) If you specify a custom distance function handle, we raise
       NotImplementedError.
    3) For specialized distance metrics (like 'seuclidean' or 'mahalanobis'),
       you may need to manually compute a distance matrix and pass
       Distance='precomputed'.

    Examples
    --------
    >>> import numpy as np
    >>> from pyEDAkit.clustering import silhouette
    >>> X = np.random.rand(10, 2)
    >>> clust = np.array([0,0,0,0,1,1,1,1,0,1])  # Some cluster labels
    >>> # 1) Basic call with default distance=euclidean, and show the plot
    >>> s, fig = silhouette(X, clust)
    >>> s
    array([ 0.62,  0.68,  0.58, ...,  0.55])  # silhouette values
    >>> # 2) Minkowski distance with exponent 3 (DistParameter)
    >>> s2, fig2 = silhouette(X, clust, Distance='minkowski', DistParameter={'p':3})
    >>> # 3) If you only want silhouette values (no plot):
    >>> s3, _ = silhouette(X, clust, do_plot=False)
    """

    # --- 1) Handle custom function distances ---
    if callable(Distance):
        raise NotImplementedError("Custom distance function handles are not supported.")

    # --- 2) Prepare metric kwargs for silhouette_samples. ---
    metric_kwargs = {}
    if isinstance(DistParameter, dict):
        metric_kwargs.update(DistParameter)

    # scikit-learn silhouette_samples can handle these metrics natively:
    #   'euclidean', 'manhattan', 'cosine', 'precomputed', etc.
    # Or 'minkowski' with a dict DistParameter like {'p': 3}.

    # --- 3) Compute silhouette samples. This returns a silhouette value per sample. ---
    s = silhouette_samples(X, clust, metric=Distance, **metric_kwargs)

    # If do_plot=False, we skip the figure creation and return (s, None)
    if not do_plot:
        return s, None

    # --- 4) Create the silhouette plot in a style similar to MATLAB. ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # For each cluster, we extract the silhouette scores and plot them in ascending order.
    labels = np.unique(clust)
    n_clusters = len(labels)

    # We'll keep track of the y-limits for the bar boundaries.
    y_lower = 0
    for i, label in enumerate(labels):
        # Extract silhouette scores for samples in cluster i
        ith_cluster_sil_values = s[clust == label]
        ith_cluster_sil_values.sort()
        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.barh(range(y_lower, y_upper),
                ith_cluster_sil_values,
                height=1.0,
                edgecolor='none',
                color=color)
        # Label the silhouette plots with their cluster numbers in the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))

        # Compute next y_lower for next cluster
        y_lower = y_upper

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    # The vertical line for average silhouette score:
    avg_score = np.mean(s)
    ax.axvline(x=avg_score, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the y-axis labels / ticks

    ax.set_xlim([min(s) - 0.1, 1.0])
    ax.set_ylim([0, len(X)])
    ax.set_title(f"Silhouette plot for {n_clusters} clusters")

    plt.tight_layout()
    return s, fig


class SilhouetteEvaluation:
    """
    A Python class that mimics MATLAB's SilhouetteEvaluation object.
    It evaluates various cluster solutions (for different k) via the silhouette criterion,
    and identifies the optimal number of clusters based on the silhouette measure.

    Properties (mimicking MATLAB):
    -----------------------------
    ClusteringFunction : str or callable
        The clustering algorithm used ('kmeans', 'linkage', etc.).
        For this example, we focus on 'kmeans'.

    ClusterPriors : {'empirical','equal'}
        - 'empirical': Weighted average of silhouette scores based on cluster sizes.
        - 'equal': Unweighted average of silhouette scores across clusters.

    ClusterSilhouettes : list of 1D numpy arrays
        Each element i in this list is an array of length == number of clusters for KList[i].
        That array holds the mean silhouette value of each cluster.

    CriterionName : str
        The name of the criterion, always 'Silhouette' for this object.

    CriterionValues : 1D numpy array
        The silhouette criterion values (one for each k in InspectedK).
        This is the overall measure of cluster quality for that number of clusters.

    Distance : str
        The distance metric used, e.g., 'sqEuclidean', 'Euclidean', 'cityblock', etc.
        (Implementation focuses primarily on 'sqEuclidean' or 'euclidean'.)

    InspectedK : 1D numpy array
        The list of candidate cluster counts that were evaluated.

    OptimalK : int
        The best number of clusters based on the maximum silhouette criterion value.

    OptimalY : 1D numpy array or None
        The cluster assignments for the best clustering solution
        (same length as the number of rows in X).
        If there is missing data in X, the corresponding row in OptimalY is NaN.

    Missing : 1D boolean array or None
        A mask of which rows were ignored (if they contained NaNs).
        If no missing data, this can be None or an all-False array.

    NumObservations : int
        The number of valid observations (rows) used from X after ignoring NaNs.

    X : 2D numpy array or None
        The original data (rows with NaNs removed). If you want a “compact” object,
        you could set this to None afterward.

    Example
    -------
    # Suppose you have data X with 600 rows (some from different distributions).
    # Evaluate k=1..6 using kmeans and silhouette:
    evalObj = SilhouetteEvaluation(X,
                                  clusteringFunction='kmeans',
                                  KList=[1,2,3,4,5,6],
                                  Distance='sqEuclidean',
                                  ClusterPriors='empirical')
    print(evalObj.OptimalK)   # best number of clusters
    print(evalObj.CriterionValues)  # silhouette scores for each k
    print(evalObj.OptimalY)   # best cluster assignments
    """

    def __init__(self,
                 X,
                 clusteringFunction='kmeans',
                 KList=None,
                 Distance='sqEuclidean',
                 ClusterPriors='empirical'):
        """
        Constructor for SilhouetteEvaluation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to cluster. Rows with NaNs are ignored.
        clusteringFunction : str
            The clustering method. For this example, we support only 'kmeans'.
        KList : list or array-like
            A list of candidate cluster numbers to try. E.g., [1,2,3,4,5,6].
        Distance : str
            Distance metric, e.g. 'sqEuclidean', 'euclidean', 'cityblock'.
        ClusterPriors : {'empirical','equal'}
            - 'empirical': Weighted average by cluster size
            - 'equal': Each cluster contributes equally
        """
        print("The execution of this method might take several seconds. Please wait...")
        if KList is None:
            KList = [2, 3, 4, 5, 6]  # default
        self.ClusteringFunction = clusteringFunction
        self.ClusterPriors = ClusterPriors
        self.CriterionName = 'Silhouette'
        self.Distance = Distance
        self.InspectedK = np.asarray(KList, dtype=int)

        # Handle missing data
        X = np.asarray(X, dtype=float)
        nan_mask = np.isnan(X).any(axis=1)
        self.Missing = None
        if np.any(nan_mask):
            self.Missing = nan_mask
            X_valid = X[~nan_mask]
        else:
            X_valid = X

        self.X = X  # store the entire raw data if desired
        self.NumObservations = X_valid.shape[0]

        # Prepare placeholders
        self.CriterionValues = np.full(len(KList), np.nan, dtype=float)
        self.ClusterSilhouettes = [None] * len(KList)
        self.OptimalK = None
        self.OptimalY = None  # best cluster labels

        # Evaluate silhouette for each k in KList
        self._evaluate_solutions(X_valid)

        # Identify the best k
        max_index = np.nanargmax(self.CriterionValues)
        self.OptimalK = int(self.InspectedK[max_index])

        # Compute the cluster assignment for the best k on the full data
        # (including missing => those remain NaN in the assignment).
        bestK = self.OptimalK
        if self.ClusteringFunction.lower() == 'kmeans':
            # We use our kmeans wrapper
            idx_full, _, _, _ = kmeans_matlab(X, bestK,
                                       'Distance', self.Distance,
                                       'Replicates', 5,
                                       'EmptyAction', 'singleton')
            self.OptimalY = idx_full  # shape (n_samples,)

        else:
            raise NotImplementedError(f"Only 'kmeans' clusteringFunction is currently implemented.")

    def _evaluate_solutions(self, X_valid):
        """
        For each candidate k, run clustering and compute silhouette.
        Fill out CriterionValues and ClusterSilhouettes.
        """
        for i, k in enumerate(self.InspectedK):
            # Silhouette is undefined (and scikit-learn errors out) if k < 2
            if k < 2:
                self.CriterionValues[i] = np.nan
                # We can store an array of [np.nan] for the cluster silhouette means
                self.ClusterSilhouettes[i] = np.array([np.nan])
                continue

            # If we have k>=2, proceed:
            if self.ClusteringFunction.lower() == 'kmeans':
                # cluster the valid data
                idx_valid, _, _, _ = kmeans_matlab(X_valid, k,
                                            'Distance', self.Distance,
                                            'Replicates', 5,
                                            'EmptyAction', 'singleton',
                                            'MaxIter', 100,
                                            'Display', 'off')
                # silhouette
                s, _ = silhouette(X_valid, idx_valid,
                                  Distance=self._map_distance(self.Distance),
                                  do_plot=False)

                # compute cluster-level means
                cluster_labels = np.unique(idx_valid[~np.isnan(idx_valid)])
                cluster_labels = cluster_labels.astype(int)

                cluster_means = []
                cluster_sizes = []
                for lbl in cluster_labels:
                    s_lbl = s[idx_valid == lbl]
                    cluster_means.append(np.mean(s_lbl))
                    cluster_sizes.append(len(s_lbl))

                cluster_means = np.array(cluster_means)
                cluster_sizes = np.array(cluster_sizes)

                self.ClusterSilhouettes[i] = cluster_means

                # overall silhouette => depends on ClusterPriors
                if self.ClusterPriors.lower() == 'empirical':
                    # Weighted by cluster size
                    w = cluster_sizes / cluster_sizes.sum()
                    overall = np.sum(cluster_means * w)
                elif self.ClusterPriors.lower() == 'equal':
                    # Unweighted average across clusters
                    overall = np.mean(cluster_means)
                else:
                    raise ValueError("ClusterPriors must be 'empirical' or 'equal'.")

                self.CriterionValues[i] = overall

            else:
                raise NotImplementedError(f"Only 'kmeans' is implemented in _evaluate_solutions().")

    def _map_distance(self, dist_name):
        """
        Map MATLAB-style distance names to something scikit-learn understands.
        E.g., 'sqEuclidean' -> 'euclidean'.
        """
        dn = dist_name.lower()
        if dn == 'sqeuclidean':
            return 'euclidean'
        # you can add more mappings here as needed
        return dn

    def plot(self):
        """
        Plot the silhouette criterion values vs. the inspected K.
        Similar to MATLAB's plot(evaluation).
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(self.InspectedK, self.CriterionValues, 'bo-', mfc='white')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Values")
        plt.title("Silhouette Criterion Evaluation")
        plt.grid(True)
        plt.show()
