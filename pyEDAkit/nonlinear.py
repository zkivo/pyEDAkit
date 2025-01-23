from sklearn.manifold import MDS as skMDS


def MDS(X, d, metric=True):
    mds = skMDS(n_components=d, random_state=42, metric=metric)
    return mds.fit_transform(X)