from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
def KNeighbors(fromxl,feat,n_neighbors):
    nei = KNeighborsClassifier(n_neighbors=n_neighbors)
    nei.fit(fromxl, feat)
    return nei

def RForest(fromxl,feat,n_estimators):
    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest.fit(fromxl, feat)
    return forest