import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def hac_single_linkage(D):
    n = D.shape[0]
    clusters = {i:[i] for i in range(n)}
    active = list(range(n))
    Z = []

    while len(active) > 1:
        i_min, j_min = -1, -1
        min_dist = np.inf
        for i in range(len(active)):
            for j in range(i+1, len(active)):
                a, b = active[i], active[j]
                if D[a, b] < min_dist:
                    min_dist, i_min, j_min = D[a, b], a, b

        new_cluster = max(clusters.keys()) + 1
        Z.append([i_min, j_min, min_dist, len(clusters[i_min]) + len(clusters[j_min])])
        clusters[new_cluster] = clusters[i_min] + clusters[j_min]
        active = [c for c in active if c not in (i_min, j_min)] + [new_cluster]

        new_row = np.zeros((D.shape[0]+1,))
        new_col = np.zeros((D.shape[0]+1,1))
        D = np.pad(D, ((0,1),(0,1)), constant_values=np.inf)
        for c in active:
            if c != new_cluster:
                dist = min(D[c, i_min], D[c, j_min])
                D[c, new_cluster] = dist
                D[new_cluster, c] = dist

    return np.array(Z)

# مصفوفة المسافات (مثال من التاسك)
D = np.array([[0,3,4,7],
              [3,0,4,6],
              [4,4,0,5],
              [7,6,5,0]], dtype=float)

Z = hac_single_linkage(D.copy())
print("Linkage Matrix (Manual Implementation):\n", Z)

plt.figure()
dendrogram(Z)
plt.title('Manual HAC - Single Linkage')
plt.show()

Z2 = linkage(D, method='single')
plt.figure()
dendrogram(Z2)
plt.title('SciPy HAC - Single Linkage')
plt.show()
