from sklearn.cluster import KMeans
import numpy as np

def bitMeans(block_salience):

    saliency = np.array(block_salience).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(saliency)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_cluster_indices = np.argsort(cluster_centers)
    new_labels = np.zeros_like(cluster_labels)
    for i, cluster_index in enumerate(sorted_cluster_indices):
        new_labels[cluster_labels == cluster_index] = i

    unique, counts = np.unique(new_labels, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    adjusted_labels = new_labels.copy()

    min_count = counts_dict[2]

    indices_label_0 = np.where(new_labels == 0)[0]
    indices_label_0 = indices_label_0.tolist()
    values_label_0 = [block_salience[i] for i in indices_label_0]
    sorted_indices_values = sorted(zip(indices_label_0, values_label_0), key=lambda x: x[1])
    # 计算需要保留标签0的数量（即与标签2的数量相等）
    num_to_keep = min_count
    # 分离出将要改变标签的索引（即值较大的部分）
    indices_to_change = [idx for idx, val in sorted_indices_values[num_to_keep:]]
    # 将这部分数据点的标签改为1
    for idx in indices_to_change:
        adjusted_labels[idx] = 1
    return adjusted_labels