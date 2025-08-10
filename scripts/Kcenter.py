import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
import time
import subprocess

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def generate_fmt(num_dimensions):
    """
    根据数据维度动态生成保存格式字符串。

    Parameters:
    - num_dimensions (int): 数据的维度。

    Returns:
    - fmt (str): 格式化字符串，例如 '%.6f,%.6f,...'
    """
    return ",".join(["%.6f"] * num_dimensions)

def graph_partition_clustering(data, num_clusters):
    num_points = len(data)
    k = 100  # Number of neighbors, adjust as necessary

    start_total = time.time()
    knn_graph = kneighbors_graph(data, k, mode='connectivity', include_self=False)
    adj_matrix = csr_matrix(knn_graph)
    xadj = np.zeros(num_points + 1, dtype=int)
    adjncy = []

    for i in range(num_points):
        neighbors = adj_matrix[i].indices
        xadj[i + 1] = xadj[i] + len(neighbors)
        adjncy.extend(neighbors)

    graph_filename = "graph.txt"
    with open(graph_filename, "w") as f:
        f.write(f"{num_points} {len(adjncy) // 2}\n")
        for i in range(num_points):
            neighbors = [x + 1 for x in adjncy[xadj[i]:xadj[i+1]]]
            f.write(" ".join(map(str, neighbors)) + "\n")

    parhip_command = ["parhip", graph_filename, "--k", str(num_clusters), "--preconfiguration=ultrafastsocial", "--save_partition"]
    result = subprocess.run(parhip_command, capture_output=True, text=True)

    partition_filename = "tmppartition.txtp"
    try:
        with open(partition_filename, "r") as f:
            blocks = np.array([int(line.strip()) for line in f])
    except FileNotFoundError:
        print(f"Error: Partition file {partition_filename} not found.")
        blocks = np.zeros(num_points)

    end_total = time.time()
    print(f"Total execution time: {end_total - start_total:.4f} seconds")
    return blocks, adj_matrix

def visualize_clusters_3d(data, blocks, num_clusters):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_clusters))
    for i in range(num_clusters):
        points = data[blocks == i]
        ax.scatter(points[:, 1], points[:, 0], points[:, 2], s=30, alpha=0.5, color=colors[i], label=f'Cluster {i}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.title("3D Graph Partition Clustering")
    plt.legend()
    plt.show()

def select_pivots_for_cluster(cluster_data, num_pivots=2):
    if len(cluster_data) < num_pivots:
        return cluster_data, list(range(len(cluster_data)))

    pivots = []
    pivot_indices = []

    first_pivot_index = np.random.randint(len(cluster_data))
    pivots.append(cluster_data[first_pivot_index])
    pivot_indices.append(first_pivot_index)

    for _ in range(1, num_pivots):
        distances = euclidean_distances(cluster_data, np.array(pivots))
        max_dist = np.min(distances, axis=1)
        next_pivot_index = np.argmax(max_dist)
        pivots.append(cluster_data[next_pivot_index])
        pivot_indices.append(next_pivot_index)

    return np.array(pivots), pivot_indices

def gen_eqdata(data_folder, output_folder, num_clusters):
    data_file = os.path.join(data_folder, "skewed_dataset.csv")
    print("Loading data...")
    df = pd.read_csv(data_file)
    data = df.iloc[:, :4].values.astype(float)  # 读取前16列

    if len(data) > 1000000:
        data = data[:1000000]

    print("Normalizing data...")
    normalized_data = normalize_data(data)

    print("Performing graph partition clustering...")
    blocks, adj_matrix = graph_partition_clustering(normalized_data, num_clusters)

    print("Visualizing clustering results...")
    visualize_clusters_3d(normalized_data, blocks, num_clusters)

    for i in range(num_clusters):
        cluster_data = normalized_data[blocks == i]
        cluster_data_with_id = np.hstack((np.arange(len(cluster_data)).reshape(-1, 1), cluster_data))
        npy_file_path = os.path.join(output_folder, f'cluster_{i}.npy')
        txt_file_path = os.path.join(output_folder, f'cluster_{i}.txt')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        np.save(npy_file_path, cluster_data_with_id)
        np.savetxt(txt_file_path, cluster_data_with_id, fmt='%i,' + generate_fmt(cluster_data.shape[1]), delimiter=',')
        print(f"Cluster {i} saved in {npy_file_path} and {txt_file_path}")

        pivots, pivot_indices = select_pivots_for_cluster(cluster_data, num_pivots=2)
        if len(pivots) > 0:
            ref_file_path = os.path.join(output_folder, f'pivot_{i}.txt')
            np.savetxt(ref_file_path, pivots, fmt=generate_fmt(cluster_data.shape[1]), delimiter=',')
            print(f"Pivot points for cluster {i} saved in {ref_file_path}")

    main_pivots = []
    for i in range(num_clusters):
        cluster_data = normalized_data[blocks == i]
        cluster_indices = np.where(blocks == i)[0]
        degrees = np.array([adj_matrix[idx].count_nonzero() for idx in cluster_indices])
        main_pivot_index = cluster_indices[np.argmax(degrees)]
        main_pivot = normalized_data[main_pivot_index]
        main_pivots.append(main_pivot)

    ref_file_path = os.path.join(output_folder, 'ref.txt')
    np.savetxt(ref_file_path, np.array(main_pivots), fmt=generate_fmt(normalized_data.shape[1]), delimiter=',')
    print(f"Main pivots saved in {ref_file_path}")
