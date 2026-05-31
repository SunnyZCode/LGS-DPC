import time
import numpy as np
from munkres import Munkres
from scipy.spatial import cKDTree
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def natural_search(data):
    n = data.shape[0]
    k_num = np.sqrt(n).astype(int)

    tree = cKDTree(data)
    dists, index = tree.query(data, k=k_num)

    nb = np.zeros(n, dtype=int)
    t = 1
    num_1 = 0
    num_2 = 0
    all_neighbors = [[] for _ in range(n)]
    while t < n:
        y = index[:, t]
        for i in range(n):
            all_neighbors[i].append(y[i])
        nb += np.bincount(y, minlength=n)
        num_1 = np.count_nonzero(nb == 0)
        if num_1 == num_2:
            break
        num_2 = num_1
        t += 1
    Point = np.full(n, -1, dtype=int)
    gb_list = []
    for i in range(n):
        if Point[i] == -1:
            all_indices = [i] + all_neighbors[i]
            all_indices = list(set(all_indices))
            group_coords = data[all_indices]
            gb_list.append(group_coords)
            Point[all_indices] = 1

    return gb_list


def calculate_center_and_radius(gb):
    data_no_label = gb[:, :]  # 取坐标
    center = data_no_label.mean(axis=0)  # 压缩行，对列取均值  取出平均的 x,y
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))  # （x1-x1）**2 + (y1-y2)**2   所有点到中心的距离平均
    return center, radius


def calculate_radius(gb):
    data_no_label = gb[:, :]
    center = data_no_label.mean(axis=0) 
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return radius


def get_DM(hb):
    num = len(hb)
    if num <= 2:
        return 20
    center = np.mean(hb, axis=0)
    distances = np.linalg.norm(hb - center, axis=1)
    return np.mean(distances)


def spilt_ball(data):
    dist_matrix = squareform(pdist(data, metric='euclidean'))
    r, c = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    ball1 = []
    ball2 = []
    for i in range(len(data)):
        if dist_matrix[i, r] < dist_matrix[i, c]:
            ball1.append(data[i])
        else:
            ball2.append(data[i])
    return np.array(ball1), np.array(ball2)


def normalized_ball(hb_list, radius_detect):
    hb_list_temp = []
    for hb in hb_list:
        if len(hb) < 2:
            hb_list_temp.append(hb)
        else:
            if calculate_radius(hb) <= 2 * radius_detect:
                hb_list_temp.append(hb)
            else:
                ball_1, ball_2 = spilt_ball(hb)
                if len(ball_1) > 0 and len(ball_2) > 0:
                    hb_list_temp.extend([ball_1, ball_2])
                else:
                    hb_list_temp.append(hb)
    return hb_list_temp


def get_ball_quality(gb):
    N = gb.shape[0]
    ball_quality = get_DM(gb)  # 用DM代表质量
    return ball_quality, N


def ball_density(radiusAD, ball_qualitysA, ball_mA, centersA, k):
    N = radiusAD.shape[0]
    ball_dens = np.where(ball_mA >= 2, ball_qualitysA, 0)
    dist_matrix = distance.cdist(centersA, centersA, 'euclidean')
    densities = np.zeros(N)
    for i in range(N):
        nearest_neighbors = np.argpartition(dist_matrix[i], k)[:k + 1]  
        neighbor_density_sum = np.sum(ball_dens[nearest_neighbors])
        densities[i] = np.exp(-(neighbor_density_sum / (k + 1)))
    return densities


def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    ball_min_distAD = np.zeros(shape=N3)
    ball_nearestAD = np.zeros(shape=N3)
    index_ball_dens = np.argsort(-ball_densS)
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index, j] for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j] for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD


def ball_draw_decision(ball_densS, ball_min_distS):
    centers = []
    rho = ball_densS * ball_min_distS
    indices = np.argsort(-rho)[:k]
    for i in range(ball_densS.shape[0]):
        if i in indices:
            centers.append(i)
    return np.array(centers)


def ball_cluster(ball_densS, ball_centers, ball_nearest):
    K1 = len(ball_centers)
    if K1 == 0:
        print('no centers')
        return
    N5 = ball_densS.shape[0]
    ball_labs = -1 * np.ones(N5).astype(int)
    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5 + 1)
    ball_index_density = np.argsort(-ball_densS)
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]
    return ball_labs


def update_point_labels(data, ball_labs, gb_list):
    labels = -np.ones(data.shape[0], dtype=int)
    gb_dict = {}
    for i6 in range(len(gb_list)):
        for j6, point in enumerate(gb_list[i6]):
            gb_dict[tuple(point)] = ball_labs[i6]
    for i, data1 in enumerate(data):
        if tuple(data1) in gb_dict and labels[i] == -1:
            labels[i] = gb_dict[tuple(data1)]
    return labels


def evaluation(y_true, y_pred):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')
    return acc, nmi, ari, f1


def fit_sphere(points):
    A = np.hstack((2 * points, np.ones((points.shape[0], 1))))
    f = (points ** 2).sum(axis=1)
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:-1]
    radius = np.sqrt((center ** 2).sum() + C[-1])
    return center, radius


def sphere_arc_length(p1, p2, center, radius):
    v1 = p1 - center
    v2 = p2 - center
    theta = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return radius * theta


def build_spherelet_graph(X, k):
    N = X.shape[0]
    dist_mat = distance.cdist(X, X)
    neighbors = np.argsort(dist_mat, axis=1)[:, 1:k + 1]
    G = np.full((N, N), np.inf)
    for i in range(N):
        local_idx = np.append(i, neighbors[i])
        local_points = X[local_idx]
        center, radius = fit_sphere(local_points)
        for j in neighbors[i]:
            arc = sphere_arc_length(X[i], X[j], center, radius)
            G[i, j] = arc
            G[j, i] = arc  
    return G


def estimate_spherelet_geodesic(X, k):
    from scipy.sparse.csgraph import dijkstra
    G = build_spherelet_graph(X, k)
    geodesic_distances = dijkstra(G, indices=None)
    return geodesic_distances



if __name__ == "__main__":
  
    Realt_all = []

    df = np.loadtxt(f'LGS-DPC\Seed.txt')
    data = df[:, :-1]
    data_label = df[:, -1]

    k_n = 4

    k = len(set(data_label))
   
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    start = time.time()

    gb_list = natural_search(data)
    radius = []
    for hb in gb_list:
        radius.append(calculate_radius(hb))
    radius_detect = np.mean(radius)
    while 1:
        ball_number_old = len(gb_list)
        gb_list = normalized_ball(gb_list, radius_detect)
        ball_number_new = len(gb_list)
        if ball_number_new == ball_number_old:
            break

    centers = []
    radiuss = []
    ball_qualitys = []
    ball_m = []
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)  
        ball_quality, m = get_ball_quality(gb)  
        ball_qualitys.append(ball_quality) 
        ball_m.append(m)
        centers.append(center)
        radiuss.append(radius)
    centersA = np.array(centers)
    radiusA = np.array(radiuss)
    ball_qualitysA = np.array(ball_qualitys)
    ball_mA = np.array(ball_m)

    ball_densS = ball_density(radiusA, ball_qualitysA, ball_mA, centersA, k_n)
    ball_distS = estimate_spherelet_geodesic(centersA, k_n)
    ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)
    ball_centers = ball_draw_decision(ball_densS, ball_min_distS)
    ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest)

    end = time.time()

    label = update_point_labels(data, ball_labs, gb_list)
    ACC, NMI, ARI, F1 = evaluation(data_label, label)
    print(f'ACC: {ACC:.3f}, NMI: {NMI:.3f}, ARI: {ARI:.3f}, F1: {F1:.3f}, Time: {end - start:.3f} s')

     

