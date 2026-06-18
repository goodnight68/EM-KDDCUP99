import os
import sys
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# PyInstaller 打包后路径适配
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(__file__)

KDD_DIR = os.path.join(BASE_DIR, 'data')
DATA_GZ = os.path.join(KDD_DIR, 'kddcup.data_10_percent.gz')


def load_kddcup99(n_samples=6000, pca_dim=8, random_state=42):
    """加载 KDDCUP99 数据集，返回 PCA 降维后的特征"""
    if not os.path.exists(DATA_GZ):
        raise FileNotFoundError(f"数据文件不存在: {DATA_GZ}")

    with gzip.open(DATA_GZ, 'rt') as f:
        lines = f.readlines()

    rng = np.random.RandomState(random_state)
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) > n_samples * 3:
        # 随机采样，避免加载全部数据
        indices = rng.choice(len(lines), n_samples * 3, replace=False)
        lines = [lines[i] for i in indices]

    data_list = []
    labels = []
    for line in lines:
        parts = line.split(',')
        if len(parts) < 42:
            continue
        # 跳过 protocol_type、service、flag 三个分类列
        numeric_idx = [0] + list(range(4, 41))
        try:
            row = [float(parts[i]) for i in numeric_idx]
        except (ValueError, IndexError):
            continue
        data_list.append(row)
        labels.append(parts[41].strip()[:-1] if parts[41].endswith('.') else parts[41].strip())

    X_raw = np.array(data_list, dtype=np.float64)
    y_raw = np.array(labels)

    # 将具体攻击类型映射为五大类：normal, dos, probe, r2l, u2r
    label_map = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
        'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
        'warezclient': 'r2l', 'warezmaster': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    }
    y_mapped = np.array([label_map.get(l, 'unknown') for l in y_raw])

    # 只选取 normal, dos, probe 三类
    use_cats = ['normal', 'dos', 'probe']
    mask = np.isin(y_mapped, use_cats)
    X_raw = X_raw[mask]
    y_mapped = y_mapped[mask]

    # 分层采样，保持各类别数量均衡
    rng = np.random.RandomState(random_state)
    cat_indices = {}
    for cat in use_cats:
        cat_indices[cat] = np.where(y_mapped == cat)[0]

    n_probe = len(cat_indices['probe'])
    n_other = (n_samples - n_probe) // 2
    n_other = min(n_other, len(cat_indices['normal']), len(cat_indices['dos']))

    indices_list = []
    indices_list.extend(rng.choice(cat_indices['normal'], n_other, replace=False))
    indices_list.extend(rng.choice(cat_indices['dos'], n_other, replace=False))
    indices_list.extend(rng.choice(cat_indices['probe'], min(n_probe, n_other), replace=False))
    idx = np.array(indices_list)
    X = X_raw[idx]
    y_final = y_mapped[idx]

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_final)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 降维
    pca_dim = min(pca_dim, X.shape[1])
    pca = PCA(n_components=pca_dim, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    # 单独的 2D PCA 用于可视化
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_2d = pca_2d.fit_transform(X_scaled)

    info = {
        'total_samples': len(X_reduced),
        'original_features': X.shape[1],
        'pca_dim': pca_dim,
        'classes': list(le.classes_),
        'n_classes': len(le.classes_),
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'class_distribution': dict(zip(le.classes_, np.bincount(y).tolist()))
    }

    return X_reduced, X_2d, y, info


if __name__ == '__main__':
    for n in [3000, 6000, 9000]:
        X, X2d, y, info = load_kddcup99(n_samples=n, pca_dim=8)
        print(f"样本: {info['total_samples']}, "
              f"特征: {info['original_features']} -> PCA {info['pca_dim']}")
        print(f"类别: {info['classes']}")
        print(f"分布: {info['class_distribution']}")
