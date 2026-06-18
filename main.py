import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture as SklearnGMM
import warnings
warnings.filterwarnings('ignore')

from em_gmm import EMGMM


def generate_complex_dataset(n_samples=1200, random_state=42):
    """生成包含 4 个高斯簇的合成数据集，模拟客户细分场景"""
    rng = np.random.RandomState(random_state)
    means = np.array([
        [2.0, 3.0],
        [8.0, 7.0],
        [0.0, 8.0],
        [6.0, 1.0],
    ])
    covs = [
        [[0.8, 0.3], [0.3, 0.8]],
        [[1.5, -0.5], [-0.5, 1.0]],
        [[0.6, 0.0], [0.0, 0.6]],
        [[1.2, 0.8], [0.8, 1.2]],
    ]
    proportions = [0.35, 0.28, 0.20, 0.17]  # 样本不平衡
    X_list, y_list = [], []
    for k in range(4):
        n_k = int(n_samples * proportions[k])
        X_k = rng.multivariate_normal(means[k], covs[k], size=n_k)
        X_list.append(X_k)
        y_list.append(np.full(n_k, k))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def preprocess_data(X):
    """Z-score 标准化"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def align_labels(y_true, y_pred):
    """匈牙利算法映射聚类标签到真实标签"""
    from scipy.optimize import linear_sum_assignment
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[p] for p in y_pred])


def train_and_evaluate(X, y, n_components=4):
    """训练自实现 EM 和 sklearn GMM，返回预测标签"""
    # 自实现 EM-GMM
    em_gmm = EMGMM(n_components=n_components, max_iter=150, tol=1e-4, random_state=42)
    em_gmm.fit(X)
    y_pred_ours = em_gmm.predict(X)

    # sklearn GMM 对照
    sk_gmm = SklearnGMM(n_components=n_components, covariance_type='full',
                        max_iter=150, tol=1e-4, random_state=42)
    sk_gmm.fit(X)
    y_pred_sk = sk_gmm.predict(X)

    y_pred_ours = align_labels(y, y_pred_ours)
    y_pred_sk = align_labels(y, y_pred_sk)
    return em_gmm, sk_gmm, y_pred_ours, y_pred_sk


def evaluate_clustering(y_true, y_pred):
    """计算聚类评估指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'ari': adjusted_rand_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred),
    }


def draw_ellipse(ax, mean, cov, color, alpha=0.15, lw=2, linestyle='-'):
    """绘制协方差椭圆（1σ 和 2σ）"""
    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)
    for std in [1, 2]:
        ell = Ellipse(xy=mean, width=std * width, height=std * height,
                      angle=angle, edgecolor=color, facecolor=color,
                      alpha=alpha * (1 if std == 1 else 0.3),
                      linewidth=lw if std == 1 else 1,
                      linestyle=linestyle)
        ax.add_patch(ell)


def plot_all_results(X_raw, y_true, em_gmm, sk_gmm, y_pred_ours, y_pred_sk,
                     metrics_ours, metrics_sk):
    """生成全部可视化图表并保存到 figures 目录"""
    import os
    os.makedirs('figures', exist_ok=True)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # 图1：数据真实分布
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(4):
        mask = y_true == k
        ax.scatter(X_raw[mask, 0], X_raw[mask, 1], c=colors[k], s=30, alpha=0.7,
                   label=f'簇 {k+1}', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('特征1', fontsize=12)
    ax.set_ylabel('特征2', fontsize=12)
    ax.set_title('真实标签分布', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig1_true_labels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图2：EM 收敛曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(em_gmm.history_) + 1), em_gmm.history_, 'b-', linewidth=2)
    ax.axvline(x=em_gmm.n_iter_, color='r', linestyle='--',
               label=f'收敛点 (迭代{em_gmm.n_iter_})')
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('对数似然', fontsize=12)
    ax.set_title('EM 算法收敛曲线', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig2_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图3：聚类结果对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for k in range(4):
        mask = y_true == k
        axes[0].scatter(X_raw[mask, 0], X_raw[mask, 1], c=colors[k],
                        s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[0].set_title('真实标签', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    for k in range(4):
        mask = y_pred_ours == k
        axes[1].scatter(X_raw[mask, 0], X_raw[mask, 1], c=colors[k],
                        s=30, alpha=0.7, edgecolors='white', linewidth=0.5,
                        label=f'簇 {k+1}')
    axes[1].set_title(f'自实现 EM (ACC={metrics_ours["accuracy"]:.3f})',
                      fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig3_clustering_result.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图4：协方差椭圆对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, title, m_obj in [
        (axes[0], y_pred_ours, '自实现 EM-GMM', em_gmm),
        (axes[1], y_pred_sk, 'sklearn GMM', sk_gmm)
    ]:
        for k in range(4):
            mask = y_pred == k
            ax.scatter(X_raw[mask, 0], X_raw[mask, 1], c=colors[k],
                       s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
        if m_obj is not None:
            for k in range(4):
                draw_ellipse(ax, m_obj.means_[k], m_obj.covariances_[k], colors[k], alpha=0.1)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig4_comparison_ellipses.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图5：混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, title in [
        (axes[0], y_pred_ours, '自实现 EM-GMM'),
        (axes[1], y_pred_sk, 'sklearn GMM')
    ]:
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig('figures/fig5_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图6：指标对比柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ['准确率', '精确率', '召回率', 'F1', 'ARI', 'NMI']
    ours_values = [metrics_ours['accuracy'], metrics_ours['precision'],
                   metrics_ours['recall'], metrics_ours['f1'],
                   metrics_ours['ari'], metrics_ours['nmi']]
    sk_values = [metrics_sk['accuracy'], metrics_sk['precision'],
                 metrics_sk['recall'], metrics_sk['f1'],
                 metrics_sk['ari'], metrics_sk['nmi']]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, ours_values, width, label='自实现 EM-GMM',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width / 2, sk_values, width, label='sklearn GMM',
                   color='#e74c3c', alpha=0.8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_title('聚类指标对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('figures/fig6_metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    X_raw, y_true = generate_complex_dataset(n_samples=1200, random_state=42)
    X_scaled, _ = preprocess_data(X_raw)
    em_gmm, sk_gmm, y_pred_ours, y_pred_sk = train_and_evaluate(X_scaled, y_true)
    metrics_ours = evaluate_clustering(y_true, y_pred_ours)
    metrics_sk = evaluate_clustering(y_true, y_pred_sk)
    plot_all_results(X_scaled, y_true, em_gmm, sk_gmm, y_pred_ours, y_pred_sk,
                     metrics_ours, metrics_sk)
    print("实验完成，图表已保存至 figures/ 目录")


if __name__ == '__main__':
    main()
