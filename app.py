import io
import base64
import traceback
import webbrowser
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from flask import Flask, render_template, request, jsonify
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, adjusted_rand_score,
                             normalized_mutual_info_score)

from em_gmm import EMGMM
from data_loader import load_kddcup99

app = Flask(__name__)

current_data = {}


def align_labels(y_true, y_pred):
    """匈牙利算法映射聚类标签到真实标签"""
    from scipy.optimize import linear_sum_assignment
    n_true = len(np.unique(y_true))
    n_pred = len(np.unique(y_pred))
    size = max(n_true, n_pred)
    cm = np.zeros((size, size), dtype=int)
    for i in range(len(y_true)):
        p = min(y_pred[i], size - 1)
        t = min(y_true[i], size - 1)
        cm[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    result = np.array([mapping.get(p, 0) % n_true for p in y_pred])
    return result


def fig_to_base64(fig):
    """matplotlib 图像转 base64 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_data', methods=['POST'])
def api_load_data():
    """加载 KDDCUP99 数据集"""
    try:
        data = request.get_json() or {}
        n = int(data.get('n_samples', 3000))
        pca_dim = int(data.get('pca_dim', 5))

        X, X_2d, y, info = load_kddcup99(n_samples=n, pca_dim=pca_dim)
        current_data['X'] = X
        current_data['X_2d'] = X_2d
        current_data['y'] = y
        current_data['info'] = info

        # 生成 PCA 散点图
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        class_names = info['classes']
        for k in range(info['n_classes']):
            mask = y == k
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=colors_list[k % len(colors_list)], s=8, alpha=0.6,
                       label=f'{class_names[k]} ({np.sum(mask)})')
        ax.set_xlabel('PCA 成分1', fontsize=12)
        ax.set_ylabel('PCA 成分2', fontsize=12)
        ax.set_title('KDDCUP99 数据集 (PCA 可视化)', fontsize=13, fontweight='bold')
        ax.legend(markerscale=3, fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return jsonify({
            'success': True,
            'n_samples': info['total_samples'],
            'original_features': info['original_features'],
            'pca_dim': info['pca_dim'],
            'classes': info['classes'],
            'n_classes': info['n_classes'],
            'pca_var': [round(v, 3) for v in info['pca_explained_variance']],
            'image': fig_to_base64(fig)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'msg': str(e)})


@app.route('/api/run_em', methods=['POST'])
def api_run_em():
    """运行 EM 聚类算法"""
    try:
        if current_data.get('X') is None:
            return jsonify({'success': False, 'msg': '请先加载数据集'})

        data = request.get_json() or {}
        K = int(data.get('n_components', 3))       # 高斯分量个数
        max_iter = int(data.get('max_iter', 150))   # 最大迭代次数

        X = current_data['X']
        X_2d = current_data['X_2d']
        y_true = current_data['y']
        info = current_data['info']
        class_names = info['classes']
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        # 训练自实现 EM-GMM
        em = EMGMM(n_components=K, max_iter=max_iter, tol=1e-4, random_state=42)
        em.fit(X)
        y_pred = em.predict(X)
        y_pred_aligned = align_labels(y_true, y_pred)

        # 计算聚类评估指标
        acc = accuracy_score(y_true, y_pred_aligned)
        prec = precision_score(y_true, y_pred_aligned, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred_aligned, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred_aligned, average='macro', zero_division=0)
        ari = adjusted_rand_score(y_true, y_pred_aligned)
        nmi = normalized_mutual_info_score(y_true, y_pred_aligned)

        # 图1：EM 收敛曲线
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(1, len(em.history_) + 1), em.history_, 'b-', linewidth=2)
        ax1.axvline(x=em.n_iter_, color='r', linestyle='--',
                    label=f'收敛点 (迭代 {em.n_iter_})')
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('对数似然', fontsize=12)
        ax1.set_title('EM 算法收敛曲线', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        img1 = fig_to_base64(fig1)
        plt.close(fig1)

        # 图2：聚类结果对比（真实标签 vs 聚类结果）
        fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
        for k in range(info['n_classes']):
            mask = y_true == k
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            c=colors_list[k % len(colors_list)], s=8, alpha=0.6)
        axes[0].set_title('真实标签', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        for k in range(min(K, info['n_classes'])):
            mask = y_pred_aligned == k
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            c=colors_list[k % len(colors_list)], s=8, alpha=0.6,
                            label=f'{class_names[k]}')
        axes[1].set_title(f'EM 聚类结果 (ACC={acc:.3f})', fontsize=13, fontweight='bold')
        axes[1].legend(markerscale=3)
        axes[1].grid(True, alpha=0.3)
        fig2.tight_layout()
        img2 = fig_to_base64(fig2)
        plt.close(fig2)

        # 图3：混淆矩阵
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred_aligned)
        im = ax3.imshow(cm, cmap='Blues')
        n_classes_cm = cm.shape[0]
        for i in range(n_classes_cm):
            for j in range(n_classes_cm):
                ax3.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax3.set_xticks(range(n_classes_cm))
        ax3.set_yticks(range(n_classes_cm))
        short_names = [n[:8] for n in class_names]
        ax3.set_xticklabels(short_names, fontsize=10)
        ax3.set_yticklabels(short_names, fontsize=10)
        ax3.set_xlabel('预测标签', fontsize=12)
        ax3.set_ylabel('真实标签', fontsize=12)
        ax3.set_title('混淆矩阵', fontsize=13, fontweight='bold')
        fig3.tight_layout()
        img3 = fig_to_base64(fig3)
        plt.close(fig3)

        return jsonify({
            'success': True,
            'n_iter': em.n_iter_,
            'converged': em.converged_,
            'log_likelihood': round(em.log_likelihood_, 1),
            'weights': [round(w, 4) for w in em.weights_.tolist()],
            'pca_var': info.get('pca_explained_variance', []),
            'metrics': {
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1': round(f1, 4),
                'ari': round(ari, 4),
                'nmi': round(nmi, 4)
            },
            'images': {'convergence': img1, 'clustering': img2, 'confusion': img3}
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'msg': str(e)})


if __name__ == '__main__':
    # 延迟打开浏览器，等服务器启动
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=False, host='0.0.0.0', port=5000)
