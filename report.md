# EM算法在KDDCUP99网络入侵检测中的应用

---

## 使用说明

### 方式一：直接运行 exe（推荐）

双击 `EM算法-KDDCUP99.exe`，浏览器会自动打开 http://127.0.0.1:5000，在网页中进行数据加载和EM聚类分析。

### 方式二：运行源代码

```bash
pip install flask numpy scikit-learn scipy matplotlib
python app.py
```

然后浏览器打开 http://127.0.0.1:5000

### 网页操作

1. 左侧面板设置**样本数量**和**PCA维度**，点击"加载数据"
2. 查看KDDCUP99数据集的PCA可视化散点图
3. 设置**高斯分量个数K**和**最大迭代次数**，点击"运行EM算法"
4. 查看收敛曲线、聚类结果对比、混淆矩阵和评估指标

### 命令行实验

```bash
python main.py
```

将生成合成数据集，使用自实现EM和sklearn GMM进行对比实验，结果图表保存至 `figures/` 目录。

### 目录结构

```
├── app.py              # Flask Web 后端
├── data_loader.py      # KDDCUP99 数据加载与预处理
├── em_gmm.py           # 自实现 EM-GMM 算法
├── main.py             # 命令行实验脚本（合成数据）
├── EM算法-KDDCUP99.exe # 打包好的可执行文件
├── report.md           # 实验报告
├── flowchart.drawio    # 算法流程图（Draw.io 打开）
├── data/
│   └── kddcup.data_10_percent.gz  # KDDCUP99 10% 数据集
└── templates/
    └── index.html      # 前端页面
```

---

## 1 背景简介

### 1.1 算法产生原因

在许多实际问题中，数据往往来自多个混合分布，但我们并不知道每个样本具体属于哪个子分布。高斯混合模型（Gaussian Mixture Model, GMM）用多个高斯分布的加权和来拟合数据，而EM（Expectation-Maximization，期望最大化）算法正是求解GMM参数的经典方法。

### 1.2 算法基本原理

EM算法是一种迭代优化算法，通过交替执行E步（Expectation）和M步（Maximization）来最大化似然函数：

- **E步**：根据当前参数估计每个样本属于各高斯分量的后验概率（责任值）
- **M步**：根据责任值重新估计模型参数（混合系数、均值、协方差）

反复迭代直至对数似然函数收敛。

### 1.3 术语定义

| 术语 | 定义 |
|------|------|
| **高斯混合模型（GMM）** | K个高斯分布的线性组合：p(x) = Σ π_k N(x|μ_k, Σ_k) |
| **混合系数 π_k** | 第k个高斯分量的权重，满足 Σ π_k = 1 |
| **责任值 γ_ik** | 样本 x_i 属于第k个分量的后验概率 |
| **对数似然** | ln p(X|π,μ,Σ) = Σ ln Σ π_k N(x_i|μ_k, Σ_k) |
| **KDDCUP99** | 经典的网络入侵检测数据集，包含正常连接和多种攻击类型 |

---

## 2 算法应用

### 2.1 数据预处理

本实验使用KDDCUP99数据集的10%子集（`kddcup.data_10_percent.gz`），预处理流程如下：

1. **数据读取**：解析CSV格式的原始数据，提取数值特征列（跳过 protocol_type、service、flag 三个分类列），保留38维数值特征
2. **标签映射**：将22种具体攻击类型映射为5大类：normal（正常）、dos（拒绝服务）、probe（探测）、r2l（远程非法访问）、u2r（权限提升），本实验选取 normal、dos、probe 三类
3. **分层采样**：为保证各类别均衡，对 normal 和 dos 类随机采样，probe 类全部保留
4. **标准化**：使用 Z-score 归一化，使各特征均值为0、方差为1
5. **PCA降维**：将38维特征降维（默认5维用于聚类计算，2维用于可视化）

### 2.2 算法流程图

详见 `flowchart.drawio`，用 Draw.io 打开可查看完整流程图。

### 2.3 算法伪代码

```
算法：EM for Gaussian Mixture Model

输入：数据集 X = {x1, x2, ..., xN}，分量数 K，最大迭代次数 max_iter，收敛阈值 tol
输出：混合系数 π_k，均值 μ_k，协方差 Σ_k (k = 1, ..., K)

1.  初始化 π_k, μ_k, Σ_k (K-means++ 初始化)
2.  prev_ll ← -∞
3.  for iter = 1 to max_iter do
4.      // E步：计算责任值
5.      for i = 1 to N do
6.          for k = 1 to K do
7.              γ_ik ← π_k * N(x_i | μ_k, Σ_k)
8.          end for
9.          归一化 γ_i 使其和为1
10.     end for
11.     // M步：更新参数
12.     for k = 1 to K do
13.         N_k ← sum_i γ_ik
14.         π_k ← N_k / N
15.         μ_k ← (1/N_k) * sum_i γ_ik * x_i
16.         Σ_k ← (1/N_k) * sum_i γ_ik * (x_i - μ_k)(x_i - μ_k)^T + ε*I
17.     end for
18.     // 计算对数似然
19.     ll ← sum_i ln(sum_k π_k * N(x_i | μ_k, Σ_k))
20.     // 检查收敛
21.     if |ll - prev_ll| < tol then
22.         converged ← True, break
23.     end if
24.     prev_ll ← ll
25. end for
26. return π_k, μ_k, Σ_k
```

### 2.4 程序说明

#### `em_gmm.py` — EM-GMM 算法核心

| 类/方法 | 说明 |
|---------|------|
| `EMGMM` | EM算法求解高斯混合模型 |
| `__init__` | 初始化模型参数（分量数、最大迭代、收敛阈值、随机种子） |
| `_initialize` | K-means++ 风格初始化均值和协方差 |
| `_e_step` | E步：计算每个样本属于各分量的后验概率（责任值） |
| `_m_step` | M步：根据责任值更新混合系数、均值、协方差 |
| `_compute_log_likelihood` | 计算完整数据的对数似然值 |
| `fit` | 训练模型：交替执行E步和M步直到收敛 |
| `predict` | 预测每个样本所属的簇（取最大后验概率） |
| `predict_proba` | 返回每个样本属于各分量的概率 |
| `score` | 返回模型的对数似然值 |

#### `data_loader.py` — 数据加载

| 函数 | 说明 |
|------|------|
| `load_kddcup99` | 加载KDDCUP99数据集，返回PCA降维特征和标签 |

#### `app.py` — Web应用后端

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 渲染前端页面 |
| `/api/load_data` | POST | 加载数据集，返回PCA散点图 |
| `/api/run_em` | POST | 运行EM聚类，返回收敛曲线、聚类结果图、混淆矩阵及评估指标 |

| 辅助函数 | 说明 |
|----------|------|
| `align_labels` | 匈牙利算法映射聚类标签到真实标签 |
| `fig_to_base64` | 将matplotlib图像转为base64字符串 |

### 2.5 时间和空间复杂度

**时间复杂度：** O(T · K · N · D²)

- T：迭代次数（通常为 20~100）
- K：高斯分量个数
- N：样本数
- D：特征维度
- E步中计算多元高斯概率密度需要 O(K · N · D²)（协方差矩阵求逆和行列式），是主要计算瓶颈
- M步更新协方差矩阵需要 O(K · N · D²)
- 通过PCA降维（D 从 38 降至 5）显著减少了计算量

**空间复杂度：** O(K · N + K · D²)

- 责任矩阵 γ：O(K · N)
- K 个协方差矩阵：O(K · D²)
- 数据矩阵 X：O(N · D)

### 2.6 算法边界

**适用场景：**
- 数据近似服从多个高斯分布的混合
- 各类别样本量差异不大
- 特征维度适中（经PCA降维后效果更好）
- 簇之间有一定分离度

**不适用场景：**
- 非凸形状的簇（如环形、月牙形），高斯分布无法拟合
- 簇数量未知且无法通过BIC/AIC等准则确定
- 数据中存在大量离群点时，EM容易受到干扰
- 高维稀疏数据，协方差矩阵容易奇异
- 样本量远小于特征维度时，需要正则化（本实现已添加 εI 防奇异）

**边界情况分析：**
1. **初始化敏感**：不同随机种子可能收敛到不同的局部最优解，可通过多次运行取最优
2. **K值选择**：K值过小导致欠拟合，过大导致过拟合，建议用BIC/AIC选取最优K
3. **簇高度重叠**：簇间协方差大且重叠严重时，EM算法分离效果下降
4. **协方差奇异**：当某分量分配到极少样本时，通过添加 10⁻⁶ I 正则化项防止奇异

---

## 3 运行结果

### 3.1 数据集概览

KDDCUP99数据集经预处理后包含三类网络流量：normal（正常）、dos（拒绝服务攻击）、probe（探测攻击）。经PCA降至2维后分布见前端页面散点图。

### 3.2 EM聚类结果

以 K=3（normal、dos、probe）运行EM-GMM算法：

| 指标 | 含义 |
|------|------|
| Accuracy | 聚类准确率 |
| Precision | 宏平均精确率 |
| Recall | 宏平均召回率 |
| F1 Score | 宏平均F1分数 |
| ARI | 调整兰德指数 |
| NMI | 标准化互信息 |

### 3.3 收敛分析

EM算法通常在 20~50 次迭代内收敛，对数似然值单调递增至稳定。

### 3.4 混淆矩阵

混淆矩阵显示各类别的聚类分配情况，normal 和 dos 类由于特征差异较大，分离效果较好；probe 类因样本量少，聚类精度略低。

---

## 4 结论

本实验实现并应用了EM算法求解高斯混合模型，在KDDCUP99网络入侵检测数据集上进行了聚类分析。主要结论如下：

1. **EM算法有效性**：自实现的EM-GMM能够有效对网络流量数据进行无监督聚类，在normal、dos、probe三类上取得了较好的聚类效果
2. **数据预处理的重要性**：标准化和PCA降维对EM算法的收敛速度和聚类精度有显著提升
3. **算法局限性**：EM算法对初始值敏感，可能收敛到局部最优；对非高斯分布数据适用性有限
4. **与sklearn对比**：自实现EM-GMM与sklearn GaussianMixture在相同参数下结果一致，验证了实现的正确性

---

## 5 附录

### 附录A 核心代码

#### `em_gmm.py` — EM-GMM 算法实现

```python
import numpy as np
from scipy.stats import multivariate_normal


class EMGMM:
    """EM算法求解高斯混合模型"""

    def __init__(self, n_components=3, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None       # 混合系数
        self.means_ = None         # 均值
        self.covariances_ = None   # 协方差
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None
        self.history_ = []

    def _initialize(self, X):
        # K-means++ 初始化
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        means = np.zeros((self.n_components, n_features))
        means[0] = X[rng.randint(n_samples)]
        for k in range(1, self.n_components):
            dist = np.min([np.sum((X - means[j]) ** 2, axis=1)
                           for j in range(k)], axis=0)
            prob = dist / dist.sum()
            means[k] = X[rng.choice(n_samples, p=prob)]
        cov = np.cov(X, rowvar=False)
        covariances = np.array([cov for _ in range(self.n_components)])
        weights = np.ones(self.n_components) / self.n_components
        return weights, means, covariances

    def _e_step(self, X):
        # E步：计算责任值
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            try:
                rv = multivariate_normal(mean=self.means_[k],
                                         cov=self.covariances_[k],
                                         allow_singular=True)
                resp[:, k] = self.weights_[k] * rv.pdf(X)
            except Exception:
                reg_cov = self.covariances_[k] + 1e-6 * np.eye(X.shape[1])
                rv = multivariate_normal(mean=self.means_[k], cov=reg_cov)
                resp[:, k] = self.weights_[k] * rv.pdf(X)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum == 0] = 1e-300
        resp /= resp_sum
        return resp

    def _m_step(self, X, resp):
        # M步：更新参数
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)
        self.weights_ = Nk / n_samples
        self.means_ = np.dot(resp.T, X) / Nk[:, np.newaxis]
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k][:, np.newaxis] * diff
            self.covariances_[k] = np.dot(weighted_diff.T, diff) / Nk[k]
            self.covariances_[k] += 1e-6 * np.eye(n_features)  # 正则化

    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        log_lik = 0.0
        for i in range(n_samples):
            point_prob = 0.0
            for k in range(self.n_components):
                rv = multivariate_normal(mean=self.means_[k],
                                         cov=self.covariances_[k],
                                         allow_singular=True)
                point_prob += self.weights_[k] * rv.pdf(X[i])
            log_lik += np.log(max(point_prob, 1e-300))
        return log_lik

    def fit(self, X):
        self.weights_, self.means_, self.covariances_ = self._initialize(X)
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._compute_log_likelihood(X)
            self.history_.append(ll)
            if iteration > 0 and abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                self.log_likelihood_ = ll
                return self
            prev_ll = ll
        self.n_iter_ = self.max_iter
        self.log_likelihood_ = prev_ll
        return self

    def predict(self, X):
        resp = self._e_step(X)
        return np.argmax(resp, axis=1)

    def predict_proba(self, X):
        return self._e_step(X)

    def score(self, X):
        return self._compute_log_likelihood(X)
```

### 参考文献

[1] Dempster A P, Laird N M, Rubin D B. Maximum likelihood from incomplete data via the EM algorithm[J]. Journal of the Royal Statistical Society, 1977.

[2] 韩家炜. 数据挖掘概念与技术[M]. 机械工业出版社.

[3] 周志华. 机器学习[M]. 清华大学出版社.

[4] scikit-learn developers. sklearn.mixture.GaussianMixture[EB/OL]. https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

[5] KDD Cup 1999 Data Set[EB/OL]. UCI Machine Learning Repository. https://kdd.ics.uci.edu/databases/kddcup99/
