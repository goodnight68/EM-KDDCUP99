import numpy as np
from scipy.stats import multivariate_normal


class EMGMM:
    """EM算法求解高斯混合模型"""

    def __init__(self, n_components=3, max_iter=200, tol=1e-4, random_state=None):
        # n_components: 高斯分量个数 K
        # tol: 对数似然收敛阈值
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None       # 混合系数 pi_k
        self.means_ = None         # 均值 mu_k
        self.covariances_ = None   # 协方差 sigma_k
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None
        self.history_ = []         # 每次迭代的对数似然值

    def _initialize(self, X):
        # K-means++ 风格初始化
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
        # E步：计算责任值（后验概率）
        # gamma_ik = pi_k * N(x_i|mu_k,sigma_k) / sum_j pi_j * N(x_i|mu_j,sigma_j)
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
        # M步：用责任值更新参数
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)                       # 每个分量的有效样本数
        self.weights_ = Nk / n_samples               # pi_k = N_k / N
        self.means_ = np.dot(resp.T, X) / Nk[:, np.newaxis]  # mu_k
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k][:, np.newaxis] * diff
            self.covariances_[k] = np.dot(weighted_diff.T, diff) / Nk[k]
            self.covariances_[k] += 1e-6 * np.eye(n_features)  # 正则化防止奇异

    def _compute_log_likelihood(self, X):
        """计算对数似然"""
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
        """训练模型：E步 -> M步 -> 检查收敛，反复迭代"""
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
        """预测样本所属簇"""
        resp = self._e_step(X)
        return np.argmax(resp, axis=1)

    def predict_proba(self, X):
        """返回每个样本属于各分量的概率"""
        return self._e_step(X)

    def score(self, X):
        """返回对数似然值"""
        return self._compute_log_likelihood(X)
