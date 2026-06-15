# Muon-OGD

Implements Muon-OGD, a Muon-style spectral-norm update with orthogonal non-interference constraints for continual learning.

Muon-OGD adapts a model to a new task while protecting prior tasks. Each step is cast as a spectral-norm-constrained optimization: descend along the (momentum) gradient $G_t$ under an operator-norm trust region $\lVert \Delta \rVert_2 \le \eta$, subject to non-interference constraints $\langle C_i, \Delta\rangle = 0$ that keep the update orthogonal to past-task directions $C_i$. Unlike Frobenius-norm orthogonal projection, the spectral-norm geometry inherits Muon's operator-norm steepest descent.

The constrained primal has a dual in which one descends on the multipliers $\lambda_i$ of the constraints; the shifted matrix $H = G + \sum_i \lambda_i C_i$ is orthogonalized by the matrix sign function $\mathrm{msgn}(\cdot) = UV^\top$ (the polar factor), approximated by five Newton-Schulz iterations $\mathrm{NS5}$. The full update over an inner dual loop $m = 0,\dots,T_{\mathrm{in}}-1$ is:

$$
\begin{aligned}
G_t &= \beta\, G_{t-1} + \nabla_\theta \ell(\theta_t; \mathcal{S}_t) \\
\Delta_t &= \arg\min_{\Delta} \langle G_t, \Delta\rangle \quad \mathrm{s.t.}\ \lVert \Delta \rVert_2 \le \eta,\ \langle C_i, \Delta\rangle = 0 \\
\lambda_i^{(m+1)} &= \lambda_i^{(m)} - \eta_\lambda \big\langle C_i,\ \mathrm{NS5}\big(G_t + \textstyle\sum_j \lambda_j^{(m)} C_j\big)\big\rangle \\
H_t &= G_t + \textstyle\sum_i \lambda_i^{(T_{\mathrm{in}})} C_i \\
\Delta_t &= -\eta\, \mathrm{NS5}(H_t) \\
\theta_{t+1} &= \theta_t + \Delta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate (operator-norm trust-region radius), $g_t = \nabla_\theta \ell(\theta_t; \mathcal{S}_t)$ the gradient on mini-batch $\mathcal{S}_t$, $G_t$ the momentum matrix, $\beta$ the momentum coefficient, $C_i$ the past-task constraint directions, $\lambda_i$ their dual multipliers, $\eta_\lambda$ the dual step size, $\lVert \cdot \rVert_2$ the spectral (induced matrix-2) norm, and $\mathrm{NS5}$ five Newton-Schulz iterations of $f(X) = \tfrac{3}{2}X - \tfrac{1}{2}XX^\top X$ approximating the matrix sign $\mathrm{msgn}(X) = UV^\top$.

Reference: Binghang Lu, Zheyuan Deng, Runyu Zhang, Bing Hu, Yunhan Zhao, Yuan Tian, Changhong Mou, Guang Lin, Xiaomin Li, "Muon-OGD: Muon-based Spectral Orthogonal Gradient Projection for LLM Continual Learning", arXiv 2026. https://arxiv.org/abs/2605.08949

---
[Back to the Canon](../README.md)
