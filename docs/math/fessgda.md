# FESS-GDA

Implements FESS-GDA, federated stochastic smoothed gradient descent ascent for distributed minimax optimization.

FESS-GDA solves federated minimax problems $\min_x \max_y \frac{1}{M}\sum_i f_i(x,y)$ by carrying the centralized Smoothed-AGDA technique into the federated setting. Each communication round the server samples $m$ clients and broadcasts the global model $(x_t, y_t)$; every participating client synchronizes and runs $K$ local stochastic descent-on-$x$, ascent-on-$y$ steps with local learning rates $\eta_{x,l}, \eta_{y,l}$, then returns its local model. The key departure from plain federated GDA is an auxiliary variable $z_t$ that smooths the $x$ update: the server effectively performs gradient descent ascent on the regularized surrogate $\hat f(x,y,z) = f(x,y) + \frac{p}{2}\lVert x - z\rVert^2$, with $z_t$ tracking $x_t$ via exponential averaging. The penalty $p$ is set to $2l$ for the nonconvex-concave / PL / 1-point-concave settings (making $\hat f$ strongly convex in $x$) and $p = 0$ for the PL-PL setting.

$$
\begin{aligned}
x_{t,i}^{k+1} &= x_{t,i}^{k} - \eta_{x,l}\, \nabla_x f_i\!\left(x_{t,i}^{k}, y_{t,i}^{k}, \xi_{t,i}^{k}\right) \\
y_{t,i}^{k+1} &= P_Y\!\left(y_{t,i}^{k} + \eta_{y,l}\, \nabla_y f_i\!\left(x_{t,i}^{k}, y_{t,i}^{k}, \xi_{t,i}^{k}\right)\right) \\
x_{t+1} &= x_t + \eta_{x,g}\left(\frac{1}{m}\sum_{i\in S_t} x_{t,i}^{K+1} - x_t\right) - \eta_{x,l}\,\eta_{x,g}\,K\,p\,(x_t - z_t) \\
y_{t+1} &= P_Y\!\left(y_t + \eta_{y,g}\left(\frac{1}{m}\sum_{i\in S_t} y_{t,i}^{K+1} - y_t\right)\right) \\
z_{t+1} &= z_t + \beta\,(x_{t+1} - z_t)
\end{aligned}
$$

where $x$ is the minimization variable and $y$ the maximization variable, $z_t$ is the smoothing auxiliary variable, $f_i$ is client $i$'s local function with stochastic sample $\xi_{t,i}^{k}$, $k = 1,\dots,K$ indexes the local steps initialized at $x_{t,i}^{1}=x_t,\ y_{t,i}^{1}=y_t$, $S_t$ is the sampled client subset with $|S_t| = m$, $\eta_{x,l}, \eta_{y,l}$ are local and $\eta_{x,g}, \eta_{y,g}$ global learning rates, $p$ is the smoothing penalty, $\beta \in (0,1)$ is the smoothing rate, and $P_Y$ is projection onto the feasible set $Y$.

Reference: Wei Shen, Minhui Huang, Jiawei Zhang, Cong Shen, "Stochastic Smoothed Gradient Descent Ascent for Federated Minimax Optimization", AISTATS 2024. https://arxiv.org/abs/2311.00944

---
[Back to the Canon](../index.md)
