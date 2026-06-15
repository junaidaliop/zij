# CT-AGD

Implements CT-AGD, curvature-tuned accelerated gradient descent with a cheap diagonal-Hessian estimate.

CT-AGD splits optimization into two regimes. Within each epoch it runs ordinary first-order steps whose learning rate is divided by a curvature-aware factor $\gamma_{k,t}$ that anneals linearly toward $1$ over the $T$ steps of the epoch. Across consecutive steps it accumulates a finite-difference diagonal Hessian estimate $\hat{H}_k$, clipped into a safe interval, and uses it both for one second-order-informed step at the epoch boundary and to set the next epoch's curvature factor as a low-tail quantile of the estimated curvature. The scheme adds only the bookkeeping of one previous step, so the overhead over plain gradient descent is minimal.

$$
\begin{aligned}
\theta_{k,t+1} &= \theta_{k,t} - \frac{\eta_1}{\gamma_{k,t}}\, g_{k,t}, &\quad \gamma_{k,t} &= \gamma_k - (\gamma_k - 1)\,\frac{t}{T} \\
h_{k,t} &= \frac{g_{k,t} - g_{k,t-1}}{\theta_{k,t} - \theta_{k,t-1}}, &\quad \hat{H}_k &= \Pi_{[\lambda_{\min},\lambda_{\max}]}\!\left( \frac{\sum_{t=1}^{T-1} t\,(m_{k,t} \odot h_{k,t})}{\sum_{t=1}^{T-1} t\, m_{k,t} + \epsilon} \right) \\
\theta_{k+1,0} &= \theta_{k,T-1} - \eta_2\, \frac{1}{\hat{H}_k} \odot \tilde{g}_k, &\quad \gamma_k &= Q_\omega(\hat{H}_k)
\end{aligned}
$$

where $\theta_{k,t}$ are the parameters at step $t$ of epoch $k$, $g_{k,t}$ the gradient, $\eta_1,\eta_2$ the within-epoch and epoch-end learning rates, $h_{k,t}$ the element-wise finite-difference curvature, $m_{k,t}$ a validity mask that is $1$ where $|\theta_{k,t}-\theta_{k,t-1}| > \epsilon$ and $0$ otherwise, $\odot$ element-wise product, $\Pi_{[\lambda_{\min},\lambda_{\max}]}$ projection (clipping) onto the curvature interval, $\tilde{g}_k$ the weighted-average or last gradient of the epoch, $Q_\omega$ the low-tail $\omega$-quantile of the diagonal entries, and $\epsilon$ a stability constant.

Reference: Manuel Graca, L. Miguel Silveira, Arlindo Oliveira, Frank Liu, "Accelerated Gradient Descent for Faster Convergence with Minimal Overhead", arXiv 2026. https://arxiv.org/abs/2605.16017

---
[Back to the Canon](../README.md)
