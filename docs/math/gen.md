# GeN

Implements GeN, a learning-rate-free wrapper that sets the step size each iteration via a generalized Newton's method.

GeN sits on top of any base optimizer: the base optimizer proposes a descent direction $d_t$ (the raw gradient for SGD, the preconditioned update for Adam, and so on), and GeN computes the optimal scalar step along that direction from a second-order Taylor expansion of the loss. Minimizing the expansion over $\eta$ gives a closed form involving the gradient and the Hessian-vector product. To avoid forming the Hessian, GeN estimates the curvature with a few extra forward passes: it evaluates the loss at the current point and at two points perturbed forward and backward along $d_t$, then reads off the optimal step from the resulting finite differences. Applying this every few steps amortizes the cost.

$$
\begin{aligned}
\eta_t^{*} &= \frac{g_t^{\top} d_t}{d_t^{\top} H_t\, d_t} \\
\theta_{t+1} &= \theta_t - \eta_t^{*}\, d_t \\
\eta_t^{*} &\approx \frac{\eta_{t-1}}{2}\cdot\frac{L_+ - L_-}{L_+ - 2 L_0 + L_-}
\end{aligned}
$$

where $\theta$ are the parameters, $d_t$ the descent direction supplied by the base optimizer, $g_t$ the (oracle) gradient of the loss, $H_t$ the Hessian, and $\eta_t^{*}$ the derived step size; $L_0 = L(\theta_t)$ is the current loss and $L_\pm = L(\theta_t \pm \eta_{t-1} d_t)$ are the losses at the forward- and backward-perturbed points, so the third line replaces the gradient and Hessian terms with second-order finite differences requiring only forward passes. For plain gradient descent $d_t = g_t$, and the step reduces to $g_t^{\top} g_t / (g_t^{\top} H_t g_t)$.

Reference: Zhiqi Bu, Shiyun Xu, "Automatic gradient descent with generalized Newton's method", arXiv 2024. https://arxiv.org/abs/2407.02772

---
[Back to the Canon](../index.md)
