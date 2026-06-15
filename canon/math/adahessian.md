# Adahessian

Implements AdaHessian, an adaptive second-order optimizer.

AdaHessian replaces the squared-gradient denominator of Adam with a running
average of the squared diagonal of the Hessian, estimated with a
Hutchinson matrix-free probe. For each step a Rademacher vector $z$
(entries $\pm 1$) is drawn and the Hessian-vector product
$H_t z$ is formed by differentiating $g_t^\top z$. The
per-element magnitude $|H_t z|$ is then block-averaged to reduce its
variance, giving the block-averaged diagonal estimate $D_t^{(s)}$.
With first moment $m_t$, second moment $v_t$ over
$D_t^{(s)}$, learning rate $\eta$, decay rates $\beta_1$,
$\beta_2$, and Hessian power $k$:


$$
\begin{aligned}
D_t^{(s)} &= \frac{1}{b} \sum_{\text{block}} |H_t z|,
    \qquad z_i \in \{-1, +1\} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \bigl(D_t^{(s)}\bigr)^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
    \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \left(
    \frac{\hat{m}_t}{\hat{v}_t^{\,k/2} + \epsilon}
    + \lambda\, \theta_{t-1} \right)
\end{aligned}
$$

where $\lambda$ is the `weight_decay` and $b$ is the number of
elements in each structured block. The per-element magnitude
$|H_t z|$ is averaged (not the signed product) over each block of
size $b$: a 2D Conv kernel is averaged over its spatial extent,
matching the block-diagonal averaging of the paper. Setting $k = 1$
recovers the standard Hessian power; $k = 0.5$ is a milder
preconditioner.


**Note:** AdaHessian needs the Hessian-vector product, so the gradients passed

to `step` must carry an autograd graph. Call `loss.backward(
create_graph=True)` before `step` (or pass a closure that does so).
Without `create_graph=True` the gradients have no `grad_fn` and
`step` raises. Sparse gradients are not supported.

Reference: Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa,
Kurt Keutzer, Michael W. Mahoney, "ADAHESSIAN: An Adaptive Second Order
Optimizer for Machine Learning", AAAI 2021.
https://arxiv.org/abs/2006.00719

---
[Back to the Canon](../README.md)
