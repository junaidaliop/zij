# Lamb

Implements Lamb, layer-wise adaptive optimization for large-batch training.

Lamb rescales each layer's Adam-style update by the ratio between the norm of
the parameters and the norm of the update (the trust ratio), so that every
layer advances by a comparable relative amount regardless of its gradient
scale. This follows the v3 formulation, which omits the first-moment
de-biasing of the update and applies decoupled weight decay.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                         \\
     r_t &= \frac{m_t}{\sqrt{v_t} + \epsilon}                            \\
     \theta_{t-1} &\leftarrow (1 - \eta \lambda)\, \theta_{t-1}          \\
     \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}
         \frac{\phi(\lVert \theta_{t-1} \rVert)}{\lVert r_t \rVert} r_t
\end{aligned}
$$

where $\lambda$ is the decoupled weight decay and the trust ratio uses
$\phi(\lVert \theta \rVert) = \min(\lVert \theta \rVert, 10)$. The trust
ratio is set to one whenever the parameter norm or the update norm is zero.

Reference: Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar,
Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui
Hsieh, "Large Batch Optimization for Deep Learning: Training BERT in 76
minutes", ICLR 2020.
https://arxiv.org/abs/1904.00962

---
[Back to the Canon](../README.md)
