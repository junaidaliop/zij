# SNGM

Implements SNGM, stochastic normalized gradient descent with momentum for large-batch training.

SNGM accumulates the normalized stochastic gradient into a momentum buffer rather than the raw gradient. Dividing each gradient by its Euclidean norm before the momentum update keeps the effective step from blowing up when batch size grows, which lets SNGM train with much larger batches while retaining convergence guarantees comparable to small-batch momentum SGD.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + \frac{g_t}{\lVert g_t \rVert} \\
\theta_t &= \theta_{t-1} - \eta\, m_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the mini-batch gradient, $\lVert g_t \rVert$ its Euclidean norm, $m_t$ the momentum buffer, $\beta \in [0,1)$ the momentum coefficient, and $\eta$ the learning rate.

Reference: Shen-Yi Zhao, Yin-Peng Xie, Wu-Jun Li, "Stochastic Normalized Gradient Descent with Momentum for Large Batch Training", 2020. https://arxiv.org/abs/2007.13985

---
[Back to the Canon](../index.md)
