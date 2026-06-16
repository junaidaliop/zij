# AdaFedAdam

Implements AdaFedAdam, a federated optimizer that applies Adam to fairness-weighted client pseudo-gradients.

AdaFedAdam treats each round of federated learning as a single server-side Adam step. Each client returns its accumulated local update $\Delta_k$, which is normalized into a pseudo-gradient $U_k$ together with a certainty score $C_k$ that measures how far the local trajectory deviated from a plain gradient step. The server aggregates these pseudo-gradients with weights that combine dataset size, an inverse training-rate term $I_k$ raised to a fairness exponent $\alpha$, and then runs Adam on the aggregate.

To stay robust to heterogeneous client objectives, the aggregated certainty $C$ rescales Adam's hyperparameters per round: the learning rate is scaled by $C$ and the decay rates are raised to the power $C$, so less certain rounds take smaller, more heavily smoothed steps.

$$
\begin{aligned}
U_k &= -\frac{\Delta_k}{\eta'_k}, \qquad \eta'_k = \frac{\lVert \Delta_k \rVert_2}{\lVert \nabla F_k(\theta_t) \rVert_2}, \qquad C_k = \log\!\frac{\eta'_k}{\eta_k} + 1 \\
w_k &= \frac{S_k\, I_k^{\alpha}}{\sum_j S_j\, I_j^{\alpha}}, \qquad I_k = \frac{F_k(\theta_t)}{F_k(\theta_0)} \\
g_t &= \sum_k w_k\, U_k, \qquad C = \sum_k w_k\, C_k \\
m_t &= (1-\beta_1^{C})\, g_t + \beta_1^{C}\, m_{t-1}, \qquad v_t = (1-\beta_2^{C})\, g_t \odot g_t + \beta_2^{C}\, v_{t-1} \\
\hat{m}_t &= \frac{m_t}{1-c_{t,m}}, \qquad \hat{v}_t = \frac{v_t}{1-c_{t,v}} \\
\theta_{t+1} &= \theta_t - C\,\eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the global parameters, $\eta$ the base server learning rate, $F_k$ client $k$'s local objective, $\Delta_k$ its accumulated local update, $\eta_k$ its local learning rate, $S_k$ its dataset-size weight, $\alpha \ge 0$ the fairness exponent, $\beta_1,\beta_2$ the base decay rates, $c_{t,m},c_{t,v}$ the accumulated decay products used for bias correction, $\odot$ elementwise product, and $\epsilon$ a stability constant.

Reference: Li Ju, Tianru Zhang, Salman Toor, Andreas Hellander, "Accelerating Fair Federated Learning: Adaptive Federated Adam", arXiv 2023. https://arxiv.org/abs/2301.09357

---
[Back to the Canon](../index.md)
