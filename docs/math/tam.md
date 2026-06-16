# TAM

Implements TAM (Torque-Aware Momentum), a momentum method that damps gradients misaligned with the accumulated velocity.

At each step TAM measures the cosine similarity between the current gradient and the previous momentum, then smooths it with an exponential moving average. This smoothed correlation forms a damping coefficient that scales the gradient's contribution to momentum: gradients aligned with the velocity pass through fully, while opposing gradients are attenuated, suppressing oscillations and improving exploration of flatter regions.

$$
\begin{aligned}
S_t &= \frac{m_{t-1} \cdot g_t}{\lVert m_{t-1}\rVert \, \lVert g_t\rVert} \\
\hat{s}_t &= \gamma\,\hat{s}_{t-1} + (1-\gamma)\,S_t \\
d_t &= \frac{1 + \hat{s}_t}{2} \\
m_t &= \beta\,m_{t-1} + (\epsilon + d_t)\,g_t \\
\theta_{t+1} &= \theta_t - \eta\,m_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum, $\beta$ the momentum coefficient, $S_t$ the cosine similarity between gradient and previous momentum, $\hat{s}_t$ its exponential moving average with smoothing rate $\gamma$, $d_t \in [0,1]$ the resulting damping coefficient, and $\epsilon$ a small constant for stability.

Reference: Pranshu Malviya, Gonçalo Mordido, Aristide Baratin, Reza Babanezhad Harikandeh, Gintare Karolina Dziugaite, Razvan Pascanu, Sarath Chandar, "Torque-Aware Momentum", arXiv 2024. https://arxiv.org/abs/2412.18790

---
[Back to the Canon](../index.md)
