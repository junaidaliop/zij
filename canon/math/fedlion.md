# FedLion

Implements FedLion, a federated optimizer that ports the centralized Lion update into local client training with momentum aggregation.

Each client runs $E$ local steps of Lion: a sign-based parameter update driven by an interpolation of the running momentum and the current gradient, while a separate exponential moving average of the gradient is tracked for the next round. After local training the client transmits its accumulated model delta and its momentum to the server, which averages both across the $n$ participating clients to form the next global model and the shared momentum, so only the signed-direction information and a single momentum buffer are communicated.

For each client $i$ and local step $s = 1, \dots, E$, starting from the global iterate $\theta_{t-1}$ and shared momentum $m_{t-1}$:

$$
\begin{aligned}
c_{t,s}^{i} &= \beta_1 m_{t,s-1}^{i} + (1 - \beta_1) g_{t,s}^{i} \\
\theta_{t,s}^{i} &= \theta_{t,s-1}^{i} - \gamma \, \mathrm{sign}\!\left(c_{t,s}^{i}\right) \\
m_{t,s}^{i} &= \beta_2 m_{t,s-1}^{i} + (1 - \beta_2) g_{t,s}^{i} \\
\theta_{t} &= \theta_{t-1} - \frac{\gamma}{n} \sum_{i=1}^{n} \Delta_{t-1}^{i}, \qquad m_{t} = \frac{1}{n} \sum_{i=1}^{n} m_{t-1,E}^{i}
\end{aligned}
$$

where $g_{t,s}^{i}$ is the stochastic gradient of client $i$ at local step $s$, $\gamma$ is the learning rate, $\beta_1,\beta_2$ are the momentum coefficients (with $\mathrm{sign}$ taken elementwise), $\Delta_{t-1}^{i} = (\theta_{t-1} - \theta_{t-1,E}^{i})/\gamma$ is the client's accumulated update direction, $n$ is the number of participating clients, and the last line is the server-side aggregation forming the next global model $\theta_t$ and shared momentum $m_t$.

Reference: Zhiwei Tang, Tsung-Hui Chang, "FedLion: Faster Adaptive Federated Optimization with Fewer Communication", ICASSP 2024. https://arxiv.org/abs/2402.09941

---
[Back to the Canon](../README.md)
