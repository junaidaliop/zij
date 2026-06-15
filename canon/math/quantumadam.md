# Quantum Adam

Implements Quantum Adam, an Adam variant that couples $M$ replicas of the network through a quantum-fluctuation term derived from a path-integral representation.

The method mirrors quantum annealing: it optimizes $M$ Trotter replicas of the same network simultaneously and adds an elastic, attracting force between neighboring replicas. This force is the discrete Laplacian $g^q_t = 2\theta^k_t - \theta^{k+1}_t - \theta^{k-1}_t$ (with periodic boundary $\theta^0 = \theta^M = \theta$), which lets replicas tunnel past potential barriers toward broader, better-generalizing minima. Each replica runs ordinary Adam on its data gradient $g_t$, plus a second Adam-style term on the quantum gradient $g^q_t$ scaled by a mass $\rho_t$ that grows from $0$ to large values over training, so the replicas gradually merge.

$$
\begin{aligned}
m_t &= (1-\beta_1)\,m_{t-1} + \beta_1\,g_t, & v_t &= (1-\beta_2)\,v_{t-1} + \beta_2\,g_t \odot g_t,\\
m^q_t &= (1-\beta_1)\,m^q_{t-1} + \beta_1\,g^q_t, & v^q_t &= (1-\beta_2)\,v^q_{t-1} + \beta_2\,g^q_t \odot g^q_t,\\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}, & \hat{m}^q_t &= \frac{m^q_t}{1-\beta_1^{\,t}}, \quad \hat{v}^q_t = \frac{v^q_t}{1-\beta_2^{\,t}},\\
\theta^k_{t+1} &= \theta^k_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\,\hat{m}_t - \frac{\eta\,\rho_t}{\sqrt{\hat{v}^q_t}+\epsilon}\,\hat{m}^q_t.
\end{aligned}
$$

where $\theta^k$ are the parameters of replica $k$, $g^q_t = 2\theta^k_t - \theta^{k+1}_t - \theta^{k-1}_t$ is the quantum (replica-coupling) gradient, $\eta$ is the learning rate, $\beta_1,\beta_2$ are the moment decay rates, $\rho_t$ is the coupling mass that increases over the schedule, and $\epsilon$ is for numerical stability.

Reference: Masayuki Ohzeki, Shuntaro Okada, Masayoshi Terabe, Shinichiro Taguchi, "Optimization of neural networks via finite-value quantum fluctuations", Scientific Reports 2018. https://www.nature.com/articles/s41598-018-28212-4

---
[Back to the Canon](../README.md)
