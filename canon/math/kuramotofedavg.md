# Kuramoto-FedAvg

Implements Kuramoto-FedAvg, a federated aggregation rule that reweights client updates by their phase synchronization with the mean update direction.

Drawing on the Kuramoto model of coupled oscillators $\dot{\theta}_i = \omega_i + \tfrac{K}{N}\sum_j \sin(\theta_j - \theta_i)$, each client's local update is assigned a phase given by its angular deviation from the weighted mean update direction. Aggregation weights are then derived from the sine of each client's offset relative to the mean phase, amplifying in-phase (well-aligned) contributions and damping misaligned ones to combat statistical heterogeneity across clients.

$$
\begin{aligned}
\theta_k^t &= \arccos\!\left(\frac{\langle \Delta w_k^t,\, \bar{\Delta}^t \rangle}{\lVert \Delta w_k^t \rVert \cdot \lVert \bar{\Delta}^t \rVert}\right),
\qquad \bar{\Delta}^t = \sum_{k=1}^{K} p_k\, \Delta w_k^t, \qquad \bar{\theta}^t = \frac{1}{K}\sum_{k=1}^{K} \theta_k^t \\
\rho_k^t &= \frac{\sin(\bar{\theta}^t - \theta_k^t)}{\sum_{j=1}^{K} \sin(\bar{\theta}^t - \theta_j^t)} \\
w^{t+1} &= w^t + \kappa_t \sum_{k=1}^{K} \rho_k^t\, \Delta w_k^t
\end{aligned}
$$

where $\Delta w_k^t$ is client $k$'s local update at round $t$, $p_k$ the client mixing weight, $\bar{\Delta}^t$ the weighted mean update direction, $\theta_k^t$ the client phase, $\bar{\theta}^t$ the mean phase, $\rho_k^t$ the synchronization-based aggregation weight, $\kappa_t$ a decaying synchronization-strength step size, and $w^t$ the global model.

Reference: Aggrey Muhebwa, Khotso Selialia, Fatima Anwar, Khalid K. Osman, "Kuramoto-FedAvg: Using Synchronization Dynamics to Improve Federated Learning Optimization under Statistical Heterogeneity", arXiv 2025. https://arxiv.org/abs/2505.19605

---
[Back to the Canon](../README.md)
