# Fractional-order FL with adaptive momentum

Implements FOFedAvg, a federated averaging scheme whose client steps use fractional-order SGD with a memory-aware power-law step size.

FOFedAvg replaces the local SGD step of FedAvg with Fractional-Order SGD (FOSGD), derived from a Caputo derivative of order $\alpha\in(0,1]$. Truncating the Caputo expansion to its leading term turns the fractional derivative into a scalar modulation of the gradient: the most recent parameter displacement is raised to the power $1-\alpha$ and divided by $\Gamma(2-\alpha)$, compressing the past trajectory into a single memory term. Larger between-round changes inflate the effective step, while smaller ones lean on accumulated history, giving a non-local, history-dependent update that tempers client drift under non-IID data.

Each round the server samples a client subset $S_t$; every client $k$ runs the FOSGD step on its mini-batches, then the server forms the new global model by a data-size-weighted average of the returned parameters.

$$
\begin{aligned}
\theta_{t+1}^{(k)} &= \theta_t^{(k)} - \frac{\mu_t}{\Gamma(2-\alpha)}\,\left(\left\lVert \theta_t^{(k)} - \theta_{t-1}^{(k)}\right\rVert + \delta\right)^{1-\alpha} g_t^{(k)}, \\
\mu_t &= \frac{\mu_0}{\sqrt{t+1}}, \\
\theta_{t+1} &= \sum_{k\in S_t} \frac{n_k}{n}\,\theta_{t+1}^{(k)}
\end{aligned}
$$

where $\theta^{(k)}$ are client $k$'s parameters, $g_t^{(k)}=\nabla \ell(\theta_t^{(k)};b)$ is the local mini-batch gradient, $\mu_t$ is the decaying learning rate with base $\mu_0$, $\alpha\in(0,1]$ is the fractional order, $\Gamma(\cdot)$ is the gamma function, $\delta>0$ guards the displacement term against vanishing steps, $n_k$ is the number of samples on client $k$, and $n=\sum_{k\in S_t} n_k$ is the total over participating clients.

Reference: Mohammad Partohaghighi, Roummel Marcia, YangQuan Chen, "Fractional-Order Federated Learning", arXiv 2026. https://arxiv.org/abs/2602.15380

---
[Back to the Canon](../README.md)
