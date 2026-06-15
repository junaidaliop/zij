# SignMuon

Implements SignMuon, a communication-efficient Muon variant that takes the entrywise sign of the orthogonalized momentum.

SignMuon combines the polar (Newton-Schulz orthogonalization) step of Muon with the sign compression of signSGD. Momentum is accumulated and orthogonalized into a polar factor $U_t$, after which the update direction is reduced to its entrywise sign. In the distributed setting each worker transmits only these signs and they are aggregated by majority vote, but the core single-worker update is given below.

$$
\begin{aligned}
\tilde{g}_t &= g_t + \lambda \theta_t \\
m_{t} &= \beta\, m_{t-1} + (1-\beta)\, \tilde{g}_t \\
U_t &= \mathrm{PolarNS}(m_t) \\
\theta_{t+1} &= \theta_t - \eta_t\, \mathrm{sign}(U_t)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $m_t$ the momentum buffer, $\beta$ the momentum decay, $\lambda$ the weight decay, $\mathrm{PolarNS}(\cdot)$ the polar factor obtained by $K$ Newton-Schulz iterations (with stability constant $\epsilon$), and $\mathrm{sign}(\cdot)$ the entrywise sign.

Reference: Neel Mishra, Kushagara Trivedi, Pawan Kumar, "SignMuon: Communication-Efficient Distributed Muon Optimization", arXiv 2025. https://arxiv.org/abs/2605.16311

---
[Back to the Canon](../README.md)
