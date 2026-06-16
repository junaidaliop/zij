# FLeNS

Implements FLeNS (Federated Learning with Enhanced Nesterov-Newton Sketch), a communication-efficient second-order federated optimizer combining Nesterov acceleration with a sketched Hessian Newton step.

FLeNS targets the communication and computation cost of exact federated Newton methods. Rather than sending full $d \times d$ Hessians, each client compresses its local Hessian with a random sketch $S_j \in \mathbb{R}^{k \times d}$ ($k \ll d$), evaluated at a Nesterov-accelerated lookahead point $v_t$ to speed convergence. The server aggregates the sketched Hessians and gradients with data-proportional weights $n_j/N$ and applies a single global Newton step, yielding super-linear convergence in communication rounds without transmitting full second-order information.

$$
\begin{aligned}
v_t &= w_t + \beta_t\,(w_t - w_{t-1}) \\
g_{D_j,t} &= \nabla L_{D_j}(v_t), \qquad \tilde{H}_{D_j,t} = S_j^{\top}\, \nabla^2 L_{D_j}(v_t)\, S_j \\
g_{D,t} &= \sum_{j=1}^{m} \frac{n_j}{N}\, g_{D_j,t}, \qquad \tilde{H}_{D,t} = \sum_{j=1}^{m} \frac{n_j}{N}\, \tilde{H}_{D_j,t} \\
w_{t+1} &= w_t - \mu\, \tilde{H}_{D,t}^{-1}\, g_{D,t}
\end{aligned}
$$

where $w_t$ are the global model parameters, $v_t$ the Nesterov lookahead point, $\beta_t$ the momentum parameter, $S_j$ the per-client sketch matrix, $\tilde{H}_{D_j,t}$ the local sketched Hessian and $g_{D_j,t}$ the local gradient (both evaluated at $v_t$), $n_j$ the number of samples at client $j$ with $N=\sum_j n_j$, $m$ the number of clients, and $\mu$ the global step size.

Reference: Sunny Gupta, Mohit Jindal, Pankhi Kashyap, Pranav Jeevan, Amit Sethi, "FLeNS: Federated Learning with Enhanced Nesterov-Newton Sketch", arXiv 2024. https://arxiv.org/abs/2409.15216

---
[Back to the Canon](../index.md)
