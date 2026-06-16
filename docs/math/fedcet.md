# FedCET

Implements FedCET, a communication-efficient federated method achieving linear convergence on heterogeneous data.

FedCET runs an adapted-gradient-tracking recursion that uses the learning rate itself as the weighting mechanism to cancel client drift, so each communication round transmits only a single model-sized vector in each direction. Between communication rounds clients perform a momentum-style local recursion driven purely by successive gradient differences; at a communication round the server averages the clients' transmitted perturbations and mixes them back in.

Writing the per-client iterate as $\theta_i$, the local update (when $(t+1)\bmod\tau \neq 0$) and the equivalent compact recursion with correction variable $d_t$ are

$$
\begin{aligned}
\theta_i^{t+1} &= 2\theta_i^{t} - \theta_i^{t-1} - \eta\, g_i^{t} + \eta\, g_i^{t-1}, \\
d^{t+1} &= d^{t} + c\,(I - W^{t+1})\bigl(\theta^{t} - \eta\, g^{t} - \eta\, d^{t}\bigr), \\
\theta^{t+1} &= \theta^{t} - \eta\, g^{t} - \eta\, d^{t+1}, \qquad
d^{t} = \tfrac{1}{\eta}(\theta^{t-1} - \theta^{t}) - g^{t-1},
\end{aligned}
$$

where $\theta^{t}=[\theta_1^{t},\dots,\theta_N^{t}]$ stacks the $N$ client iterates, $g_i^{t}=\nabla f_i(\theta_i^{t})$ with $g^{t}$ the stacked gradients, $\eta$ is the learning rate, $c$ is the weighting (mixing) parameter, $\tau$ is the local-training period, $d^{t}$ is the drift-correction variable, and $W^{t+1}=\frac{1}{N}\mathbf{1}\mathbf{1}^{\top}$ at communication rounds ($t+1=\tau k$) and $W^{t+1}=I$ otherwise.

Reference: Jie Liu, Yongqiang Wang, "Communication Efficient Federated Learning with Linear Convergence on Heterogeneous Data", arXiv 2025. https://arxiv.org/abs/2503.15804

---
[Back to the Canon](../index.md)
