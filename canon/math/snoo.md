# SNOO

Implements SNOO, an outer optimizer that applies Nesterov momentum to the pseudo-gradients of an inner optimizer.

SNOO (Step-K Nesterov Outer Optimizer) is a two-loop scheme. An inner optimizer (AdamW, Muon, etc.) runs for $K$ steps on fast weights $\tilde w$, and the displacement of the fast weights over those $K$ steps defines a pseudo-gradient $s_t$. The slow weights are then advanced by applying Nesterov momentum to this pseudo-gradient. With outer momentum $\mu = 0$ the method reduces exactly to Lookahead.

$$
\begin{aligned}
\tilde w_{t,0} &= w_t \\
\tilde w_{t,k+1} &= \tilde w_{t,k} - \tilde\eta_{t,k}\, \mathcal{T}_{t,k}(f, \tilde w_{t,k}; \xi_{t,k}), \qquad k = 0, \dots, K-1 \\
s_t &= w_t - \tilde w_{t,K} \\
b_t &= \mu\, b_{t-1} + s_t \\
w_{t+1} &= w_t - \eta\,(\mu\, b_t + s_t)
\end{aligned}
$$

where $w_t$ are the slow (outer) weights, $\tilde w_{t,k}$ the fast (inner) weights at inner step $k$, $\mathcal{T}_{t,k}$ the inner optimizer's update map applied to minibatch $\xi_{t,k}$ with inner learning rate $\tilde\eta_{t,k}$, $s_t$ the pseudo-gradient (trajectory displacement over $K$ inner steps), $b_t$ the outer Nesterov momentum buffer, $\eta$ the outer learning rate, $\mu$ the outer momentum coefficient, and $K$ the outer step frequency.

Reference: Dominik Kallusky, Vinay Rao, Vishal Nandavanam, Hao-Jun Michael Shi, "SNOO: Step-K Nesterov Outer Optimizer - The Surprising Effectiveness of Nesterov Momentum Applied to Pseudo-Gradients", arXiv 2025. https://arxiv.org/abs/2510.15830

---
[Back to the Canon](../README.md)
