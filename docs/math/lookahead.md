# Lookahead

Implements Lookahead, "k steps forward, 1 step back" around any optimizer.

Lookahead keeps two sets of weights. The fast weights $\theta$ are
advanced for $k$ inner steps by a base optimizer, after which the slow
weights $\phi$ are pulled toward them by interpolation, and the fast
weights are reset to the slow ones:


$$
\begin{aligned}
\theta_{t,0} &= \phi_{t-1} \\
\theta_{t,i} &= \theta_{t,i-1} + A(L, \theta_{t,i-1}, d), \quad i = 1, \dots, k \\
\phi_t &= \phi_{t-1} + \alpha\,(\theta_{t,k} - \phi_{t-1}) \\
\theta_{t+1,0} &= \phi_t
\end{aligned}
$$

where $A$ is the inner optimizer's update on minibatch $d$,
$\alpha$ is the slow-weights step size, and $k$ is the
synchronization period.

Reference: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba,
"Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
https://arxiv.org/abs/1907.08610


**Note:** this is a wrapper around a base optimizer. Pass an already constructed

optimizer instance, e.g.
`Lookahead(torch.optim.Adam(model.parameters(), lr=1e-3), k=5, alpha=0.5)`.

---
[Back to the Canon](../index.md)
