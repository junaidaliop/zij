# GreedyLR

Implements GreedyLR, a loss-feedback learning-rate scheduler that greedily raises the rate while the loss keeps improving and lowers it while the loss stalls.

GreedyLR tracks a (optionally smoothed) loss signal and counts consecutive epochs of improvement and deterioration. Once a run of good epochs exceeds the patience, the rate is multiplied up toward an upper bound; once a run of bad epochs exceeds the patience, the rate is multiplied down toward a lower bound. Improvement is judged against the best observed loss up to a tolerance $\delta$, and the rate is always clamped to $[\eta_{\min}, \eta_{\max}]$. In its simplest form (no patience, no smoothing) the rule reduces to dividing the rate by a factor $F \in (0,1)$ when the loss drops and multiplying by $F$ when it does not.

$$
\begin{aligned}
\ell_t &= \mathrm{smooth}(L_t), \qquad \text{best}_t = \min_{s \le t} \ell_s \\
n^{+}_t &= \begin{cases} n^{+}_{t-1} + 1 & \ell_t < \text{best}_{t-1} - \delta \\ 0 & \text{otherwise} \end{cases}, \qquad
n^{-}_t = \begin{cases} n^{-}_{t-1} + 1 & \ell_t > \text{best}_{t-1} + \delta \\ 0 & \text{otherwise} \end{cases} \\
\eta_t &= \begin{cases} \min(\eta_{t-1}\, F,\ \eta_{\max}) & n^{+}_t > p \\ \max(\eta_{t-1}\, F,\ \eta_{\min}) & n^{-}_t > p \\ \eta_{t-1} & \text{otherwise} \end{cases}
\end{aligned}
$$

where $\ell_t$ is the streaming-averaged loss at epoch $t$, $\text{best}_t$ the running best loss, $\delta$ the improvement threshold, $n^{+}_t$ and $n^{-}_t$ the counts of consecutive good and bad epochs, $p$ the patience, $F$ the multiplicative factor ($F>1$ for the increase branch, $F<1$ for the decrease branch), and $[\eta_{\min},\eta_{\max}]$ the learning-rate bounds. Warmup and cooldown counters extend a run of increases or decreases for a fixed number of epochs after the loss trend reverses.

Reference: Shreyas Subramanian, Bala Krishnamoorthy, Pranav Murthy, "Dynamic Learning Rate Scheduling based on Loss Changes Leads to Faster Convergence", arXiv 2025. https://arxiv.org/abs/2512.14527

---
[Back to the Canon](../README.md)
