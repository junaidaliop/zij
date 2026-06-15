# MuonClip

Implements MuonClip, the Muon optimizer augmented with QK-Clip to bound attention logits during large-scale training.

MuonClip keeps the standard Muon update — momentum on the gradient, orthogonalized by a Newton–Schulz iteration and scaled to match RMS — but adds a post-update rescaling step called QK-Clip. After each step, any attention head whose maximum logit $S^h_{\max}$ exceeds a threshold $\tau$ has its query/key projection weights shrunk by a per-head factor $\gamma_h$, capping logit growth without changing the forward or backward pass. For the MLA layout, the non-rotary query and key components $q^C, k^C$ are each scaled by $\sqrt{\gamma_h}$ and the per-head rotary query $q^R$ by $\gamma_h$, while the shared rotary key $k^R$ is left untouched so the clip does not couple across heads.

$$
\begin{aligned}
M_t &= \mu M_{t-1} + g_t \\
O_t &= \mathrm{NewtonSchulz}(M_t)\,\sqrt{\max(n,m)}\,\cdot 0.2 \\
\theta_t &= \theta_{t-1} - \eta\,(O_t + \lambda\,\theta_{t-1}) \\
S^h_{\max} &= \frac{1}{\sqrt{d}}\max_{X\in B}\,\max_{i,j} Q^h_i (K^h_j)^{\top} \\
\gamma_h &= \min\!\Big(1,\ \frac{\tau}{S^h_{\max}}\Big) \\
q^C \leftarrow \sqrt{\gamma_h}\,q^C,\quad
&k^C \leftarrow \sqrt{\gamma_h}\,k^C,\quad
q^R \leftarrow \gamma_h\,q^R
\end{aligned}
$$

where $\theta$ are the parameters (weight matrix of shape $n\times m$), $\eta$ the learning rate, $g_t$ the gradient, $M_t$ the momentum buffer, $\mu$ the momentum coefficient, $\lambda$ the weight decay, and $\mathrm{NewtonSchulz}(\cdot)$ the orthogonalization iteration. $S^h_{\max}$ is the largest attention logit for head $h$ over batch $B$ (with $Q^h_i, K^h_j$ the query/key for tokens $i,j$ and $d$ the head dimension), $\tau$ the logit threshold, and $\gamma_h$ the resulting per-head clip factor applied to the query/key components $q^C, k^C, q^R$.

Reference: Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2025. https://arxiv.org/abs/2507.20534

---
[Back to the Canon](../README.md)
