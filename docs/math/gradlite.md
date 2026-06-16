# GradLite

Implements GradLite, a backward-friendly optimizer that trains under memory constraints using approximate low-rank gradients with error feedback.

To avoid caching full activations during backpropagation, GradLite reconstructs each gradient from a low-rank factorization of the layer Jacobian, projecting the top-level error signal through compact basis matrices. Because the reconstructed gradient is biased, an error accumulator carries the discarded residual forward and folds it into the next step, so the bias is corrected over time rather than discarded.

$$
\begin{aligned}
\tilde{g}_t &= V_t\left(U_t^\top \delta_t\right) \\
\theta_{t+1} &= \theta_t - \eta\left(\tilde{g}_t + r_t\right) \\
r_{t+1} &= r_t + \left(g_t - \tilde{g}_t\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the exact gradient, $\tilde{g}_t$ is its low-rank approximation, $\delta_t$ is the top-level error signal, $U_t$ and $V_t$ are the rank-$k$ basis matrices factoring the Jacobian, and $r_t$ is the error-feedback accumulator holding the running approximation residual.

Reference: Jing Yang, Kaitong Cai, Yijia Fan, Yufeng Yang, Keze Wang, "Backward-Friendly Optimization: Training Large Language Models with Approximate Gradients under Memory Constraints", arXiv 2025. https://arxiv.org/abs/2510.22467

---
[Back to the Canon](../index.md)
