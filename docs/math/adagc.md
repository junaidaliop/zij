# AdaGC

Implements AdaGC, Adam with adaptive per-tensor gradient clipping.

AdaGC stabilizes large language model pretraining by clipping each tensor's
gradient against an exponential moving average of its own past clipped
gradient norms. During an initial warmup the clipping is global and the
threshold $\gamma$ tracks the running minimum of the clipped norms;
afterwards each tensor is clipped locally relative to its own history. The
clipped gradient then drives a standard Adam update.


$$
\begin{aligned}
     h_t &= \min\!\left(
         \frac{\lambda_{\text{rel}} \, \gamma_{t-1}}{\lVert g_t \rVert},
         1 \right)                                                        \\
     \hat{g}_t &= h_t \, g_t                                              \\
     \gamma_t &= \beta \gamma_{t-1}
         + (1 - \beta) \lVert \hat{g}_t \rVert                           \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \hat{g}_t                     \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \hat{g}_t^2                   \\
     \theta_t &= \theta_{t-1} - \eta \,
         \frac{m_t / (1 - \beta_1^t)}
              {\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
\end{aligned}
$$

where $g_t$ is the gradient, $\hat{g}_t$ the clipped gradient,
$\gamma_t$ the per-tensor exponential moving average of clipped norms,
$\lambda_{\text{rel}}$ the relative clipping threshold, $\beta$
the smoothing coefficient, and $m_t$, $v_t$ the Adam moments.
During the first `warmup_steps` iterations the clipping factor uses the
absolute threshold, $h_t = \min(\lambda_{\text{abs}} / \lVert g_t
\rVert, 1)$, and $\gamma_t = \min(\gamma_{t-1}, \lVert \hat{g}_t
\rVert)$.

Reference: Guoxia Wang, Shuai Li, Congliang Chen, Jinle Zeng, Jiabin Yang,
Dianhai Yu, Yanjun Ma, Li Shen, "AdaGC: Enhancing LLM Pretraining Stability
via Adaptive Gradient Clipping", ICML 2026.
https://arxiv.org/abs/2502.11034

---
[Back to the Canon](../index.md)
