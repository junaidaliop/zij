# GWT

Implements GWT (Gradient Wavelet Transform), a memory-efficient optimizer that applies Adam to the wavelet-compressed gradient.

GWT decomposes each gradient $g_t$ with a discrete Haar wavelet transform, splitting it into low-frequency approximation coefficients $A_t$ and high-frequency detail coefficients $D_t$. Adam's optimizer states $m_t$ and $v_t$ are maintained only over the smaller approximation part, which roughly halves the moment-buffer memory per level of decomposition. The normalized approximation is reconstructed together with the (state-free) detail coefficients via the inverse transform, scaled by a constant $\alpha$, and used to update the weights.

$$
\begin{aligned}
[A_t, D_t] &= g_t H \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) A_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) A_t^2 \\
\tilde{A}_t &= \frac{m_t}{\sqrt{v_t} + \epsilon}, \qquad \tilde{D}_t = \frac{D_t}{\sqrt{v_t} + \epsilon} \\
\tilde{g}_t &= \alpha \, [\tilde{A}_t, \tilde{D}_t] \tilde{H} \\
\theta_t &= \theta_{t-1} - \eta \, \tilde{g}_t
\end{aligned}
$$

where $H$ is the forward Haar wavelet matrix, $\tilde{H}$ its inverse ($H\tilde{H}=I$), $A_t$/$D_t$ the approximation/detail coefficients, $m_t$/$v_t$ the Adam moments over $A_t$, $\beta_1,\beta_2$ the decay rates, $\eta$ the learning rate, $\alpha$ a scale factor (default $0.25$), and $\epsilon$ a stability constant.

Reference: Ziqing Wen, Ping Luo, Jiahuan Wang, Xiaoge Deng, Jinping Zou, Kun Yuan, Tao Sun, Dongsheng Li, "Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training", arXiv 2025. https://arxiv.org/abs/2501.07237

---
[Back to the Canon](../index.md)
