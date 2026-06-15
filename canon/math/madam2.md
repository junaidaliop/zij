# M+Adam

Implements M+Adam, a low-precision optimizer that splits each weight into a mantissa-exponent pair and updates the two with different rules.

Writing a parameter elementwise as $w = m \cdot 2^e$, M+Adam applies an additive Adam step to the mantissa $m$ and an additive Madam step to the exponent $e$, then recombines. Because exponent moves act multiplicatively through the $2^e$ scaling, additive updates give fine intra-bin control while exponent updates traverse quantization bins, which keeps training stable in pure BF16 without FP32 master weights. At each step the weight gradient $g_t$ is projected onto the two components as $g_m = 2^e g_t$ and $g_e = (w \log 2)\, g_t$, and each update is clamped so its magnitude does not exceed a relative cap.

$$
\begin{aligned}
m_t^{(\cdot)} &\leftarrow \beta_1 m_{t-1}^{(\cdot)} + (1-\beta_1)\, g_\cdot, &
v_t^{(\cdot)} &\leftarrow \beta_2 v_{t-1}^{(\cdot)} + (1-\beta_2)\, g_\cdot^2 \\
m &\leftarrow m - \eta_m\, \mathrm{clamp}_{\eta_m^\star/\eta_m}\!\left(\frac{m_t^{(m)}}{\sqrt{v_t^{(m)}}+\epsilon}\right) \\
e &\leftarrow e - \eta_e\, \mathrm{clamp}_{\eta_e^\star/\eta_e}\!\left(\frac{m_t^{(e)}}{\sqrt{v_t^{(e)}}+\epsilon}\right) \\
w &\leftarrow \mathrm{clamp}_{w_{\max}}\!\left(m \cdot 2^e\right)
\end{aligned}
$$

where $\cdot \in \{m, e\}$ indexes the mantissa and exponent paths; $g_m = 2^e g_t$ and $g_e = (w \log 2)\, g_t$ are the projected gradients; $m_t^{(\cdot)}, v_t^{(\cdot)}$ are the first and second moments; $\eta_m, \eta_e$ are the mantissa and exponent learning rates; $\eta_m^\star, \eta_e^\star$ are the maximum per-step perturbations (the clamp bounds each ratio to $\pm\,\eta^\star/\eta$); $\beta_1, \beta_2$ are the moment decays; $\epsilon$ is for numerical stability; and $w_{\max}$ caps the recombined weight magnitude.

Reference: Xiaoyuan Liang, Sebastian Loeschcke, Mads Toftrup, Anima Anandkumar, "M+Adam: Stable Low-Precision Training with Combined Adam–Madam Updates", OPT2025: 17th Annual Workshop on Optimization for Machine Learning (NeurIPS workshop) 2025. https://opt-ml.org/papers/2025/paper141.pdf

---
[Back to the Canon](../README.md)
