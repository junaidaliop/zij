# SOAP

Implements SOAP, Shampoo with Adam in the Preconditioner's eigenbasis.

SOAP keeps a Shampoo preconditioner for each tensor dimension as a running
average of one-sided gradient outer products. Let $L_t$ and
$R_t$ be these preconditioners for a matrix parameter, with
eigenbases $Q_L$ and $Q_R$ refreshed every
`precondition_frequency` steps. The gradient is rotated into that
eigenbasis, Adam runs on the rotated coordinates, and the resulting update
is rotated back:


$$
\begin{aligned}
L_t &= \beta_2 L_{t-1} + (1 - \beta_2)\, G_t G_t^\top \\
R_t &= \beta_2 R_{t-1} + (1 - \beta_2)\, G_t^\top G_t \\
\tilde{G}_t &= Q_L^\top G_t Q_R \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde{G}_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \tilde{G}_t^2 \\
\theta_t &= \theta_{t-1}
    - \gamma\, \frac{\sqrt{\mathrm{bc}_2}}{\mathrm{bc}_1}\, Q_L
    \frac{m_t}{\sqrt{v_t} + \epsilon} Q_R^\top
    - \gamma\, \lambda\, \theta_{t-1}
\end{aligned}
$$

where $m_t$ and $v_t$ are the Adam moments of the rotated
gradient, $\mathrm{bc}_1 = 1 - \beta_1^t$ and
$\mathrm{bc}_2 = 1 - \beta_2^t$ are the bias-correction terms folded
into the scalar step size. Following the HuggingFace AdamW convention,
$\epsilon$ is added to the un-bias-corrected second moment
$\sqrt{v_t}$ (rather than to $\sqrt{\hat{v}_t}$), which yields
an effective denominator of
$\sqrt{\hat{v}_t} + \epsilon / \sqrt{\mathrm{bc}_2}$. The final
term $-\gamma\,\lambda\,\theta_{t-1}$ is decoupled weight decay
($\lambda$ = `weight_decay`), applied after the gradient step and
scaled by the raw learning rate $\gamma$.

Reference: Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira,
David Brandfonbrener, Lucas Janson, Sham Kakade,
"SOAP: Improving and Stabilizing Shampoo using Adam for Language
Modeling", ICLR 2025.
https://arxiv.org/abs/2409.11321

---
[Back to the Canon](../README.md)
