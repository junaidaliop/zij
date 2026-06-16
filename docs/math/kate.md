# KATE

Implements KATE, a scale-invariant version of AdaGrad that removes the square root from the denominator.

AdaGrad scales each coordinate by $1/\sqrt{b_t}$ where $b_t$ accumulates squared gradients. KATE keeps the running sum $b_t^2$ but drops the square root in the update, dividing by $b_t^2$ instead. To preserve a sensible effective step size it tracks a separate numerator sequence $m_t$ that accumulates a tunable multiple of $g_t^2$ plus the normalized term $g_t^2/b_t^2$. The resulting per-coordinate step is invariant to a rescaling of the objective, and the method matches AdaGrad's convergence guarantees without the square-root preconditioner.

$$
\begin{aligned}
b_t^2 &= b_{t-1}^2 + g_t^2 \\
m_t^2 &= m_{t-1}^2 + \eta\, g_t^2 + \frac{g_t^2}{b_t^2} \\
\theta_{t+1} &= \theta_t - \gamma\, \frac{m_t}{b_t^2}\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ is the step size, $g_t$ is the (stochastic) gradient, $b_t$ is the running AdaGrad-style accumulator, $m_t$ is the numerator sequence, $\eta$ is a per-coordinate hyperparameter, and all operations (squaring, division, multiplication) are element-wise with $b_{-1} = m_{-1} = 0$.

Reference: Sayantan Choudhury, Nazarii Tupitsa, Nicolas Loizou, Samuel Horváth, Martin Takáč, Eduard Gorbunov, "Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad", arXiv 2024. https://arxiv.org/abs/2403.02648

---
[Back to the Canon](../index.md)
