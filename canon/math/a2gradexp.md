# A2GradExp

Implements A2Grad (exponential variant), adaptive accelerated SGD.

Same accelerated coupling as `A2GradUni`, but the adaptive
accumulator is an exponential moving average of the gradient deviation,
kept monotone and scaled by the step count:


$$
\begin{aligned}
\tilde{v}_k &= \rho\, \tilde{v}_{k-1}
               + (1 - \rho)\, \lVert \delta_k \rVert^2 \\
v_k &= \max(v_{k-1}, \tilde{v}_k),
     \qquad h_k = \sqrt{(k + 1)\, v_k}
\end{aligned}
$$

Reference: Qi Deng, Yi Cheng, Guanghui Lan, "Optimal Adaptive and
Accelerated Stochastic Gradient Descent", arXiv 2018.
https://arxiv.org/abs/1810.00553

---
[Back to the Canon](../README.md)
