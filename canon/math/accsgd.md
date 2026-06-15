# AccSGD

Implements AccSGD, an accelerated stochastic gradient method.

AccSGD couples a short, plain SGD step with a long, momentum-like step and
blends the two iterates each update. With $\eta$ the learning rate,
$\kappa$ the long-to-short step ratio, $\xi$ the statistical
advantage parameter, and a constant $0 < c \le 1$, the derived
coefficients are


$$
\begin{aligned}
     \alpha &= 1 - \frac{c^2\,\xi}{\kappa}, \qquad
     \beta = 1 - \alpha, \qquad
     \zeta = \frac{c}{c + \beta},                                        \\
     \tilde{w}_t &= \beta\Big[(\tfrac{1}{\beta} - 1)\,\tilde{w}_{t-1}
         - \tfrac{\eta\kappa}{c}\,g_t + \theta_{t-1}\Big],               \\
     \theta_t &= \zeta\,(\theta_{t-1} - \eta\,g_t)
         + (1 - \zeta)\,\tilde{w}_t,
\end{aligned}
$$

where $\tilde{w}_t$ is the accelerated running iterate, initialized to
$\theta_0$.

Reference: Prateek Jain, Sham M. Kakade, Rahul Kidambi, Praneeth Netrapalli,
Aaron Sidford, "Accelerating Stochastic Gradient Descent For Least Squares
Regression", COLT 2018. https://arxiv.org/abs/1704.08227
Companion analysis: Rahul Kidambi, Praneeth Netrapalli, Prateek Jain,
Sham M. Kakade, "On the insufficiency of existing momentum schemes for
Stochastic Optimization", ICLR 2018. https://arxiv.org/abs/1803.05591
Reference implementation: https://github.com/rahulkidambi/AccSGD

---
[Back to the Canon](../README.md)
