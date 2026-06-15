# FBPTT

Implements FBPTT (Fractional Back-Propagation Through Time), a fractional-calculus learning rule for recurrent neural networks.

FBPTT replaces the integer-order gradient of standard back-propagation through time with a fractional-order one obtained from the Caputo derivative of the quadratic cost. Following the fractional LMS construction of the same authors, the weight update keeps the conventional first-order term and adds a fractional term: the ordinary gradient is multiplied by the Caputo fractional derivative of the weight, which for a power evaluates to $|w|^{1-\nu}/\Gamma(2-\nu)$. The fractional order $\nu \in (0,1)$ tunes how much memory of the weight magnitude enters the step; setting $\nu = 1$ recovers ordinary BPTT.

For a weight $w$ with conventional cost gradient $g_t = \partial J / \partial w$ computed by unrolling the network through time, the update is

$$
\begin{aligned}
\nabla^{\nu}_{w} J &= g_t \cdot \frac{|w_t|^{\,1-\nu}}{\Gamma(2-\nu)} \\
w_{t+1} &= w_t - \eta\, g_t - \eta_f\, \nabla^{\nu}_{w} J \\
&= w_t - \eta\, g_t - \frac{\eta_f}{\Gamma(2-\nu)}\, g_t \,|w_t|^{\,1-\nu}
\end{aligned}
$$

where $w$ is a network weight, $\eta$ and $\eta_f$ are the conventional and fractional step sizes, $g_t = \partial J/\partial w$ is the standard BPTT gradient of the quadratic error $J$, $\nu \in (0,1)$ is the fractional order, $\Gamma(\cdot)$ is the gamma function, and $\nabla^{\nu}_{w} J$ is the Caputo fractional gradient of the cost. The factor $|w_t|^{1-\nu}/\Gamma(2-\nu)$ is the Caputo fractional derivative of the weight applied through the chain rule.

Reference: Shujaat Khan, Jawwad Ahmad, Imran Naseem, Muhammad Moinuddin, "A Novel Fractional Gradient-Based Learning Algorithm for Recurrent Neural Networks", Circuits, Systems, and Signal Processing 2018. https://doi.org/10.1007/s00034-017-0572-z

---
[Back to the Canon](../README.md)
