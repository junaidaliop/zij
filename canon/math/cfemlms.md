# CFEM-LMS

Implements CFEM-LMS, a convex combination of the LMS filter and a fractional-order error-modified LMS filter for identifying Van der Pol-Duffing oscillator systems.

Built on a functional-link (FLANN) structure, the method runs two adaptive filters in parallel. One is the ordinary least mean square (LMS) filter; the other, FEM-LMS, replaces the squared-error cost with an error-function (erf) cost that saturates large errors and then takes the $v$-order Caputo fractional gradient, yielding an extra fractional term whose nonlocality improves identification of the nonlinear dynamics. The two weight vectors are mixed by a sigmoid-controlled parameter $\lambda_t$ so the overall filter inherits the fast convergence of LMS and the low steady-state error of FEM-LMS.

Each branch updates with a first-order gradient term plus a Caputo fractional term; the Caputo derivative of $w^{1-v}$ contributes the $|w_t|^{1-v}/\Gamma(2-v)$ factor. The combination weight follows from a sigmoid of an auxiliary scalar $a_t$ adapted to minimize the overall squared error.

$$
\begin{aligned}
e_t &= d_t - w_t^\top x_t, \\
w^{(1)}_{t+1} &= w^{(1)}_t + \gamma\, e_t\, x_t, \\
w^{(2)}_{t+1} &= w^{(2)}_t + \gamma\, \psi(e_t)\, x_t + \gamma_f\, \psi(e_t)\, x_t \odot \frac{|w^{(2)}_t|^{\,1-v}}{\Gamma(2-v)}, \\
\lambda_t &= \frac{1}{1 + e^{-a_t}}, \\
w_{t+1} &= \lambda_t\, w^{(1)}_{t+1} + (1-\lambda_t)\, w^{(2)}_{t+1}.
\end{aligned}
$$

where $w^{(1)}$ is the LMS branch and $w^{(2)}$ the FEM-LMS branch, $x_t$ is the FLANN-expanded input, $d_t$ the desired output, $e_t$ the error, $\psi(\cdot)$ the erf-based saturation nonlinearity of the modified cost, $\gamma$ and $\gamma_f$ the integer- and fractional-order step sizes, $v\in(0,1)$ the fractional order, $\Gamma$ the gamma function, $\lambda_t\in(0,1)$ the mixing parameter from sigmoid of auxiliary variable $a_t$, and $\odot$ elementwise product.

Reference: Kai-Li Yin, Yi-Fei Pu, Lu Lu, "Combination of fractional FLANN filters for solving the Van der Pol-Duffing oscillator", Neurocomputing 399 (2020) 183-192. https://doi.org/10.1016/j.neucom.2020.02.022

---
[Back to the Canon](../README.md)
