# EFSGD / EN-EFSGD

Implements EFSGD / EN-EFSGD, enhanced fractional stochastic gradient descent for matrix-factorization recommender systems with chaotic feedback.

The method augments the ordinary SGD step on a matrix-factorization objective $e_{ui}=r_{ui}-p_u^\top q_i$ with a fractional-order correction. The fractional gradient of the squared-error loss is obtained from the Caputo derivative (in EFSGD, evaluated through the Faà di Bruno formula), which introduces a long-memory term: a $1/\Gamma(2-v)$ weighting and the current latent factor raised to the power $1-v$, so that past rating history influences each update. Taking the fractional order $v\to 1$ recovers standard SGD.

EN-EFSGD adds an elastic-net penalty (combined $\ell_1$ and $\ell_2$ terms) to the latent factors, which performs feature selection over highly correlated user-item latent variables while retaining the fractional memory effect. The per-rating update for the user factor $p_u$ and item factor $q_i$ is

$$
\begin{aligned}
e_{ui} &= r_{ui} - p_u^\top q_i, \\
p_u &\leftarrow p_u + \eta\, e_{ui}\, q_i + \frac{\eta_{f}}{\Gamma(2-v)}\, e_{ui}\, q_i \odot |p_u|^{\,1-v} - \lambda_2 p_u - \lambda_1 \,\mathrm{sign}(p_u), \\
q_i &\leftarrow q_i + \eta\, e_{ui}\, p_u + \frac{\eta_{f}}{\Gamma(2-v)}\, e_{ui}\, p_u \odot |q_i|^{\,1-v} - \lambda_2 q_i - \lambda_1 \,\mathrm{sign}(q_i),
\end{aligned}
$$

where $p_u,q_i$ are the user and item latent vectors, $r_{ui}$ the observed rating, $e_{ui}$ the prediction error, $\eta$ the integer-order learning rate, $\eta_f$ the fractional learning rate, $v\in(0,1)$ the fractional order, $\Gamma$ the gamma function, $\odot$ element-wise product, and $\lambda_1,\lambda_2$ the elastic-net $\ell_1$ and $\ell_2$ regularization weights (EN-EFSGD only; plain EFSGD omits the $\lambda$ terms).

Reference: Zeshan Aslam Khan, Naveed Ishtiaq Chaudhary, Taimoor Ali Khan, Umair Farooq, Carla M. A. Pinto, Muhammad Asif Zahoor Raja, "Enhanced fractional prediction scheme for effective matrix factorization in chaotic feedback recommender systems", Chaos, Solitons & Fractals 2023. https://doi.org/10.1016/j.chaos.2023.114109

---
[Back to the Canon](../README.md)
