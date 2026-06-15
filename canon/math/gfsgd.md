# GFSGD

Implements GFSGD, a generalized fractional stochastic gradient descent for matrix-factorization recommender systems.

GFSGD augments the standard SGD update for the user and item latent factors with a Caputo-type fractional-derivative term. The fractional term adds a memory of the parameter's own magnitude through a power law, which captures the history of chaotic ratings and broadens the usable range of the fractional order. The fractional order $\alpha$ controls the memory strength: at $\alpha = 1$ the fractional term collapses to the ordinary gradient and GFSGD recovers plain SGD, while larger fractional contributions accelerate convergence.

For a rating $r_{ij}$ with prediction error $e_{ij} = r_{ij} - p_i^\top q_j$, the user factor $p_i$ and item factor $q_j$ are updated alternately by

$$
\begin{aligned}
p_i &\leftarrow p_i + \gamma\, e_{ij}\, q_j + \frac{\gamma_f}{\Gamma(2-\alpha)}\, e_{ij}\, q_j \odot |p_i|^{\,1-\alpha}, \\
q_j &\leftarrow q_j + \gamma\, e_{ij}\, p_i + \frac{\gamma_f}{\Gamma(2-\alpha)}\, e_{ij}\, p_i \odot |q_j|^{\,1-\alpha},
\end{aligned}
$$

where $\gamma$ is the integer-order learning rate, $\gamma_f$ is the fractional-order learning rate, $\alpha$ is the fractional order, $\Gamma$ is the gamma function, $\odot$ is the elementwise product, and $|\cdot|^{1-\alpha}$ is applied componentwise to the latent factor.

Reference: Zeshan Aslam Khan, Naveed Ishtiaq Chaudhary, Muhammad Asif Zahoor Raja, "Generalized fractional strategy for recommender systems with chaotic ratings behavior", Chaos, Solitons & Fractals 2022. https://doi.org/10.1016/j.chaos.2022.112204

---
[Back to the Canon](../README.md)
