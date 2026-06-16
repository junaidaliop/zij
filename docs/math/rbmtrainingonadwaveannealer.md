# RBM training on a D-Wave annealer

Implements RBM training on a D-Wave annealer, contrastive gradient ascent whose model expectations are sampled from a quantum annealer instead of MCMC.

A restricted Boltzmann machine with visible units $v$, hidden units $h$, weights $W$, and biases $b,c$ has energy $E(v,h) = -b^\top v - c^\top h - v^\top W h$. Maximizing the log-likelihood of the data yields a gradient that is the difference between a data-dependent expectation and a model-dependent expectation, the latter requiring samples from the model's Boltzmann distribution.

The method replaces the expensive Markov-chain Monte Carlo estimate of the model term with samples drawn directly from a D-Wave quantum annealer, whose Ising hardware realizes the RBM Boltzmann distribution at an effective inverse temperature controlled by a scaling hyperparameter $S$. Parameters are then updated by gradient ascent:

$$
\begin{aligned}
\frac{\partial \log p(v)}{\partial W_{ij}} &= \langle v_i h_j \rangle_{\mathrm{data}} - \langle v_i h_j \rangle_{\mathrm{model}} \\
\frac{\partial \log p(v)}{\partial b_i} &= \langle v_i \rangle_{\mathrm{data}} - \langle v_i \rangle_{\mathrm{model}} \\
\frac{\partial \log p(v)}{\partial c_j} &= \langle h_j \rangle_{\mathrm{data}} - \langle h_j \rangle_{\mathrm{model}} \\
\theta_{t+1} &= \theta_t + \alpha\, \frac{\partial \log p(v)}{\partial \theta}
\end{aligned}
$$

where $\theta \in \{W, b, c\}$ are the RBM parameters, $\alpha$ is the learning rate, $\langle \cdot \rangle_{\mathrm{data}}$ is the expectation over the training data with hidden units inferred, and $\langle \cdot \rangle_{\mathrm{model}}$ is the expectation over the model distribution estimated from D-Wave annealer samples taken at scale $S$.

Reference: Vivek Dixit, Raja Selvarajan, Muhammad A. Alam, Travis S. Humble, Sabre Kais, "Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer", Frontiers in Physics 2021. https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.589626/full

---
[Back to the Canon](../index.md)
