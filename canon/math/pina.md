# PINA

Implements PINA, a two-stage differentially private clustered federated learning method with privacy-preserving initialization and normality-driven aggregation.

PINA targets the server-side aggregation step of federated learning under differential privacy. Each sampled client returns the difference $\Delta_k$ between its locally trained model and the global model, clips it to a fixed $L_2$ norm, and the server forms a noised secure sum to update the global weights. The normality-driven aggregation reweights each cluster's update by its Shapiro-Wilk statistic, zeroing out clusters whose updates fail the normality test, while a separate clipping threshold is used during the privacy-preserving initialization stage.

$$
\begin{aligned}
\Delta_k^t &= W_k^t - W^t, \\
\mathrm{Clip}(\Delta_k^t) &= \Delta_k^t \cdot \min\!\left(1, \frac{S}{\lVert \Delta_k^t \rVert_2}\right), \\
W^{t+1} &= W^t + \frac{1}{\lvert \mathcal{K}^t \rvert}\left( \sum_{k \in \mathcal{K}^t} \mathrm{Clip}(\Delta_k^t) + \mathcal{N}(0, \sigma^2 I) \right), \qquad \sigma = z\,S.
\end{aligned}
$$

where $W^t$ is the global model at round $t$, $W_k^t$ is client $k$'s locally trained model, $\mathcal{K}^t$ is the randomly sampled set of clients, $S$ is the clipping threshold, $z$ is the noise multiplier set via the Renyi differential privacy moments accountant, and $\mathcal{N}(0, \sigma^2 I)$ is the added Gaussian noise.

Reference: Jie Xu, Haaris Mehmood, Rogier Van Dalen, Karthikeyan Saravanan, Mete Ozay, "Differentially Private Clustered Federated Learning with Privacy-Preserving Initialization and Normality-Driven Aggregation", ICASSP 2026. https://arxiv.org/abs/2604.20596

---
[Back to the Canon](../README.md)
