# Memory-Efficient Optimizers

Memory-efficient optimizers reduce the optimizer-state memory that dominates large-model training budgets, where Adam-style methods store two extra full-precision values per parameter. The methods below cover factored second moments, 8-bit and 4-bit state quantization, low-rank gradient projection, block-coordinate updates, zeroth-order gradient estimates, and stateless update rules.

| Optimizer | Venue | Paper | Code | `zij` |
|---|---|---|---|---|
| [Adafactor](math/adafactor.md) | ICML 2018 | [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235) | [official](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/adafactor.py) | `Adafactor` |
| [SM3](math/sm3.md) | NeurIPS 2019 | [Memory-Efficient Adaptive Optimization](https://arxiv.org/abs/1901.11150) | [official](https://github.com/google-research/google-research/tree/master/sm3) | `SM3` |
| [8-bit Optimizers](math/8bitoptimizers.md) | ICLR 2022 | [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) | [official](https://github.com/bitsandbytes-foundation/bitsandbytes) | — |
| [tpSGD](math/tpsgd.md) | arXiv 2022 | [Learning with Local Gradients at the Edge](https://arxiv.org/abs/2208.08503) | — | — |
| [4-bit Optimizers](math/4bitoptimizers.md) | NeurIPS 2023 | [Memory Efficient Optimizers with 4-bit States](https://arxiv.org/abs/2309.01507) | [official](https://github.com/thu-ml/low-bit-optimizers) | — |
| [Adalite](math/adalite.md) | GitHub 2023 | [Adalite: a custom optimizer based on Adafactor and LAMB](https://github.com/euclaise/SlimTrainer) | [official](https://github.com/euclaise/SlimTrainer) | — |
| [AdaLomo](math/adalomo.md) | ACL 2024 Findings | [AdaLomo: Low-memory Optimization with Adaptive Learning Rate](https://arxiv.org/abs/2310.10195) | [official](https://github.com/OpenLMLab/LOMO) | `AdaLomo` |
| [CAME](math/came.md) | ACL 2023 | [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047) | [official](https://github.com/yangluo7/CAME) | `CAME` |
| [Lion](math/lion-v.md) | NeurIPS 2023 | [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) | [official](https://github.com/google/automl/tree/master/lion) | — |
| [LOMO](math/lomo.md) | ACL 2024 | [Full Parameter Fine-tuning for Large Language Models with Limited Resources](https://arxiv.org/abs/2306.09782) | [official](https://github.com/OpenLMLab/LOMO) | `Lomo` |
| [MeZO](math/mezo.md) | NeurIPS 2023 | [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333) | [official](https://github.com/princeton-nlp/MeZO) | — |
| [Tiger](math/tiger.md) | GitHub 2023 | [Tiger: A Tight-fisted Optimizer](https://github.com/bojone/tiger/blob/main/README_en.md#citation) | [official](https://github.com/bojone/tiger) | `Tiger` |
| [4-bit Shampoo](math/4bitshampoo.md) | NeurIPS 2024 | [4-bit Shampoo for Memory-Efficient Network Training](https://arxiv.org/abs/2405.18144) | [official](https://github.com/Sike-Wang/low-bit-Shampoo) | — |
| [Adam-mini](math/adammini.md) | ICLR 2025 | [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793) | [official](https://github.com/zyushun/Adam-mini) | `AdamMini` |
| [Adapprox](math/adapprox.md) | arXiv 2024 | [Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices](https://arxiv.org/abs/2403.14958) | — | — |
| [AdaRankGrad](math/adarankgrad.md) | ICLR 2025 | [AdaRankGrad: Adaptive Gradient-Rank and Moments for Memory-Efficient LLMs Training and Fine-Tuning](https://arxiv.org/abs/2410.17881) | — | — |
| [Addax](math/addax.md) | ICLR 2025 | [Addax: Utilizing Zeroth-Order Gradients to Improve Memory Efficiency and Performance of SGD for Fine-Tuning Language Models](https://arxiv.org/abs/2410.06441) | [official](https://github.com/optimization-for-data-driven-science/Addax) | — |
| [APOLLO](math/apollo.md) | MLSys 2025 | [APOLLO: SGD-like Memory, AdamW-level Performance](https://arxiv.org/abs/2412.05270) | [official](https://github.com/zhuhanqing/APOLLO) | `APOLLO` |
| [BAdam](math/blockoptimizer.md) | NeurIPS 2024 | [BAdam: A Memory Efficient Full Parameter Optimization Method for Large Language Models](https://arxiv.org/abs/2404.02827) | [official](https://github.com/Ledzy/BAdam) | `BlockOptimizer` |
| [COAP](math/coap.md) | CVPR 2025 | [COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection](https://arxiv.org/abs/2412.00071) | [official](https://github.com/bytedance/coap) | — |
| [Fira](math/firaadamw.md) | NeurIPS 2025 | [Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?](https://arxiv.org/abs/2410.01623) | [official](https://github.com/xichen-fy/Fira) | `FiraAdamW` |
| [Flora](math/flora.md) | ICML 2024 | [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://arxiv.org/abs/2402.03293) | [official](https://github.com/BorealisAI/flora-opt) | — |
| [FRUGAL](math/frugal.md) | ICML 2025 | [FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training](https://arxiv.org/abs/2411.07837) | [official](https://github.com/fzmushko/FRUGAL) | — |
| [GaLore](math/galoreadamw.md) | ICML 2024 | [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) | [official](https://github.com/jiaweizzhao/GaLore) | `GaLoreAdamW` |
| [GoLore](math/golore.md) | ICML 2025 | [Subspace Optimization for Large Language Models with Convergence Guarantees](https://arxiv.org/abs/2410.11289) | [official](https://github.com/pkumelon/Golore) | — |
| [GRASS](math/grass.md) | EMNLP 2024 | [Grass: Compute Efficient Low-Memory LLM Training with Structured Sparse Gradients](https://arxiv.org/abs/2406.17660) | [official](https://github.com/aashiqmuhamed/GRASS) | — |
| [LDAdam](math/ldadamw.md) | ICLR 2025 | [LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics](https://arxiv.org/abs/2410.16103) | [official](https://github.com/IST-DASLab/LDAdam) | `LDAdamW` |
| [LoQT](math/loqt.md) | NeurIPS 2024 | [LoQT: Low-Rank Adapters for Quantized Pretraining](https://arxiv.org/abs/2405.16528) | [official](https://github.com/sebulo/LoQT) | — |
| [LoRA-RITE](math/lorarite.md) | ICLR 2025 | [LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization](https://arxiv.org/abs/2410.20625) | [official](https://github.com/gkevinyen5418/LoRA-RITE) | — |
| [MicroAdam](math/microadam.md) | NeurIPS 2024 | [MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence](https://arxiv.org/abs/2405.15593) | [official](https://github.com/IST-DASLab/MicroAdam) | — |
| [Muon](math/muon.md) | Blog 2024 | [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) | [official](https://github.com/KellerJordan/Muon) | `Muon` |
| [Online Subspace Descent](math/onlinesubspacedescent.md) | NeurIPS 2024 | [Memory-Efficient LLM Training with Online Subspace Descent](https://arxiv.org/abs/2408.12857) | [official](https://github.com/kyleliang919/Online-Subspace-Descent) | — |
| [Q-GaLore](math/qgalore.md) | CPAL 2025 | [Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients](https://arxiv.org/abs/2407.08296) | [official](https://github.com/VITA-Group/Q-GaLore) | — |
| [SGD-SaI](math/sgdsai.md) | arXiv 2024 | [No More Adam: Learning Rate Scaling at Initialization is All You Need](https://arxiv.org/abs/2412.11768) | [official](https://github.com/AnonymousAlethiometer/SGD_SaI) | `SGDSaI` |
| [SMMF](math/smmf.md) | AAAI 2025 | [SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization](https://arxiv.org/abs/2412.08894) | [official](https://github.com/eai-lab/SMMF) | — |
| [SNSM](math/snsm.md) | ICML 2025 | [Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum with Convergence Guarantees](https://arxiv.org/abs/2411.07120) | [official](https://github.com/timmytonga/sn-sm) | — |
| [SWAN](math/swan.md) | ICML 2025 | [SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training](https://arxiv.org/abs/2412.13148) | — | — |
| [AlphaGrad](math/alphagrad.md) | arXiv 2025 | [AlphaGrad: Non-Linear Gradient Normalization Optimizer](https://arxiv.org/abs/2504.16020) | — | — |
| [GWT](math/gwt.md) | arXiv 2025 | [GWT: Scalable Optimizer State Compression for Large Language Model Training](https://arxiv.org/abs/2501.07237) | — | — |
| [MLorc](math/mlorc.md) | AISTATS 2026 | [MLorc: Momentum Low-rank Compression for Memory Efficient Large Language Model Adaptation](https://arxiv.org/abs/2506.01897) | [official](https://github.com/weishen-git/MLorc) | — |
| [MoFaSGD](math/mofasgd.md) | TMLR 2025 | [Low-rank Momentum Factorization for Memory Efficient Training](https://arxiv.org/abs/2507.08091) | [official](https://github.com/pmahdavi/MoFaSGD) | — |
| [RACS / Alice](math/racsalice.md) | arXiv 2025 | [Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension](https://arxiv.org/abs/2502.07752) | [community](https://github.com/kozistr/pytorch_optimizer) | — |
| [SinkGD](math/sinkgd.md) | arXiv 2025 | [Gradient Multi-Normalization for Stateless and Scalable LLM Training](https://arxiv.org/abs/2502.06742) | — | — |
| [SPAM](math/spam.md) | ICLR 2025 | [SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training](https://arxiv.org/abs/2501.06842) | [official](https://github.com/TianjinYellow/SPAM-Optimizer) | `SPAM` |
| [SubTrack++](math/subtrack.md) | NeurIPS 2025 | [SubTrack++ : Gradient Subspace Tracking for Scalable LLM Training](https://arxiv.org/abs/2502.01586) | [official](https://github.com/criticalml-uw/SubTrack) | — |
| [SUMO](math/sumo.md) | NeurIPS 2025 | [SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training](https://arxiv.org/abs/2505.24749) | — | — |
| [TensorGRaD](math/tensorgrad.md) | arXiv 2025 | [TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient Neural Operator Training](https://arxiv.org/abs/2501.02379) | — | — |
| [FlashOptim](math/flashoptim.md) | arXiv 2026 | [FlashOptim: Optimizers for Memory-Efficient Training](https://arxiv.org/abs/2602.23349) | [official](https://github.com/databricks/flashoptim) | — |
| [Rose](math/rose.md) | GitHub 2026 | [Rose: Range-Of-Slice Equilibration optimizer](https://github.com/MatthewK78/Rose#-citation) | [official](https://github.com/MatthewK78/Rose) | — |
| [SAGE](math/sage.md) | ACL 2026 Findings | [SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](https://arxiv.org/abs/2604.07663) | — | — |
| [BlockLLM](math/blockllm.md) | arXiv 2024 | [BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks](https://arxiv.org/abs/2406.17296) | [official](https://github.com/RAmruthaVignesh/blockllm) | — |
| [Natural GaLore](math/naturalgalore.md) | arXiv 2024 | [Natural GaLore: Accelerating GaLore for memory-efficient LLM Training and Fine-tuning](https://arxiv.org/abs/2410.16029) | [official](https://github.com/selfsupervised-ai/Natural-GaLore) | — |
| [SLTrain](math/sltrain.md) | NeurIPS 2024 | [SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining](https://arxiv.org/abs/2406.02214) | [official](https://github.com/andyjm3/SLTrain) | — |
| [8-bit Muon](math/8bitmuon.md) | arXiv 2025 | [Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106) | — | — |
| [FFT-based Subspace Selection](math/fftbasedsubspaceselection.md) | ICLR 2026 | [FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of Large Language Models](https://arxiv.org/abs/2505.17967) | [official](https://github.com/IST-DASLab/ISTA-DASLab-Optimizers) | — |
| [FOAM](math/foam.md) | arXiv 2025 | [FOAM: Blocked State Folding for Memory-Efficient LLM Training](https://arxiv.org/abs/2512.07112) | [official](https://github.com/zqOuO/FOAM) | — |
| [GaLore 2](math/galore2.md) | arXiv 2025 | [GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2504.20437) | — | — |
| [GradientStabilizer](math/gradientstabilizer.md) | ICML 2026 | [GradientStabilizer: Fix the Norm, Not the Gradient](https://arxiv.org/abs/2502.17055) | [official](https://github.com/TianjinYellow/GradientStabilizer) | — |
| [GUM](math/gum.md) | arXiv 2025 | [Unbiased Gradient Low-Rank Projection](https://arxiv.org/abs/2510.17802) | — | — |
| [I3S](math/i3s.md) | NeurIPS 2025 | [Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](https://arxiv.org/abs/2502.05790) | — | — |
| [LORENZA](math/lorenza.md) | TMLR 2026 | [LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training via Efficient Zeroth-Order Adaptive SAM](https://arxiv.org/abs/2502.19571) | — | — |
| [ProjFactor (VLoRP)](math/projfactorvlorp.md) | arXiv 2025 | [Memory-Efficient LLM Training by Various-Grained Low-Rank Projection of Gradients](https://arxiv.org/abs/2505.01744) | — | — |
| [RSO](math/rso.md) | arXiv 2025 | [A Memory Efficient Randomized Subspace Optimization Method for Training Large Language Models](https://arxiv.org/abs/2502.07222) | — | — |
| [SCALE](math/scale.md) | ICML 2026 | [Memory-Efficient LLM Pretraining via Minimalist Optimizer Design](https://arxiv.org/abs/2506.16659) | — | — |
| [SlimAdam](math/slimadam.md) | arXiv 2025 | [When Can You Get Away with Low Memory Adam?](https://arxiv.org/abs/2503.01843) | [official](https://github.com/dayal-kalra/low-memory-adam) | — |
| [LoRA-Pre](math/lorapre.md) | ICLR 2026 | [Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation](https://arxiv.org/abs/2602.24283) | [official](https://github.com/mrflogs/LoRA-Pre) | — |
| [Lotus](math/lotus.md) | arXiv 2026 | [Lotus: Efficient LLM Training by Randomized Low-Rank Gradient Projection with Adaptive Subspace Switching](https://arxiv.org/abs/2602.01233) | — | — |
| [POET-X](math/poetx.md) | ICML 2026 | [POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation](https://arxiv.org/abs/2603.05500) | [official](https://github.com/Sphere-AI-Lab/poet) | — |
| [MuonQ](math/muonq.md) | arXiv 2026 | [MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization](https://arxiv.org/abs/2605.11396) | [official](https://github.com/YupengSu/MuonQ) | — |
| [4-bit-Muon-GRASP](math/4bitmuongrasp.md) | ICLR 2026 | [Achieving low-bit Muon through subspace preservation and grid quantization](https://openreview.net/forum?id=g2l9bg9DWx) | [official](https://github.com/wuhuaijin/lowbit-Muon) | — |
| [IO-Adam](math/ioadam.md) | OpenReview 2026 | [IO-Adam: Rethinking Memory-Efficient Adaptive Optimizers from Gradient Computation](https://openreview.net/forum?id=iCT5xdOlJH) | — | — |
| [H-Fac](math/hfac.md) | AISTATS 2025 | [Memory-Efficient Optimization with Factorized Hamiltonian Descent](https://arxiv.org/abs/2406.09958) | — | — |
| [LiMuon](math/limuon.md) | ICML 2026 | [LiMuon: Light and Fast Muon Optimizer for Large Models](https://arxiv.org/abs/2509.14562) | — | — |
| [M+Adam](math/madam2.md) | OPT 2025: 17th Annual Workshop on Optimization for Machine Learning (NeurIPS 2025 Workshop) | [M+Adam: Stable Low-Precision Training with Combined Adam–Madam Updates](https://opt-ml.org/papers/2025/paper141.pdf) | — | — |
| [SMET](math/smet.md) | ICML 2026 | [Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling](https://arxiv.org/abs/2606.00888) | [official](https://github.com/QiaoXiao7282/SMET) | — |
| [PowerStep](math/powerstep.md) | arXiv 2026 | [PowerStep: Memory-Efficient Adaptive Optimization via ell_p-Norm Steepest Descent](https://arxiv.org/abs/2605.10335) | [official](https://github.com/yaolubrain/PowerStep) | — |
| [SRON](math/sron.md) | OpenReview 2025 | [SRON: State-free LLM Training via Row-wise Gradient Normalization](https://openreview.net/forum?id=BtQLBWr6zI) | — | — |
| [GradLite](math/gradlite.md) | arXiv 2025 | [Backward-Friendly Optimization: Training Large Language Models with Approximate Gradients under Memory Constraints](https://arxiv.org/abs/2510.22467) | — | — |
| [Optimal Low-Rank SGE](math/optimallowranksge.md) | arXiv preprint 2026 | [Optimal low-rank stochastic gradient estimation for LLM training](https://arxiv.org/abs/2603.20632) | — | — |
| [Spectral Compact Training (SCT)](math/spectralcompacttrainingsct.md) | arXiv 2026 | [Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction](https://arxiv.org/abs/2604.00733) | [official](https://github.com/EctoSpace/SCT) | — |

## Trainer integrations

HuggingFace `transformers` exposes many of these methods through the `optim` argument of `TrainingArguments`. Each string value below maps to a memory-efficient optimizer; all backing libraries except the built-in Adafactor must be installed separately.

| `optim` value | Backing library |
|---|---|
| [`adafactor`](math/adafactor.md) | [transformers](https://github.com/huggingface/transformers) ships its own `Adafactor` implementation with relative-step and update-clipping options (Apache-2.0). |
| `adamw_bnb_8bit` / `adamw_8bit` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) AdamW with block-wise 8-bit quantized state (MIT). |
| `paged_adamw_8bit` / `paged_adamw_32bit` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) paged AdamW; optimizer state is paged between GPU and CPU memory (MIT). |
| `lion_8bit` / `lion_32bit` / `paged_lion_8bit` / `paged_lion_32bit` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) Lion, single momentum buffer, with 8-bit and paged variants (MIT). |
| `ademamix_8bit` / `paged_ademamix_8bit` / `paged_ademamix_32bit` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) AdEMAMix with 8-bit quantized and paged state (MIT). |
| `rmsprop_bnb_8bit` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) RMSprop with block-wise 8-bit quantized state (MIT). |
| `adamw_torch_4bit` / `adamw_torch_8bit` | [torchao](https://github.com/pytorch/ao) pure-PyTorch AdamW with 4-bit or 8-bit optimizer states (BSD-3-Clause). |
| `galore_adamw` / `galore_adamw_8bit` / `galore_adafactor` and `*_layerwise` variants | [galore-torch](https://github.com/jiaweizzhao/GaLore), the official GaLore release (Apache-2.0). |
| `apollo_adamw` / `apollo_adamw_layerwise` | [apollo-torch](https://github.com/zhuhanqing/APOLLO), the official APOLLO release (CC-BY-NC-4.0). |
| `lomo` / `adalomo` | [lomo-optim](https://github.com/OpenLMLab/LOMO), the official LOMO and AdaLomo release (MIT). |
