<div style="text-align: center;">
  <img src="./images/jax_logo_250px.png" alt="Description of image" style="display: block; margin: 0 auto">
</div>


# JLR - JAX Learning Repository (Machine Learning with JAX)
This is a learning repository to learn JAX and how fundamental calculus operations are performed with and within JAX while building a few models while exploring.

## Multivariable Calculus (Foundations)

- [x] Functions of multiple variables (ℝⁿ → ℝ, ℝⁿ → ℝᵐ)
- [x] Domains, codomains, level sets, contour plots
- [x] Partial derivatives and geometric interpretation
- [x] Directional derivatives and gradient vector
- [x] Total derivative and multivariable chain rule
- [x] Jacobian matrix and linear approximation
- [x] Jacobian–vector and vector–Jacobian products
- [ ] Second-order derivatives and Hessian matrix
- [ ] Curvature, saddle points, local minima
- [ ] Multivariable Taylor expansion (1st and 2nd order)
- [ ] Constrained optimization and Lagrange multipliers

---

## Linear Algebra for Representation Learning

- [ ] Vector spaces and subspaces
- [ ] Span, basis, linear independence
- [ ] Coordinate systems and change of basis
- [ ] Linear transformations and matrix interpretation
- [ ] Rank, null space, column space
- [ ] Eigenvalues and eigenvectors
- [ ] Spectral norms and conditioning
- [ ] Singular Value Decomposition (SVD)
- [ ] Low-rank approximations and information compression

---

## Probability 

- [ ] Random variables (discrete and continuous)
- [ ] Expectation, variance, covariance
- [ ] Joint and conditional distributions
- [ ] Independence vs correlation
- [ ] Multivariate Gaussian distributions
- [ ] Maximum Likelihood Estimation (MLE)
- [ ] Maximum A Posteriori (MAP) estimation
- [ ] Entropy and uncertainty
- [ ] Cross-entropy loss
- [ ] KL divergence and distribution matching
- [ ] Mutual information and representation learning

---

## Optimization for Deep Learning

- [ ] Gradient descent (batch, mini-batch, stochastic)
- [ ] Learning rate schedules
- [ ] Stochasticity and implicit regularization
- [ ] Momentum-based optimization
- [ ] RMSProp and Adam
- [ ] Bias correction in adaptive methods
- [ ] Non-convex loss landscapes
- [ ] Saddle points and flat minima
- [ ] Overparameterization and double descent
- [ ] Generalization and implicit bias

---

## JAX Core Concepts

- [ ] Functional programming principles
- [ ] Pure functions and immutability
- [ ] `jax.numpy` and array semantics
- [ ] Automatic differentiation (`grad`)
- [ ] Forward-mode vs reverse-mode AD
- [ ] JIT compilation (`jit`) and XLA
- [ ] Vectorization with `vmap`
- [ ] Parallelism with `pmap`
- [ ] Manual parameter management
- [ ] Training loops from scratch in JAX

---

## Core Deep Learning Architectures

- [ ] Multi-Layer Perceptrons (MLPs)
- [ ] Depth vs width trade-offs
- [ ] Backpropagation derivation
- [ ] Backpropagation implementation in JAX
- [ ] Convolution operation fundamentals
- [ ] Receptive fields and locality
- [ ] Translation invariance
- [ ] CNN implementation in JAX
- [ ] CNN mini-project (MNIST / CIFAR-10)

---

## NLP and Sequence Modeling

- [ ] Discrete sequences and tokenization
- [ ] Embedding layers and learned representations
- [ ] Recurrent neural networks (conceptual)
- [ ] Limitations of recurrence
- [ ] Self-attention mechanism
- [ ] Query, key, value formulation
- [ ] Transformer architecture
- [ ] Positional encoding
- [ ] Transformer mini-project in JAX (small LM)

---

## Computer Vision (Advanced)

- [ ] Feature hierarchies in CNNs
- [ ] Batch normalization
- [ ] Layer normalization
- [ ] Regularization techniques (dropout, weight decay)
- [ ] Architectural ablations
- [ ] Vision-focused mini-project with analysis

---

## Large Language Models (LLMs)

- [ ] Subword tokenization (BPE / WordPiece)
- [ ] Causal language modeling objective
- [ ] Scaling laws (data, parameters, compute)
- [ ] Training dynamics at scale
- [ ] Evaluation metrics (perplexity)
- [ ] Limitations of likelihood-based evaluation
- [ ] Alignment overview (RLHF basics)

---

## Projects

- [ ] JAX Transformer with attention visualization
- [ ] Optimization comparison study
- [ ] Loss landscape or training dynamics analysis





$$J_f(x_1, x_2, x_3) = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \frac{\partial f_1}{\partial x_3} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \frac{\partial f_2}{\partial x_3} \\ \frac{\partial f_3}{\partial x_1} & \frac{\partial f_3}{\partial x_2} & \frac{\partial f_3}{\partial x_3} \end{bmatrix}$$