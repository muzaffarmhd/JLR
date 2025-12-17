<div style="text-align: center;">
  <img src="./images/jax_logo_250px.png" alt="Description of image" style="display: block; margin: 0 auto">
</div>


# JAX Learning Repository (JLR)
This is a learning repository to learn JAX and how fundamental calculus operations are performed with and within JAX while building a few models while exploring.
## The Prerequisite Math
- Partial Differentiation & Gradients: Understand how to compute the gradient of a scalar-valued function with respect to a vector input. This is the foundation of training models.
1. Total Derivative:
  - Find du/dt using the total derivative given u = y^2 - 4ax, x = at^2, y = bt^3.
  - We'll do this in JAX as well

## About JAX
JAX often uses 32-bit precision, while standard NumPy and Python's math module default to 64-bit. For exact matches with 64-bit results, you may need to enable 64-bit computation in JAX by setting jax.config.update`('jax_enable_x64', True)`

What is 32-bit and 64-bit precision