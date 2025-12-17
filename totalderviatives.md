# Find du/dt Using Total Derivative

**Given:**
- $u = y^2 - 4ax$
- $x = at^2$
- $y = 2at$

**Solution:**

Using the total derivative:
$$\frac{du}{dt} = \frac{\partial u}{\partial x}\frac{dx}{dt} + \frac{\partial u}{\partial y}\frac{dy}{dt}$$

**Step 1: Find partial derivatives of u**
$$\frac{\partial u}{\partial x} = -4a$$
$$\frac{\partial u}{\partial y} = 2y$$

**Step 2: Find derivatives of x and y with respect to t**
$$\frac{dx}{dt} = 2at$$
$$\frac{dy}{dt} = 2a$$

**Step 3: Apply the chain rule**
$$\frac{du}{dt} = (-4a)(2at) + (2y)(2a)$$
$$\frac{du}{dt} = -8a^2t + 4ay$$

**Step 4: Substitute $y = 2at$**
$$\frac{du}{dt} = -8a^2t + 4a(2at)$$
$$\frac{du}{dt} = -8a^2t + 8a^2t$$
$$\frac{du}{dt} = 0$$

# Computing the same in JAX
Jax provides array operations that are compatible with JAX's automatic differentiation and JIT compilation. We use it just like NumPy, but it works with JAX's transformations.
```python
import jax.numpy as jnp
from jax import grad

def u(t, a):
    x = a * t**2
    y = 2*a*t
    return y**2 - 4 * a * x
du_dt = grad(u, argnums=0)

result = du_dt(1.0,1.0)

print(f"du/dt at t=1.0 and a=1.0 is {result}")
```
```
output: du/dt at t=1 and a=1 is 0.0
```

# Find dz/dt Using Total Derivative

**Given:**
- $z = x^2 + y^2$
- $x = \cos t$
- $y = \sin t$

**Solution:**

Using the total derivative:
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

**Step 1: Find partial derivatives of z**
$$\frac{\partial z}{\partial x} = 2x$$
$$\frac{\partial z}{\partial y} = 2y$$

**Step 2: Find derivatives of x and y with respect to t**
$$\frac{dx}{dt} = -\sin t$$
$$\frac{dy}{dt} = \cos t$$

**Step 3: Apply the chain rule**
$$\frac{dz}{dt} = (2x)(-\sin t) + (2y)(\cos t)$$
$$\frac{dz}{dt} = -2x\sin t + 2y\cos t$$

**Step 4: Substitute $x = \cos t$ and $y = \sin t$**
$$\frac{dz}{dt} = -2\cos t \sin t + 2\sin t \cos t$$
$$\frac{dz}{dt} = 0$$

## Computing same in JAX
```python
import jax.numpy as jnp
from jax import grad

def z(t):
    x = jnp.cos(t)
    y = jnp.sin(t)
    return x**2 + y**2
dz_dt = grad(z, argnums=0)

result = dz_dt(1.0,1.0)
print(result)
```

# Find dz/dt Using Total Derivative

**Given:**
- $z = x^2y + xy^3$
- $x = e^t$
- $y = \ln t$

**Solution:**

Using the total derivative:
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

**Step 1: Find partial derivatives of z**
$$\frac{\partial z}{\partial x} = 2xy + y^3$$
$$\frac{\partial z}{\partial y} = x^2 + 3xy^2$$

**Step 2: Find derivatives of x and y with respect to t**
$$\frac{dx}{dt} = e^t$$
$$\frac{dy}{dt} = \frac{1}{t}$$

**Step 3: Apply the chain rule**
$$\frac{dz}{dt} = (2xy + y^3)(e^t) + (x^2 + 3xy^2)\left(\frac{1}{t}\right)$$

**Step 4: Substitute $x = e^t$ and $y = \ln t$**
$$\frac{dz}{dt} = (2e^t\ln t + (\ln t)^3)(e^t) + (e^{2t} + 3e^t(\ln t)^2)\left(\frac{1}{t}\right)$$
$$\frac{dz}{dt} = 2e^{2t}\ln t + e^t(\ln t)^3 + \frac{e^{2t}}{t} + \frac{3e^t(\ln t)^2}{t}$$

## Computing same in JAX
```python
import jax.numpy as jnp
from jax import grad

def z(t):
    x = jnp.exp(t)
    y = jnp.log(t)
    return x**2 * y + x * y**3

dz_dt = grad(z, argnums=0)

result = dz_dt(1.0)
print(f"dz/dt at t=1.0 is {result}")
```



