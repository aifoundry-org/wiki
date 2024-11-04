# Make_qx2_quants
[Link to the method](https://github.com/ggerganov/llama.cpp/blob/524afeec9dad7d765ce91f5cf30c73703867cb47/ggml/src/ggml-quants.c#L1817)

# Problem Statement

The problem formulation is the same as in [make_qx_quants](make_qx_quants.md) method. 
But with additional parameter and looks as follows:

$$
F(s, z) = \sum_i w_i((s q_i + z) - x_i)^2 \quad (1)
$$

where:
- $w_i$ is the weight importance 
- $s$ is the quantization scale
- $q_i$ are the quantized weights
- $x_i$ are the original weights
- $z$ is additional parameter which represent quantization offset or zero-point

## Solving

The simplification process is similar to [make_qx_quants](make_qx_quants.md) method, 
but this time two partial derivatives should be calculated w.r.t $s$ and $z$

1.**Partial derivative with respect to** $s$:

$$
\frac{\partial F}{\partial s} = 2 \sum_i w_i q_i ((s q_i + z) - x_i)
$$

2.**Partial derivative with respect to** $z$:

$$
\frac{\partial F}{\partial z} = 2 \sum_i w_i ((s q_i + z) - x_i)
$$

Equating derivatives to 0 we can rewrite equations as:

$$
\sum_i w_i q_i x_i = s \sum_i w_i q_i^2 + z \sum_i w_i q_i
$$

$$
\sum_i w_i x_i = s \sum_i w_i q_i + z \sum_i w_i
$$

3.**This system of equations** can be rewritten in matrix form:

```math
\begin{bmatrix}
\sum_i w_i q_i^2 & \sum_i w_i q_i \\
\sum_i w_i q_i & \sum_i w_i 
\end{bmatrix}

\begin{bmatrix}
s \\
z 
\end{bmatrix}
= 
\begin{bmatrix}
\sum_i w_i q_i x_i \\
\sum_i w_i x_i 
\end{bmatrix}

\quad (2)
```

Using Crammer's rule system $(2)$ could be solved for $s$ and $z$.

*Determinant* $D$ could be calculated

```math
D = 
\begin{vmatrix}
\sum_i w_i q_i^2 & \sum_i w_i q_i \\
\sum_i w_i q_i & \sum_i w_i 
\end{vmatrix}
= 
\sum_i w_i q_i^2 \cdot \sum_i w_i - \sum w_i q_i \cdot \sum w_i q_i \quad (3)
```


*[Determinant calculation](https://github.com/ggerganov/llama.cpp/blob/524afeec9dad7d765ce91f5cf30c73703867cb47/ggml/src/ggml-quants.c#L1869)*

4.**Solving** for $s$ and $z$:

$$
s = \frac{\sum_i w_i \cdot \sum_i w_i x_i q_i - \sum_i w_i x_i \cdot \sum_i w_i q_i  }{D} \quad (4)
$$

$$
z = \frac{\sum_i w_i q_i^2 \cdot \sum_i w_i x_i - \sum_i w_i q_i \cdot \sum_i w_i x_i q_i}{D} \quad (5)
$$

[Scale ($s$) calculation ](https://github.com/ggerganov/llama.cpp/blob/524afeec9dad7d765ce91f5cf30c73703867cb47/ggml/src/ggml-quants.c#L1871C19-L1871C29)

[Offset ($z$)(min) calculation](https://github.com/ggerganov/llama.cpp/blob/524afeec9dad7d765ce91f5cf30c73703867cb47/ggml/src/ggml-quants.c#L1872)

## Quantization loop

As mentioned in [make_qx_quants](make_qx_quants.md) method, this optimization problem is not solvable in general. So in order to find optimal $s$ and $z$ values some iterative adjustments will be performed.

### For each step $is$ in $0$ to $n_{step}$
*in this context $min$ acts as z-offset in process of quantization*

1. **Set $iscale$**:

$$
iscale = \frac{r_{min} + r_{delta} \cdot is + n_{max}}{max - min}
$$

where:
* $r_{min}$ and $r_{delta}$ are parameters cpntrolling the range of tested scaling factors.
* $min$ and $max$ are observed minimum and maximum values of weights 

2. **Quantize weights using new $iscale$**:

$$
q_i = clamp( \lfloor iscale \cdot (x_i - min) + 0.5 \rfloor, 0, n_{max})
$$

3. **Compute weighted sums described in system $(2)$**

4. **Compute denominator $D$ from equation $(3)$**

5. **Check if $D > 0$ to ensure solvability.**
    * If $D \leq 0$ skip to the next iteration.

6. **Calculate $scale$ and $offset$ using $(4)$ $(5)$ equations.**

7. **Compute the error with updated parameters:**

$$
E = \sum_i w_i \cdot (s \cdot q_i + min - x_i)
$$

8. **Update the best parameters if the error has decreased:**
* If $E < best_{mad}:$
    * update quantized weights $q_i$
    * Set $best_{mad} = E$ 
    * Update $scale$ and $min$

### Return

The result that returned by this function is $scale$ and the $offest$ $(min)$.
