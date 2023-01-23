

## Forward Propogation:
$$a^{(1)} = \bf{x} $$
$$h^{(i)} = a^{(i-1)} \cdot W^{(i)}$$
$$a^{(i)} = f(h^{(i)})$$
$$a^{(final)} = y = f(h^{(final)})$$


## Back propagation:

1) Forward propagate, get $y$
2) Calculate $\it{L}(y)$
3) Calculate gradients of $\it{L}$
4) Update parameters


##### Calculate error
$$E = E(\hat{y}, y ) = \frac{1}{2}(\hat{y} - y)^2$$

##### Calculate graidents of error w/ respect to weights

$$ \frac{\partial{E}}{\partial{W^{(n)}}} $$

To calculate, think NN as a functional chain: $F = F(\bf{x}, W)$. So, the output of the NN is the output of $F$. 

Thus, $$E = E(\hat{y}, y) = E(F(\bf{x}, W), y)$$

To calculate the partial derivatives, use chain rule:
$$
\frac{\partial{E}}{\partial{W^{(2)}}}  = \frac{\partial{E}}{\partial{a^{(3)}}} \cdot \frac{\partial{a^{(3)}}}{\partial{h^{(3)}}}  \cdot \frac{\partial{h^{(3)}}}{\partial{W^{(2)}}}
$$

since for an $n$ layered neural network:

$$
F = f(a^{(n-1)}\cdot W^{n-1}) = f(f(a^{(n-2)}\cdot W^{n-2})\cdot W^{n-1}) = f(f(f(a^{(n-3)}\cdot W^{n-3})\cdot W^{n-2})\cdot W^{n-1}) = ...
$$

and for n = 3:
$$
F = a^{(3)} =  f(h^{(3)}) = f(f(a^{(2)}\cdot W^{(2)})) = f(f(f(h^{(2)}))\cdot W^{2}) \\ = f(f(f(a^{(1)}\cdot W^{(1)}))\cdot W^{2}) = f(f(f(\bold{x}\cdot W^{(1)}))\cdot W^{2}) 
$$

##### Update parameters

$$ W^{(k+1)} = W^{(k)} - \eta \nabla_W E(W)$$