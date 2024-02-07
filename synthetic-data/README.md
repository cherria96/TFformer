Ref: https://www.mdpi.com/1099-4300/21/11/1080
## Neural-mass model (Jansen-Rit model)
1. Define the parameters and the coupling matrix $C_{i,j}$ for the ring structure.
    - A, B: excitation parameter that influences the type of activity simulated by the model
    - a, b : time constant
    - r: steepness of the sigmoid
    - $e_0$ : half activation rate
    - $v_0$ : firing threshold
    - $C_1 ~ C_4$: connectivity constant
2. Implement the sigmoid function $S(v)$.
3. Define the differential equations for the extended model that includes coupling between the neural masses.
4. Implement the stochastic term $p_j(t)$
5. Solve the system of equations using an ODE solver for a given time span.
6. Collect the output from the first variable $y^j_0$ of each population to form the multivariate time series.

## Henon-map model
1. Define the parameters and the coupling strength $C$
    - K: number of components
    - T: time constant
    - C: coupling strength
2. First set values for i = 1 and K using the following equation

   $x_{i,t} = 1.4 - x^2_{i,t-1} +0.3x_{i,t-2}$
4. Calculate the remaining values using the following equation

   $x_{i,t} = 1.4 - (0.5C(x_{i-1,t-1}+x_{i+1, t-1})+(1-C)x_{i,t-1})^2 + 0.3x_{i,t-2}$
