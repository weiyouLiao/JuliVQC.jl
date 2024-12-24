# Examples 

This section demonstrates how to use **JuliVQC.jl** to create variational quantum circuits (VQCs) and integrate them with **Flux.jl** for optimization tasks. Additionally, some utility functions provided by the library are introduced.

---

## Creating a Variational Quantum Circuit 

A variational quantum circuit (VQC) is a parameterized quantum circuit composed of gates with tunable parameters. Below is an example of how to manually construct such a circuit.

```julia
using JuliVQC,QuantumCircuits

# Circuit parameters
L = 3         # Number of qubits
depth = 2     # Depth of the circuit

# Create an empty quantum circuit
circuit = QCircuit()

# Add parameterized gates to the circuit
for i in 1:L
    push!(circuit, RzGate(i, rand(),isparas=true))# Rz rotation with random parameter
	push!(circuit, RyGate(i, rand(),isparas=true))# Ry rotation with random parameter
	push!(circuit, RzGate(i, rand(),isparas=true))# Rz rotation with random parameter
end

# Add entangling gates and repeat for each layer of depth
for l in 1:depth
    for i in 1:L-1
        push!(circuit, CNOTGate((i, i+1)))       # Add CNOT gate
    end
    for i in 1:L
		push!(circuit, RzGate(i, rand(),isparas=true))
		push!(circuit, RxGate(i, rand(),isparas=true))
		push!(circuit, RzGate(i, rand(),isparas=true))
    end
end

# Extract all the parameters of the circuit
paras = parameters(circuit)

# Reset all the parameters of the circuit to zeros
new_paras = zeros(length(paras))
set_parameters!(new_paras, circuit)
paras = parameters(circuit)  # Updated parameters

# Compute the gradient of a loss function with respect to the circuit parameters
using Zygote
target_state = StateVector([.0,.0,.0,.0,.0,.0,.1,.0])        # Random target quantum state
initial_state = StateVector(3)       # Initial quantum state
loss(c) = distance(target_state, c * initial_state)  # Define the loss function
grad = gradient(loss, circuit)  # Compute the gradient
```

The above process of creating a VQC has already been encapsulated into the following convenient function:

```julia
variational_circuit(L::Int, depth::Int, g::Function=rand)
```

For example, you can directly create a VQC by calling:

```julia
circuit = variational_circuit(3, 2)
```

---

## A Simple Application of JuliVQC and Flux 

This example demonstrates how to integrate a variational quantum circuit  with **Flux.jl**, a machine learning library for Julia. The task is to optimize the parameters of a VQC to prepare a target quantum state.

```julia
using VQC
using Zygote
using Flux.Optimise

# Create a variational quantum circuit
circuit = variational_circuit(3, 2)

# Define the target state and initial state
target_state = StateVector([.0,.0,.0,.0,.0,.0,.1,.0])  # Target quantum state
initial_state = StateVector(3)                                 # Initial quantum state

# Define the loss function: distance between target and evolved state
loss(c) = distance(target_state, c * initial_state)

# Initialize the optimizer
opt = ADAM()

# Number of training epochs
epochs = 10

# Extract initial parameters
x0 = parameters(circuit)

# Optimization loop
for i in 1:epochs
    # Compute the gradient of the loss with respect to the circuit parameters
    grad = collect_variables(gradient(loss, circuit))
    
    # Update the parameters using the optimizer
    Optimise.update!(opt, x0, grad)
    set_parameters!(x0, circuit)  # Update circuit parameters
    
    # Print the loss value for each epoch
    println("Loss value at iteration $i: $(loss(circuit))")
end
```

### Explanation of the Workflow

1. **Circuit Initialization**: A variational quantum circuit (`variational_circuit`) is initialized with random parameters.
2. **Target and Initial States**: The target state and initial state are defined. The circuit's goal is to transform the initial state into the target state.
3. **Loss Function**: The loss function quantifies the "distance" between the target state and the circuit-evolved state.
4. **Optimization**: Using Flux's `ADAM` optimizer, the circuit parameters are iteratively updated to minimize the loss.

---

## Utility Functions 

The following utility functions are provided by **JuliVQC.jl** to facilitate working with quantum circuits:

```julia
collect_variables(args...)           # Collect all variables (parameters) from the circuit
parameters(args...)                  # Get the parameters of the circuit
set_parameters!(coeff::AbstractVector{<:Number}, args...)  # Set parameters of the circuit
simple_gradient(f, args...; dt::Real=1.0e-6)               # Compute numerical gradients
check_gradient(f, args...; dt::Real=1.0e-6, atol::Real=1.0e-4, verbose::Int=0)  
# Verify the correctness of gradients
```

These functions simplify common operations, such as extracting or modifying circuit parameters and computing gradients, making the library more user-friendly.

