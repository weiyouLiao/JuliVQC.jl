# Manual


In this section we show the working pipeline of JuliVQC for simulating quantum circuits and variational quantum circuits, as well as their noisy counterparts.

## Initialize a quantum state
The first step of using JuliVQC for any quantum circuit simulation is to initialize a quantum state stored as a state vector. JuliVQC provides twos function: `StateVector` and `DensityMatrix` to initialize a pure state and a mixed state respectively. Mathematically, the data of an $n$-qubit pure state should be understood as a rank-$n$ tensor, and the data of an $n$-qubit mixed state should be understood as a rank-$2n$ tensor, where each dimension has size $2$. 

As implementation-wise details, the qubits are internally labeled from $1$ to $n$ for pure state , while for mixed state the ket indices are labeled from $1$ to $n$ and the bra indices are labeled from $n+1$ to $2n$. 

Column-major storage is used for the data of both pure and mixed quantum states, e.g., the smaller indices of the tensor are iterated first. These details are not important for the users if they do not want to access the raw data of the quantum states.

```julia
using JuliVQC,QuantumCircuits

state = StateVector(2)
n = 2
pure_state = StateVector(n)
mixed_state = DensityMatrix(n)
custom_pure_state= StateVector([.0,.1,.0,.0])
custom_mixed_state = DensityMatrix([.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.1])
```


## Quantum gates

The second step of using JuliVQC is to build a quantum circuit, for which one needs to define each elementary quantum gate operations (and quantum channels for noisy quantum circuits). 

The universal way of defining quantum gates is to use the function `QuantumGate(positions, data)`, where the first argument specifies the qubits indices that the gate operates on, for example `positions =(1, 3)`, and the second argument is the raw data of the gate operation which should be a unitary matrix. 

In the meantime, JuliVQC provides specialized definitions of commonly used quantum gates "`X`, `Y`, `Z`, `S`, `H`, `sqrtX`, `sqrtY`, `T`, `Rx`, `Ry`,`Rz`, `CONTROL`, `CZ`, `CNOT`, `CX`,`SWAP`, `iSWAP`, `XGate`, `YGate`, `ZGate`, `HGate`, `SGate`, `TGate`, `SqrtXGate`, `SqrtYGate`, `RxGate`, `RyGate`, `RzGate`, `CZGate`, `CNOTGate`, `SWAPGate`, `iSWAPGate`, `CRxGate`, `CRyGate`, `CRzGate`, `TOFFOLIGate`". Specific optimizations have been implemented for most of the predefined gate operations by exploring their structures, which will usually be faster than using the `QuantumGate}` function.

JuliVQC also provides general two-qubit and three-qubit controlled gate operations: `CONTROLGate` and `CONTROLCONTROLGate`, which can be used as `CONTROLGate(i,j,data)` (`i` is the control qubit and `j` is the target qubit) and `CONTROLCONTROLGate(i,j,k,data)` (`i` and `j` are control qubits and `k` is the target qubit), with `data` the raw data for the target single-qubit operation. 

The general interface for initializing a parametric quantum gate is `G(i..., paras; isparas)` where `paras` is a single scalar if `G` only has a single parameter or an array of scalars if `G` has several parameters.

The illustrative code for initializing non-parametric and parametric quantum gates:

```julia
using JuliVQC,QuantumCircuits
n=1
X = XGate(n)
ncontrol = 1
ntarget = 2
CNOT = CNOTGate(ncontrol, ntarget)
theta = pi/2
non_para_Rx = RxGate(n, theta, isparas=false) # a non-parametric Rx gate
para_Rx = RxGate(n, theta, isparas=true) # a parametric Rx gate
```



## Noise channels

In additional to the quantum gate operations, an indispensable ingredient for noisy quantum circuit is the quantum channel, which describes the effects of noises. 

Similar to the function `QuantumGate`, JuliVQC provides a universal function `QuantumMap(positions, kraus)` which allows the user to define arbitrary quantum channels, where the first argument `positions` specifies the qubit indices that the quantum channel operates on, similar to the case of a unitary quantum gate, and the second argument `kraus` is a list of Kraus operators.

JuliVQC also provides some commonly used single-qubit quantum channels based on the function `QuantumMap`, including `AmplitudeDamping(pos, p)` ,`PhaseDamping(pos, p)`,Depolarizing`(pos, p)`



## Manipulating and running quantum circuits

JuliVQC uses a very simple wrapper `QCircuit` on top of an array of quantum operations to represent a quantum circuit. Each element of `QCircuit` can be either a (parametric) unitary gate operation, a quantum channel, or a `QCircuit`. 

After manipulating the quantum circuit, one could apply the quantum circuit onto the quantum state using the`apply!(circ, state)` function. `state` can either be a pure state or a density matrix, which modifies the quantum state in-place. There is also an out-of-place version of this operation, e.g., `apply(circ, state)` or equivalently `circ * state`, which will return a new quantum state and is useful for running variational quantum algorithms. 

```julia
using JuliVQC

state = StateVector(2)
circuit = QCircuit([HGate(1), RyGate(1,pi/4,isparas = false) ,CNOTGate(1,2)])
apply!(circuit,state)
outcome, prob = measure!(state,2)
```



## Qubit operators

The qubit operator is represented as a `QubitsOperator` object in JuliVQC, which can be built as in following example. Once a qubit operator `op` has been initialized, one could apply the function `expectation(op, state)` to evaluate the expectation of it on the quantum state `state`.

```julia
using JuliVQC

function heisenberg_1d(L; hz=1, J=1)
    terms = []
      # one site terms
    for i in 1:L
        push!(terms, QubitsTerm(i=>"z", coeff=hz))
    end
    # nearest-neighbour interactions
    for i in 1:L-1
        push!(terms, QubitsTerm(i=>"x", i+1=>"x", coeff=J))
        push!(terms, QubitsTerm(i=>"y", i+1=>"y", coeff=J))
        push!(terms, QubitsTerm(i=>"z", i+1=>"z", coeff=J))
    end
    return QubitsOperator(terms)
end
```

## Automatic differentiation

JuliVQC has a transparent support for automatic differentiation, one could simply run a variational quantum algorithm in the similar way as a standard quantum algorithm. The major difference from running a standard quantum algorithm is that one wraps the `expectation` function into a loss function, and then use the function `gradient(loss, circ)` to obtain the gradient of the parameters within the quantum circuit. 

```julia
using JuliVQC, Zygote
state = StateVector(3)
op = heisenberg_1d(3) #Construct Heisenberg Hamiltonian as a qubit operator
alpha = 0.01
circ = QCircuit()
for depth in 1:4
    for i in 1:2
        push!(circ,CNOTGate(i,i+1))
    end
    for i in 1:3
        push!(circ,RyGate(i,randn(),isparas=true))
        push!(circ,RxGate(i,randn(),isparas=true))
    end     
end

loss(circ)=real(expectation(op, circ * state))
grad = gradient(loss, circ)[1] # calculate gradient
paras = active_parameters(circ) # extracting the parameters
new_paras = paras - alpha * grad # gradient descent to update parameters
reset_parameters!(circ, new_paras) # reset parameters
```

Under the hood, the gradient is calculated using the Zygote auto-differentiation framework, by rewriting the backpropagation rules of a few elementary operations (the detailed algorithm we use to implement the classical backpropagation  can be found in our paper (https://arxiv.org/abs/2406.19212).
