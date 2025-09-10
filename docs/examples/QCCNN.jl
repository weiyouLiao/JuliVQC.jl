include("../find_meteor.jl")

using PyCall

using Zygote
using Zygote: @adjoint
import Base.*

using LinearAlgebra: dot, norm
using SparseArrays
using Random

using JSON

using Flux
using Flux.Optimise

using Meteor.QuantumCircuit
using Meteor.Diff
import Meteor.Diff: set_parameters_impl!, collect_variables_impl!

num_parameters(L::Int, depth::Int) = L * depth

py"""

from numpy.random import randn, seed
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.utils import QuantumInstance

from qiskit.opflow import (I, X, Y, Z, StateFn, Zero, One, Plus, Minus, H,
								   DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)
from qiskit.opflow import PauliExpectation, AerPauliExpectation, CircuitSampler

from qiskit.providers.aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

def build_noise_model():
	p_reset = 0.0
	p_meas = 0.0
	p_gate1 = 0.001

	# QuantumError objects
	error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
	error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
	error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
	error_gate2 = error_gate1.tensor(error_gate1)

	# Add errors to noise model
	noise_bit_flip = NoiseModel()
	noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
	noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
	noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
	noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

	return noise_bit_flip

def build_thermal_noise_model(L):
	# # T1 and T2 values for qubits 0-3
	# T1s = np.random.normal(50e3, 10e3, L) # Sampled from normal distribution mean 50 microsec
	# T2s = np.random.normal(70e3, 10e3, L)  # Sampled from normal distribution mean 50 microsec

	# # Truncate random T2s <= T1s
	# T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(L)])
	T1s = np.array([51771.57935748, 55850.72054168, 47846.21723976, 48329.55949527])
	T2s = np.array([44370.62950504, 85076.0034188 , 60774.66631507, 70040.02133599])

	# Instruction times (in nanoseconds)
	time_u1 = 0   # virtual gate
	time_u2 = 50  # (single X90 pulse)
	time_u3 = 100 # (two X90 pulses)
	time_cx = 300
	time_reset = 1000  # 1 microsecond
	time_measure = 1000 # 1 microsecond

	# QuantumError objects
	errors_reset = [thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)]
	errors_measure = [thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)]
	errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
			  for t1, t2 in zip(T1s, T2s)]
	errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
			  for t1, t2 in zip(T1s, T2s)]
	errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
			  for t1, t2 in zip(T1s, T2s)]
	errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
			 thermal_relaxation_error(t1b, t2b, time_cx))
			  for t1a, t2a in zip(T1s, T2s)]
			   for t1b, t2b in zip(T1s, T2s)]

	# Add errors to noise model
	noise_thermal = NoiseModel()
	for j in range(L):
		noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
		noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
		noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
		noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
		noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
		for k in range(L):
			noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

	return noise_thermal

def qfeature_mapping(L, depth, init_paras, paras):
	expec_op = StateFn(Z^L).adjoint()

	circuit = QuantumCircuit(L)

	if len(init_paras) != L:
		raise ValueError("wrong size of init paras.")

	for j in range(L):
		circuit.ry(init_paras[j], j)

	for j in range(L-1):
		circuit.cx(j, j+1)

	n = 0
	for i in range(depth):
		for j in range(L):
			circuit.ry(paras[n], j)
			n += 1
		for j in range(L-1):
			circuit.cx(j, j+1)

	circuit = CircuitStateFn(circuit)
	if n != len(paras):
		raise ValueError("wrong number of paras.")

	expec = AerPauliExpectation().convert(expec_op.compose(circuit))

	backend = Aer.get_backend('qasm_simulator')

	# noise_model = build_noise_model()
	noise_model = build_thermal_noise_model(L)
	# noise_model = None
	q_instance = QuantumInstance(backend, noise_model=noise_model, shots=10000)

	sampler = CircuitSampler(q_instance).convert(expec)

	return sampler.eval().real

"""

qfeature_mapping(L::Int, depth::Int, init_paras::AbstractVector{<:Real},
	paras::AbstractVector{<:Real}) = py"qfeature_mapping"(L, depth, init_paras, paras)

# function qfeature_mapping(L::Int, depth::Int, init_paras::AbstractVector, paras::AbstractVector)
# 	op = QubitsTerm(Dict(i=>"Z" for i in 1:L))

#     qureg = qstate(Float64, L)
#     circuit = QCircuit()
#     (length(init_paras)==L) || error("wrong number of initial parameters.")
#     (length(paras)==num_parameters(L, depth)) || error("wrong number of parameters.")
#     for i in 1:L
#         push!(circuit, RyGate(i, init_paras[i]))
#     end

#     for i in 1:L-1
#     	push!(circuit, CNOTGate((i, i+1)))
#     end

#     n = 1
#     for d in 1:depth
#         for i in 1:L
#             push!(circuit, RyGate(i, paras[n]))
#             n += 1
#         end
#         for i in 1:L-1
#         	push!(circuit, CNOTGate((i, i+1)))
#     	end
#     end
#     apply!(circuit, qureg)
#     return expectation(op, qureg)
# end


# function qfeature_mapping_grad(L::Int, depth::Int, init_paras::AbstractVector{<:Real}, paras::AbstractVector{<:Real})
#     grad0 = Vector{Float64}(undef, length(init_paras))
#     paras_p = copy(init_paras)
#     paras_m = copy(init_paras)
#     for i in 1:length(init_paras)
#         paras_p[i] = init_paras[i] + pi / 2
#         paras_m[i] = init_paras[i] - pi / 2
#         grad0[i] = (qfeature_mapping(L, depth, paras_p, paras) - qfeature_mapping(L, depth, paras_m, paras)) / 2
#         paras_p[i] = init_paras[i]
#         paras_m[i] = init_paras[i]
#     end

#     grad = Vector{Float64}(undef, length(paras))
#     paras_p = copy(paras)
#     paras_m = copy(paras)
#     for i in 1:length(paras)
#         paras_p[i] = paras[i] + pi / 2
#         paras_m[i] = paras[i] - pi / 2
#         grad[i] = (qfeature_mapping(L, depth, init_paras, paras_p) - qfeature_mapping(L, depth, init_paras, paras_m)) / 2
#         paras_p[i] = paras[i]
#         paras_m[i] = paras[i]
#     end
#     return grad0, grad
# end

function qfeature_mapping_grad(L::Int, depth::Int, init_paras::AbstractVector{<:Real}, paras::AbstractVector{<:Real})

	grad = Vector{Float64}(undef, length(paras))
	paras_p = copy(paras)
	paras_m = copy(paras)
	for i in 1:length(paras)
		paras_p[i] = paras[i] + pi / 2
		paras_m[i] = paras[i] - pi / 2
		grad[i] = (qfeature_mapping(L, depth, init_paras, paras_p) - qfeature_mapping(L, depth, init_paras, paras_m)) / 2
		paras_p[i] = paras[i]
		paras_m[i] = paras[i]
	end
	return nothing, grad
end


struct QCNNLayer
	storage::Vector{Vector{Float64}}
	filter_shape::Tuple{Int, Int}
	padding::Int
	depth::Int
end

function QCNNLayer(filter_shape::Tuple{Int, Int}, nfilter::Int; depth::Int, padding::Int)
	s1, s2 = filter_shape
	L = s1 * s2
	n = num_parameters(L, depth)
	storage = [randn(n) for i in 1:nfilter]
	return QCNNLayer(storage, filter_shape, padding, depth)
end

collect_variables_impl!(a::Vector, x::QCNNLayer) = collect_variables_impl!(a, x.storage)
set_parameters_impl!(x::QCNNLayer, a::Vector, start_pos::Int) = set_parameters_impl!(x.storage, a, start_pos)

get_all(x::QCNNLayer) = x.storage, x.filter_shape, x.padding, x.depth
@adjoint get_all(x::QCNNLayer) = get_all(x), z -> (z[1],)

function add_padding(m::AbstractArray{T, 3}, padding::Int) where {T <: Real}
	s1 = size(m, 1)
	s2 = size(m, 2)
	m1 = zeros(T, s1 + 2*padding, s2 + 2*padding, size(m, 3))
	m1[(padding+1):(padding+s1), (padding+1):(padding+s2), :] = m
	return m1
end

@adjoint add_padding(m::AbstractArray{T, 3}, padding::Int) where {T <: Real} = begin
	s1 = size(m, 1)
	s2 = size(m, 2)
	return add_padding(m, padding), z -> (z[(padding+1):(padding+s1), (padding+1):(padding+s2), :], nothing)
end

# @adjoint qfeature_mapping(L::Int, depth::Int, init_paras::AbstractVector{<:Real}, paras::AbstractVector{<:Real}) = begin
#     qfeature_mapping(L, depth, init_paras, paras), z -> begin
#         a, b = qfeature_mapping_grad(L, depth, init_paras, paras)
#         return (nothing, nothing, z .* a, z .* b)
#     end
# end

@adjoint qfeature_mapping(L::Int, depth::Int, init_paras::AbstractVector{<:Real}, paras::AbstractVector{<:Real}) = begin
	qfeature_mapping(L, depth, init_paras, paras), z -> begin
		a, b = qfeature_mapping_grad(L, depth, init_paras, paras)
		return (nothing, nothing, nothing, z .* b)
	end
end

function qcnn_single_filter(m::Array{<:Real, 3}, paras::Vector{Float64}, filter_shape::Tuple{Int, Int}, depth::Int)
	n1, n2, n3 = size(m)
	s1, s2 = filter_shape
	(s1 <= n1 && s2 <= n2) || error("filter size too large.")
	out = Zygote.Buffer(m, n1-s1+1, n2-s2+1, n3)
	L = s1 * s2
	for i in 1:(n1-s1+1)
		for j in 1:(n2-s2+1)
			for k in 1:n3
				out[i, j, k] = qfeature_mapping(L, depth, reshape(m[i:(i+s1-1), j:(j+s2-1), k], L), paras)
			end
		end
	end
	return copy(out)
end

function apply(x::QCNNLayer, m::Array{<:Real, 3})
	storage, filter_shape, padding, depth = get_all(x)
	n = length(storage)
	m2 = add_padding(m, padding)
	tmp = qcnn_single_filter(m2, storage[1], filter_shape, depth)
	s1, s2, s3 = size(tmp)
	r = Zygote.Buffer(m, s1, s2, s3, n)
	r[:, :, :, 1] = tmp
	for i in 2:n
		r[:, :, :, i] = qcnn_single_filter(m2, storage[i], filter_shape, depth)
	end
	return reshape(copy(r), s1, s2, s3 * n)
end

apply(x::QCNNLayer, m::Array{<:Real, 2}) = apply(x, reshape(m, size(m,1), size(m,2), 1))
*(x::QCNNLayer, m::Array) = apply(x, m)


function _max_pooling_impl(m::Array{Float64, 3}, filter_shape::Tuple{Int, Int})
	n1, n2, n3 = size(m)
	s1, s2 = filter_shape
	(s1 <= n1 && s2 <= n2) || error("filter size too large.")
	out = Zygote.Buffer(m, n1-s1+1, n2-s2+1, n3)
	for i in 1:(n1-s1+1)
		for j in 1:(n2-s2+1)
			for k in 1:n3
				out[i, j, k] = maximum(m[i:(i+s1-1), j:(j+s2-1), k])
			end
		end
	end
	return copy(out)
end
max_pooling(m::Array{Float64, 3}, filter_shape::Tuple{Int, Int}, padding::Int=0) = _max_pooling_impl(
	add_padding(m, padding), filter_shape)

function one_hot(i, d)
	i = convert(Int, i)
	d = convert(Int, d)
	(i >= 1 && i <= d) || error("out of range.")
	r = zeros(Float64, d)
	r[i] = 1
	return r
end

function prepare_data_4(path)
	data = JSON.parsefile(path)
	training = []
	testing = []
	LABEL_MAPPING = Dict("S"=>1, "T"=>2, "L"=>3, "O"=>4)
	L = length(LABEL_MAPPING)
	for (label, a) in data
		v = get(LABEL_MAPPING, label, nothing)
		if v !== nothing
			vj = one_hot(v, L)
			for item in a
				push!(training, hcat(item...))
				push!(testing, vj)
			end
		end
	end
	return [training...], [testing...]
end

# function prepare_data_2(path)
#     data = JSON.parsefile(path)
#     training = []
#     testing = []
#     LABEL_MAPPING = Dict("S"=>1, "T"=>2)
#     L = length(LABEL_MAPPING)
#     for (label, a) in data
#         v = get(LABEL_MAPPING, label, nothing)
#         if v !== nothing
#             vj = one_hot(v, L)
#             for item in a
#                 push!(training, hcat(item...))
#                 push!(testing, vj)
#             end
#         end
#     end
#     return [training...], [testing...]
# end

function prepare_data_2(path, n)
	data = JSON.parsefile(path)
	training = []
	testing = []
	LABEL_MAPPING = Dict("S"=>1, "T"=>2)
	L = length(LABEL_MAPPING)
	for (label, a) in data
		v = get(LABEL_MAPPING, label, nothing)
		if v !== nothing
			vj = one_hot(v, L)
			cnt = 0
			for item in a
				if cnt < n
					push!(training, hcat(item...))
					push!(testing, vj)
					cnt += 1
				end
			end
		end
	end
	return [training...], [testing...]
end

function conv_net(input::Array{<:Real, 2}, layer::QCNNLayer, m::AbstractMatrix)   # first quantum conv layer
	out1 = layer * input

	# maximum pooling
	out2 = max_pooling(out1, (2, 2), 0)
#     println(size(out2))


	# softmax
	return softmax(m * reshape(out2, length(out2)))
end	

function main(sed::Int)

	Random.seed!(sed)

	train_path = "data/tetris_train.json"

	test_path = "data/tetris_test.json"

	x_train, y_train = prepare_data_2(train_path, 40)

	x_test, y_test = prepare_data_2(test_path, 10000)


	# x_train, y_train = prepare_data_2(train_path, 1)

	# x_test, y_test = prepare_data_2(test_path, 1)


	println("number of training $(length(x_train)), number of testing $(length(x_test)).")

	function loss(layer::QCNNLayer, m::AbstractMatrix)
		r = 0.
		for (x, y) in zip(x_train, y_train)
		   r += distance2(conv_net(x, layer, m), y)
		end
		return r / length(x_train)
	end

	function train(depth=1, learn_rate=0.01, epoches=100)
		filter_shape = (2,2)
		L = prod(filter_shape)
		nfilter = 5
		padding = 0

		layer = QCNNLayer(filter_shape, nfilter, depth=depth, padding=padding)

		dm = randn(2, nfilter)

		_predict(input::Array{<:Real, 2}) = begin
			r = conv_net(input, layer, dm)
			return argmax(r)
		end

		_accuracy(x, y) = begin
			r = [_predict(x[i])== argmax(y[i]) for i in 1:length(x)]
			return sum(r)/length(r)
		end

		x0 = parameters(layer, dm)
		println("total number of parameters $(length(x0))")

		@time r = loss(layer, dm)
		println("initial loss is $r")

		println("score before training $(_accuracy(x_test, y_test)).")

		#learn_rate = 0.05
		opt = ADAM(learn_rate)

		accuracies = []
		losses = []


		for i in 1:epoches
			train_loss, back = Zygote.pullback(loss, layer, dm)
			@time grad = parameters(back(one(train_loss)))
			Optimise.update!(opt, x0, grad)
			set_parameters!(x0, layer, dm)

			ac = _accuracy(x_test, y_test)
			push!(accuracies, ac)
			push!(losses, train_loss)
			println("accuracy at the $i-th step is $ac, loss is $(train_loss).")
		end
		return [accuracies...], [losses...]
	end	

	depth = 2
	epoches = 200
	learn_rate = 0.05
	# accuracies = zeros(epoches)
	# losses = zeros(epoches)
	# for i in 1:10
 #    	a, b = train(learn_rate, epoches)
 #    	accuracies .+= a
	#     losses .+= b
	# end
	accuracies, losses = train(depth, learn_rate, epoches)

	file_name = "result/NoiseQCNNOneLayer2LabelDepth$(depth)Seed$(sed).json"
	println("save results to file $file_name")

	result = JSON.json(Dict("accuracies"=>accuracies, "losses"=>losses))
	open(file_name, "w") do f
		write(f, result)
	end	
end


# filter_shape = (2,3)
# L = prod(filter_shape)
# depth = 2
# nfilter = 2
# padding = 1

# layer = QCNNLayer(filter_shape, nfilter, depth=depth, padding=padding)

# m = randn(4,3,2)

# tmp1(a) = sum(apply(a, m))

# println("gradient is correct? $(check_gradient(tmp1, layer, verbose=2)).")




main(100)


# L = 4
# depth = 1
# init_paras = randn(L)
# paras = randn(L * depth)

# for i in 1:10
# 	@time v = qfeature_mapping(L, depth, init_paras, paras)
# 	@time grad = qfeature_mapping_grad(L, depth, init_paras, paras)
# 	println("the $i-th iter, with value $v.")
# end



