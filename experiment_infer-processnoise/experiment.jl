# Reactive message-passing in linear Gaussian dynamical system
# Experiment to infer states and process noise simultaneously
#
# Wouter Kouw, BIASlab
# 02-01-2020

using Random
using Revise
using Distributions
using DataStructures
using LightGraphs
using MetaGraphs
using CPUTime
using Plots
pyplot()

# Factor graph components
include(joinpath(@__DIR__, "../factor_nodes/factor_gaussian.jl"))
include(joinpath(@__DIR__, "../factor_nodes/factor_gamma.jl"))
include(joinpath(@__DIR__, "../variables/var_gaussian.jl"))
include(joinpath(@__DIR__, "../variables/var_gamma.jl"))
include(joinpath(@__DIR__, "../variables/var_delta.jl"))
include(joinpath(@__DIR__, "../prob_operations.jl"))
include(joinpath(@__DIR__, "../graph_operations.jl"))
include(joinpath(@__DIR__, "../util.jl"))

# Data
include(joinpath(@__DIR__, "../gen_data.jl"))

"""
Experiment parameters
"""

# Signal time horizon
T = 50

# Reaction-time clock
TT = 10

# Transition and observation matrices
gain = 0.8
emission = 1.0

# Noises (variance form)
process_noise = 2.0
measurement_noise = 0.5

# Parameters for state x_0
px0 = Normal(0.0, 1.0)

# Generate data
Random.seed!(256)
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                mean(px0),
                                inv(var(px0)),
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t | x_t) p(x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph
				 _
				[Γ]
                 |
                (γ)
         ________|________________
         |      _|_       ___    |
(x_0)--  | --->[g_t]---->(x_t)---|
         |                 |     |
         |                _|_    |
         |               [f_t]   |
         |                 |     |
         | T            [[y_t]]  |
		 -------------------------

x_t-1 = previous state variable
g_t = state transition factor node
γ = process noise variable
Γ = process noise prior factor node
x_t = current state variable
f_t = likelihood factor node
y_t = observed variable

---------------------------------
#TODO: Graph specification using a graphical user interface
"""

CPUtic()

# Start graph
global graph = MetaGraph(SimpleGraph(4*T+3))

let

# Set initial state variable
set_props!(graph, 1, Dict(:id => :x_0,
						  :time => 0,
						  :node => VarGaussian(:x_0, marginal=px0, time=0)))

# Connect initial state prior to first state transition
add_edge!(graph, (1, 4))

# Process noise variable
set_props!(graph, 2, Dict(:id => :γ,
						  :time => nothing, #TODO: list of time-points
						  :node => VarGamma(:γ, time=nothing)))

# Process noise prior
set_props!(graph, 3, Dict(:id => :Γ,
						  :time => nothing,
						  :node => FactorGamma(:Γ, out=:γ, shape=1.0, scale=1.0, time=nothing)))

# Connect process noise prior to noise variable
add_edge!(graph, (2, 3))

# Start node numbering
node_num = 3

for t = 1:T

	# Cast current time to string
	t_ = string(t)

	# State transition node
	props_gt = Dict(:id => Symbol("g_"*t_),
				    :time => t,
				    :node => FactorGaussian(Symbol("g_"*t_),
								            out=Symbol("x_"*t_),
	                                        mean=Symbol("x_"*string(t-1)),
	                                        precision=:γ,
	                                        transition=gain))

	# Current state
	props_xt = Dict(:id => Symbol("x_"*t_),
				    :time => t,
				    :node => VarGaussian(Symbol("x_"*t_), time=t))

	# Observation likelihood node
	props_ft = Dict(:id => Symbol("f_"*t_),
				    :time => t,
				    :node => FactorGaussian(Symbol("f_"*t_),
		 						            out=Symbol("y_"*t_),
		 							        mean=Symbol("x_"*t_),
		 							        precision=inv(measurement_noise),
		 							        transition=emission,
		 							        time=t))

	# Observation
	props_yt = Dict(:id => Symbol("y_"*t_),
				    :time => t,
				    :node => VarDelta(Symbol("y_"*t_), observed[t], time=t))

	# Add properties to nodes in current time-slice
	set_props!(graph, node_num+1, props_gt)
	set_props!(graph, node_num+2, props_xt)
	set_props!(graph, node_num+3, props_ft)
	set_props!(graph, node_num+4, props_yt)

	# Add edges between factors and variables
	add_edge!(graph, (node_num-2, node_num+1)) # x_t-1 -- g_t
	add_edge!(graph, (node_num+1, node_num+2)) # g_t -- x_t
	add_edge!(graph, (node_num+2, node_num+3)) # x_t -- f_t
	add_edge!(graph, (node_num+3, node_num+4)) # f_t -- y_t
	add_edge!(graph, (node_num+1, 2)) # g_t -- γ

	# Increment node number
	node_num += 4
end
end

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)

"""
Inference: filtering
"""

function infer!(graph::MetaGraph; start_nodes=[:x_0, :γ], time=T, num_iterations=TT)

	# Start message routine
	for node in start_nodes
		act!(graph, node)
	end

	for t = 1:time

		# Report progress
		if mod(t, time/10) == 1
		  println("At iteration "*string(t)*"/"*string(time))
		end

		# Start clock for reactions
		for tt = 1:num_iterations

			# Iterate over all nodes to react
			for node in nodes_t(graph, t, include_notime=true) #TODO: parallelize
				react!(graph, node)
			end
		end
	end
end

infer!(graph)

CPUtoc()

"""
Visualize experimental results
"""

# Preallocate state vectors
estimated_states = zeros(T,2)
energies = zeros(T,)

# Loop over time
for t = 1:T

	# Loop over nodes in timeslice
	for node in nodes_t(graph, t)

		# Check for state node
		if string(node)[1] == 'x'

			# Retrieve object from node
			node_x = graph[graph[node, :id], :node]

			# Extract moments of marginals
			estimated_states[t,1] = mean(node_x.marginal)
			estimated_states[t,2] = sqrt(var(node_x.marginal))

			# Free energy
			energies[t] = node_x.free_energy

		end
	end
end

# Plot estimated states
scatter(1:T, observed, color="black", label="observations")
plot!(hidden[2:end], color="red", label="true states")
plot!(estimated_states[:,1], color="blue", label="inferred")
plot!(estimated_states[:,1],
	  ribbon=[estimated_states[:,2], estimated_states[:,2]],
	  linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
	  label="")
xlabel!("time (t)")
savefig(joinpath(@__DIR__, "viz/state_estimates_experiment.png"))

# Plot estimated process noise
x_range = 0.05:0.02:4
plot(x_range, pdf.(graph[graph[:γ, :id], :node].marginal, x_range), color="blue", label="posterior")
vline!([inv(process_noise)], color="red", label="true")
xlabel!("process noise (γ⁻¹)")
ylabel!("p(γ⁻¹)")
savefig(joinpath(@__DIR__, "viz/process-noise_estimates_experiment.png"))

# Plot free energy
plot(1:T, energies, color="green", label="")
xlabel!("time (t)")
ylabel!("free energy (F)")
savefig(joinpath(@__DIR__, "viz/free_energy_experiment.png"))
