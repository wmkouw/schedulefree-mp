# Reactive message-passing in linear Gaussian dynamical system
# Experiment to infer process noise
#
# Wouter Kouw, BIASlab
# 15-08-2019

using Random
using Revise
using CPUTime
using JLD
using Distributions
using DataStructures
using LightGraphs, MetaGraphs
using Plots
pyplot()

# Factor graph components
include("../nodes/node_equality.jl")
include("../nodes/node_gamma.jl")
include("../nodes/node_gaussian.jl")
include("../nodes/transition_gaussian.jl")
include("../nodes/likelihood_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_gamma.jl")
include("../edges/edge_delta.jl")
include("../util.jl")

# Data
include("../gen_data.jl")


"""
Experiment parameters
"""

# Signal time horizon
T = 50

# Reaction-time clock
TT = 10

# Known transition and observation matrices
gain = 1.0
emission = 1.0

# Noise parameters
process_noise = 0.8
measurement_noise = 0.3

# Clamped parameters
params_x0 = [0.0, 1.0]
params_γ0 = [1.0, 1.0]

# Generate data
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                params_x0[1],
                                params_x0[2],
                                time_horizon=T)

# # Write data set to file
# save(pwd()*"/experiment_infer-processnoise/data/LGDS_T"*string(T)*".jld", "observed", observed, "hidden", hidden)

# X = load(pwd()*"/experiment_infer-processnoise/data/LGDS_T"*string(T)*".jld")
# observed = X["observed"]
# hidden = X["hidden"]

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t | x_t) p(x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph


... (γ_t-1)----[e_γ]               ...
                 |
                _|_
               (γ_t)
                 |
                _|_       ___
... (x_t-1)--->[g_t]---->(x_t)     ...
                           |
                          _|_
                         [f_t]
                           |
                           ⊡
                          y_t

x_t-1 = previous state edge
γ_t-1 = previous process noise
g_t = state transition edge
e_γ = process noise equality
γ_t = process noise edge
x_t = current state edge
f_t = likelihood node
y_t = observation node
"""

CPUtic()

# Start graph
graph = MetaGraph(SimpleGraph(8))

# Edge variable: previous state
set_props!(graph, 1, Dict{Symbol,Any}(:id => "x_tmin", :type => "variable"))

# Edge variable: previous process noise
set_props!(graph, 2, Dict{Symbol,Any}(:id => "γ_tmin", :type => "variable"))

# Factor node: state transition
set_props!(graph, 3, Dict{Symbol,Any}(:id => "g_t", :type => "factor"))

# Edge variable: current process noise
set_props!(graph, 4, Dict{Symbol,Any}(:id => "γ_t", :type => "variable"))

# Factor node: process noise equality
set_props!(graph, 5, Dict{Symbol,Any}(:id => "e_γ", :type => "factor"))

# Edge variable: current state
set_props!(graph, 6, Dict{Symbol,Any}(:id => "x_t", :type => "variable"))

# Factor node: likelihood
set_props!(graph, 7, Dict{Symbol,Any}(:id => "f_t", :type => "factor"))

# Edge variable: observation
set_props!(graph, 8, Dict{Symbol,Any}(:id => "y_t", :type => "variable"))

# Add edges between factor nodes and variables
add_edge!(graph, 1, 3) # x_t-1 -- g_t
add_edge!(graph, 2, 5) # γ_t-1 -- e_γ
add_edge!(graph, 3, 4) # g_t -- γ_t
add_edge!(graph, 3, 6) # g_t -- x_t
add_edge!(graph, 4, 5) # γ_t -- e_γ
add_edge!(graph, 6, 7) # x_t -- f_t
add_edge!(graph, 7, 8) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_noises = zeros(T, 2, TT)
free_energies = zeros(T, TT)

# Set initial priors
global x_t = EdgeGaussian("x_0"; mean=params_x0[1], precision=params_x0[2])
global γ_t = EdgeGamma("γ_0"; shape=params_γ0[1], scale=params_γ0[2])

for t = 1:T

      # Report progress
      if mod(t, T/10) == 1
          println("At iteration "*string(t)*"/"*string(T))
      end

      # Previous state
      global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision, block=true)

      # Previous process noise
      global γ_tmin = EdgeGamma("γ_tmin", shape=γ_t.shape, scale=γ_t.scale, block=true)

      # Process noise equality node
      global e_γ = NodeEquality("e_γ", edges=["γ_tmin", "γ_t"])

      # State transition node
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_transition=gain, edge_precision="γ_t")

      # Current process noise
      global γ_t = EdgeGamma("γ_t")

      # Current state
      global x_t = EdgeGaussian("x_t")

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Start message routine
      act(x_tmin, belief(x_tmin), 1e3, graph);
      act(γ_tmin, belief(γ_tmin), 1e3, graph);
      act(y_t, belief(y_t), 1e3, graph);

      # Start clock for reactions
      for tt = 1:TT
      # tt=1

          # Write out estimated state parameters
          estimated_states[t, 1, tt] = mean(x_t)
          estimated_states[t, 2, tt] = sqrt(var(x_t))
          estimated_noises[t, 1, tt] = inv(mean(γ_t))
          estimated_noises[t, 2, tt] = sqrt(var(γ_t))

          # Iterate over nodes
          react(e_γ, graph)
          react(g_t, graph)
          react(f_t, graph)

          # Iterate over edges
          react(γ_t, graph)
          react(x_t, graph)

      end
end

CPUtoc()

"""
Visualize experimental results
"""

# Visualize final state estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[estimated_states[:,2,end], estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", markersize=3, label="observations")
xlabel!("time (t)")
title!("State estimates, q(x_t)")
savefig(pwd()*"/experiment_infer-processnoise/viz/experiment_state_estimates.png")

# Visualize final noise estimates over time
plot(process_noise*ones(T,1), color="black", label="true process noise", linewidth=2)
plot!(estimated_noises[:,1,end], color="blue", label="estimates")
plot!(estimated_noises[:,1,end],
      ribbon=[estimated_noises[:,2,end], estimated_noises[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Noise estimates, q(γ_t)")
savefig(pwd()*"/experiment_infer-processnoise/viz/experiment_noise_estimates.png")

# Visualize state belief parameter trajectory
t = Integer(T/2)
plot(estimated_states[t,1,1:end], color="blue", label="q(x_"*string(t)*")")
plot!(estimated_states[t,1,1:end],
      ribbon=[estimated_states[t,2,1:end], estimated_states[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(x_t) for t="*string(t))
savefig(pwd()*"/experiment_infer-processnoise/viz/state_parameter_trajectory_t" * string(t) * ".png")

# Visualize noise belief parameter trajectory
t = Integer(T/2)
plot(estimated_noises[t,1,1:end], color="blue", label="q(γ_"*string(t)*")")
plot!(estimated_noises[t,1,1:end],
      ribbon=[estimated_noises[t,2,1:end], estimated_noises[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(γ_t) for t="*string(t))
savefig(pwd()*"/experiment_infer-processnoise/viz/noise_parameter_trajectory_t" * string(t) * ".png")
