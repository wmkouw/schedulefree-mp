# Reactive message-passing in linear Gaussian dynamical system
# Experiment to infer process noise
#
# Wouter Kouw, BIASlab
# 15-08-2019

using Random
using Revise
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
T = 80

# Reaction-time clock
TT = 10

# Known transition and observation matrices
gain = 1.0
emission = 1.0

# Noise parameters
process_noise = 1.2
measurement_noise = 0.3

# Clamped parameters
x_0 = [0.0, 0.1]
γ_0 = [1.0, 1.0]

# Generate data
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                x_0[1],
                                x_0[2],
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph

     γ′_x     e_γ
...   ∘------>[=]                     ...
               |
               ∘ γ_x
               |
     x_t-1     |     x'_t    e_x
...   ∘----->[g_t]----∘------[=]      ...
                              |
                              ∘ x_t
                              |
                            [f_t]
                              |
                              ⊡
                             y_t

x_t-1 = previous state edge
g_t = state transition edge
γ_x = process noise edge
γ'_x = previous process noise
e_γ = process noise equality node
x_t = current state edge
x'_t = current state edge
e_x = current state equality node
f_t = likelihood node
y_t = observation node
"""

# Start graph
graph = MetaGraph(SimpleGraph(10))

# Edge variable: previous state
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_tmin, :id => "x_tmin"))

# Factor node: state transition
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_t, :id => "g_t"))

# Edge variable: current process noise
set_props!(graph, 3, Dict{Symbol,Any}(:object => :γ_x, :id => "γ_x"))

# Edge variable: previous process noise
set_props!(graph, 4, Dict{Symbol,Any}(:object => :γ_xp, :id => "γ_xp"))

# Factor node: process noise equality
set_props!(graph, 5, Dict{Symbol,Any}(:object => :e_γ, :id => "e_γ"))

# Edge variable: current state
set_props!(graph, 6, Dict{Symbol,Any}(:object => :x_t, :id => "x_t"))

# Factor node: current state equality
set_props!(graph, 7, Dict{Symbol,Any}(:object => :e_x, :id => "e_x"))

# Edge variable: current state
set_props!(graph, 8, Dict{Symbol,Any}(:object => :x_tp, :id => "x_tp"))

# Factor node: likelihood
set_props!(graph, 9, Dict{Symbol,Any}(:object => :f_t, :id => "f_t"))

# Edge variable: observation
set_props!(graph, 10, Dict{Symbol,Any}(:object => :y_t, :id => "y_t"))

# Add edges between factor nodes and variables
add_edge!(graph, 1, 2) # x_t-1 -- g_t
add_edge!(graph, 2, 3) # g_t -- γ_x
add_edge!(graph, 3, 5) # γ_x -- e_γ
add_edge!(graph, 4, 5) # γ_xp -- e_γ
add_edge!(graph, 2, 6) # g_t -- x_t
add_edge!(graph, 6, 7) # x_t -- e_x
add_edge!(graph, 7, 8) # e_x --- x_tp
add_edge!(graph, 8, 9) # x_tp -- f_t
add_edge!(graph, 9, 10) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)
set_indexing_prop!(graph, :object)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_noises = zeros(T, 2, TT)
free_energy_gradients = zeros(T, TT)
check = zeros(T, TT)

# Set initial priors
global x_t = EdgeGaussian("x_0"; mean=x_0[1], precision=x_0[2])
global γ_x = EdgeGamma("γ_0"; shape=γ_0[1], scale=γ_0[2])

# for t = 1:T
    t=1

      # Report progress
      if mod(t, T/10) == 1
          println("At iteration "*string(t)*"/"*string(T))
      end

      # Previous state
      global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision)

      # Previous previous noise
      global γ_xp = EdgeGamma("γ_xp", shape=γ_x.shape, scale=γ_x.scale, block=true)

      # State transition node
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_precision="γ_x")

      # Current process noise
      global γ_x = EdgeGamma("γ_x", shape=1.0, scale=1.0)

      # Process noise equality node
      global e_γ = NodeEquality("e_γ", edges=["γ_x", "γ_xp"])

      # Current state
      global x_t = EdgeGaussian("x_t", mean=0.0, precision=1.0)

      # State equality node
      global e_x = NodeEquality("e_x", edges=["x_t", "x_tp"])

      # Current state
      global x_tp = EdgeGaussian("x_tp", mean=0.0, precision=1.0)

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Start message routine
      act(x_tmin, belief(x_tmin), 1e3, graph);
      act(γ_xp, belief(γ_xp), 1e3, graph);

      # Start clock for reactions
      for tt = 1:TT

          # Write out estimated state parameters
          estimated_states[t, 1, tt] = x_t.mean
          estimated_states[t, 2, tt] = sqrt(1/x_t.precision)
          estimated_noises[t, 1, tt] = γ_x.shape * γ_x.scale
          estimated_noises[t, 2, tt] = sqrt(γ_x.shape * γ_x.scale^2)
          check[t,tt] = γ_xp.shape * γ_xp.scale

          # Keep track of FE gradients
          free_energy_gradients[t, tt] = x_t.grad_free_energy

          react(g_t, graph)
          react(γ_x, graph)
          react(e_γ, graph)
          react(x_t, graph)
          react(e_x, graph)
          react(x_tp, graph)
          react(f_t, graph)
          react(y_t, graph)
      end
# end

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
savefig(pwd()*"/experiment_infer-processnoise/viz/state_estimates.png")

# Visualize final noise estimates over time
plot(inv(process_noise)*ones(T,1), color="black", label="true process noise", linewidth=2)
plot!(estimated_noises[:,1,end], color="blue", label="estimates")
plot!(estimated_noises[:,1,end],
      ribbon=[estimated_noises[:,2,end], estimated_noises[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Noise estimates, q(γ_x)")
savefig(pwd()*"/experiment_infer-processnoise/viz/noise_estimates.png")

# Visualize state belief parameter trajectory
t = T
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
t = 1
plot(estimated_noises[t,1,1:end], color="blue", label="q(γ_"*string(t)*")")
plot!(estimated_noises[t,1,1:end],
      ribbon=[estimated_noises[t,2,1:end], estimated_noises[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(γ_x) for t="*string(t))
savefig(pwd()*"/experiment_infer-processnoise/viz/noise_parameter_trajectory_t" * string(t) * ".png")

# Visualize free energy gradients over time-series
plot(free_energy_gradients[1:end,end], color="black", label="||dF||_t")
xlabel!("time (t)")
ylabel!("Norm of free energy gradient")
savefig(pwd()*"/experiment_infer-processnoise/viz/FE_gradients.png")

# Visualize FE gradient for a specific time-step
t = T
plot(free_energy_gradients[t,:], color="blue", label="||dF||_t"*string(t))
xlabel!("iterations")
ylabel!("Norm of free energy gradient")
savefig(pwd()*"/experiment_infer-processnoise/viz/FE-gradient_trajectory_t" * string(t) * ".png")
