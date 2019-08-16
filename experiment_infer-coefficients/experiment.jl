# Reactive message-passing in linear Gaussian dynamical system.
# Experiment to infer transition coefficients.
#
# Wouter Kouw, BIASlab
# 15-08-2019

using Revise
using Distributions
using DataStructures
using LightGraphs, MetaGraphs
using Plots
gr()

# Factor graph components
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
T = 100

# Reaction-time clock
TT = 20

# Known transition and observation matrices
transition_coefficient = 0.8
emission_coefficient = 1.0

# Noise parameters
measurement_noise = 0.5
process_noise = 0.5

# Clamped parameters
x_0_params = [0.0, 1.0]
Γ_params = [0.05, 0.01]
A_params = [0.8, 1.0]

# Generate data
observed, hidden = gendata_LGDS(transition_coefficient,
                                emission_coefficient,
                                process_noise,
                                measurement_noise,
                                x_0_params[1],
                                x_0_params[2],
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph
             _     _
          A |_|   |_| Γ
             |     |
         a_t ∘     ∘ γ_t
             |_____|       x_t
.. ---∘---->|_______|----->|=|-----> ...
    x_t-1      g_t          |
                            |
                           |_| f_t
                            |
                            ∘
                           y_t

x_t-1   = previous state edge
g_t     = state transition edge
γ_t     = process noise edge
γ       = process noise prior node
a_t     = transition coefficient edge
A       = transition coefficient prior node
x_t     = current state edge
f_t     = likelihood node
y_t     = observation node
"""

# Start graph
graph = MetaGraph(SimpleGraph(9))

# Previous state edge
x_tmin = EdgeGaussian("x_tmin")
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_tmin, :id => "x_tmin"))

# State transition node
g_t = TransitionGaussian("g_t")
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_t, :id => "g_t"))

# Process noise edge
γ_t = EdgeGamma("γ_t")
set_props!(graph, 3, Dict{Symbol,Any}(:object => :γ_t, :id => "γ_t"))

# Process noise prior node
Γ = NodeGamma("Γ")
set_props!(graph, 4, Dict{Symbol,Any}(:object => :Γ, :id => "Γ"))

# Transition coefficient edge
a_t = EdgeGaussian("a_t")
set_props!(graph, 5, Dict{Symbol,Any}(:object => :a_t, :id => "a_t"))

# Transition coefficient prior node
A = NodeGaussian("A")
set_props!(graph, 6, Dict{Symbol,Any}(:object => :A, :id => "A"))

# Current state edge
x_t = EdgeGaussian("x_t")
set_props!(graph, 7, Dict{Symbol,Any}(:object => :x_t, :id => "x_t"))

# Observation likelihood node
f_t = LikelihoodGaussian("f_t")
set_props!(graph, 8, Dict{Symbol,Any}(:object => :f_t, :id => "f_t"))

# Observation edge
y_t = EdgeDelta("y_t")
set_props!(graph, 9, Dict{Symbol,Any}(:object => :y_t, :id => "y_t"))

add_edge!(graph, 1, 2) # x_t-1 -- g_t
add_edge!(graph, 2, 3) # g_t -- γ_t
add_edge!(graph, 3, 4) # γ_t -- Γ
add_edge!(graph, 2, 5) # g_t -- a_t
add_edge!(graph, 5, 6) # a_t -- A
add_edge!(graph, 2, 7) # g_t -- x_t
add_edge!(graph, 7, 8) # x_t -- f_t
add_edge!(graph, 8, 9) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)
set_indexing_prop!(graph, :object)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_noises = zeros(T, 2, TT)
estimated_transition = zeros(T, 2, TT)

# Set state prior x_0
global x_t = EdgeGaussian("x_0"; mean=x_0_params[1], precision=x_0_params[2])
global a_t = EdgeGaussian("a_0"; mean=0.0, precision=0.1)
global γ_t = EdgeGamma("γ_0"; shape=0.01, rate=0.01)

for t = 1:T
    # t=1

      # Report progress
      if mod(t, T/10) == 1
          println("At iteration "*string(t)*"/"*string(T))
      end

      # Previous state
      global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision)

      # State transition node
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_precision="γ_t", edge_transition="a_t")

      # Process noise edge
      global γ_t = EdgeGamma("γ_t", shape=γ_t.shape, rate=γ_t.rate)

      # Process noise prior node
      global Γ = NodeGamma("Γ", edge_data="γ_t", edge_shape=Γ_params[1], edge_rate=Γ_params[2])

      # Transition coefficient edge
      global a_t = EdgeGaussian("a_t", mean=a_t.mean, precision=a_t.precision)

      # Transition coefficient prior node
      global A = NodeGaussian("A", edge_data="a_t", edge_mean=A_params[1], edge_precision=A_params[2])

      # Current state
      global x_t = EdgeGaussian("x_t", mean=x_tmin.mean, precision=x_tmin.precision)

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Start message routine
      act(x_tmin, belief(x_tmin), 1e12, graph);

      # Start clock for reactions
      for tt = 1:TT

          react(g_t, graph)
          react(γ_t, graph)
          react(Γ, graph)
          react(a_t, graph)
          react(A, graph)
          react(x_t, graph)
          react(f_t, graph)
          react(y_t, graph)

          # Write out estimated state parameters
          estimated_states[t, 1, tt] = x_t.mean
          estimated_states[t, 2, tt] = sqrt(1/x_t.precision)
          estimated_noises[t, 1, tt] = γ_t.shape / γ_t.rate
          estimated_noises[t, 2, tt] = γ_t.shape / γ_t.rate^2
          estimated_transition[t, 1, tt] = a_t.mean
          estimated_transition[t, 2, tt] = sqrt(1/a_t.precision)
      end
end

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
savefig(pwd()*"/experiment_infer-coefficients/viz/state_estimates.png")

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
savefig(pwd()*"/experiment_infer-coefficients/viz/state_parameter_trajectory_t" * string(t) * ".png")

# Visualize final transition coefficient estimates over time
plot(transition_coefficient*ones(T,1), color="black", label="transition")
plot!(estimated_transition[:,1,end], color="blue", label="estimates")
plot!(estimated_transition[:,1,end],
      ribbon=[estimated_transition[:,2,end], estimated_transition[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Transition coefficient estimates, q(a_t)")
savefig(pwd()*"/experiment_infer-coefficients/viz/transition_estimates.png")

# Visualize transition coefficient belief parameter trajectory
t = T
plot(estimated_transition[t,1,1:end], color="blue", label="q(a_"*string(t)*")")
plot!(estimated_transition[t,1,1:end],
      ribbon=[estimated_transition[t,2,1:end], estimated_transition[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(a_t) for t="*string(t))
savefig(pwd()*"/experiment_infer-coefficients/viz/transition_parameter_trajectory_t" * string(t) * ".png")
