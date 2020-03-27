# Reactive message-passing in linear Gaussian dynamical system
# Experiment to pass messages without a schedule
#
# Wouter Kouw, BIASlab
# 12-08-2019

using Random
using Revise
using CPUTime
using JLD
using ForneyLab
using Plots
pyplot()

# Functions for generating data
include(joinpath(@__DIR__, "../gen_data.jl"))

"""
Experiment parameters
"""

# Signal time horizon
T = 500

# Reaction-time clock
TT = 10

# Known transition and observation matrices
gain = 0.7
emission = 1.0

# Noise parameters
process_noise = 0.6
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

# # Load previously generated data
# X = load(pwd()*"/experiment_infer-processnoise/data/LGDS_T"*string(T)*".jld")
# observed = X["observed"]
# hidden = X["hidden"]

"""
Model specification
"""

# Start timer
CPUtic()

g = FactorGraph()

@RV γ ~ Gamma(params_γ0[1], params_γ0[2])
@RV x_0 ~ GaussianMeanVariance(params_x0[1], params_x0[2])

global x = Vector{Variable}(undef, T)
global y = Vector{Variable}(undef, T)
global x_tmin = x_0
for t = 1:T

    @RV x[t] ~ GaussianMeanPrecision(x_tmin, γ)
    @RV y[t] ~ GaussianMeanVariance(x[t], measurement_noise)

    global x_tmin = x[t]

    placeholder(y[t], :y, index=t)
end

"""
Mean-field algorithm generation
"""

# Define recognition factorization
RecognitionFactorization()

q_γ = RecognitionFactor(γ)
q_x_0 = RecognitionFactor(x_0)

q_x = Vector{RecognitionFactor}(undef, T)
for t=1:T
    q_x[t] = RecognitionFactor(x[t])
end

# Compile algorithm
algo_γ_mf = variationalAlgorithm(q_γ, name="PMF")
algo_x_mf = variationalAlgorithm([q_x_0; q_x], name="XMF")
algo_F_mf = freeEnergyAlgorithm(name="MF");

"""
Run inference algorithm
"""

# Load algorithm
eval(Meta.parse(algo_γ_mf))
eval(Meta.parse(algo_x_mf))
eval(Meta.parse(algo_F_mf))

# Initialize data
global data = Dict(:y => observed)

# Initial recognition distributions
global marginals_mf = Dict{Symbol, ProbabilityDistribution}()
global marginals_mf[:γ] = vague(Gamma)
for t = 0:T
    global marginals_mf[:x_*t] = vague(GaussianMeanPrecision)
end

# Preallocate
global estimated_states = zeros(T,2,TT)
global estimated_noises = zeros(T,2,TT)

# Run algorithm
global F_mf = Vector{Float64}(undef, TT)
for tt = 1:TT
    stepPMF!(data, marginals_mf)
    stepXMF!(data, marginals_mf)

    for t = 1:T
        estimated_states[t,1,tt] = mean(marginals_mf[:x_*t])
        estimated_states[t,2,tt] = sqrt(var(marginals_mf[:x_*t]))
        estimated_noises[t, 1, tt] = mean(marginals_mf[:γ])
        estimated_noises[t, 2, tt] = sqrt(var(marginals_mf[:γ]))
    end

    global F_mf[tt] = freeEnergyMF(data, marginals_mf)
end

CPUtoc()

"""
Visualize experimental results
"""

# Visualize final estimates over time
plot(hidden[2:end], color="red", label="states", ylims=[-5, 2.5])
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[estimated_states[:,2,end], estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", label="observations")
xlabel!("time (t)")
title!("State estimates, q(x_t)")
savefig(pwd()*"/experiment_infer-processnoise/viz/baseline_state_estimates.png")

# Visualize final noise estimates over time
plot(inv(process_noise)*ones(T,1), color="black", label="true process noise", linewidth=2, ylims=[0., 2.0])
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
savefig(pwd()*"/experiment_infer-processnoise/viz/baseline_noise_estimates.png")
