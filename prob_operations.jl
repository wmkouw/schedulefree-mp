"""
Operations on probability distributions.

Wouter Kouw
28-03-2029
"""

export Delta

import Base: *
import Statistics: mean, var
import Distributions: params, pdf, logpdf
# import Distributions: ContinuousUnivariateDistribution

struct Delta <: ContinuousUnivariateDistribution
    """
    Delta distribution.

    This is not a proper distribution, but this structure allows for a general
    interface for observed variables and unobserved variables (i.e. it is
    possible to call mean/var on a fixed variable).
    """

    # Attributes
    value::Float64

    function Delta(value::Float64)
        return new(value);
    end
end

function mean(d::Delta)
    return d.value
end

function std(d::Delta)
    return 0.0
end

function var(d::Delta)
    return 0.0
end

function meanvar(d::Delta)
    return mean(d), var(d)
end

struct Flat <: ContinuousUnivariateDistribution
    """
    Non-informative (flat) prior distribution.

    This is an improper prior that treats each element of its sample space as
    equally likely. It is used as an identity function for multiplication with
    other probability distrbutions. In other words:
        Flat(x) ⋅ Normal(x | 0,1) = Normal(x | 0,1).

    # TODO: Flat can be replaced by Uniform(-Inf, Inf)
    """
    Flat() = new()
end

function pdf(p::Flat, x::Float64)
    "Assuming sample space of Flat is [-Inf, Inf]"
    return 0.0
end

function logpdf(p::Flat, x::Float64)
    return log(pdf(p, x))
end

function mean(d::Flat)
    return 0.0
end

function var(d::Flat)
    return Inf
end

function meanvar(p::Flat)
    return mean(p), var(p)
end

function meanvar(p::Normal)
    "Mean and variance of a Normal distribution."
    return mean(p), var(p)
end

# function pdf(p::Normal, x::Float64)
#     "Probability of sample under Normal distribution."
#     return 1/sqrt(2*π*var(p))*exp(-(x-mean(p))^2/(2*var(p)))
# end
#
# function logpdf(p::Normal, x::Float64)
#     "Probability of sample under Normal distribution."
#     return -log(2*π)/2 - log(std(p)) -(x-mean(p))^2/(2*var(p))
# end

function meanvar(p::Gamma)
    "Mean and variance of a Gamma distribution."
    return mean(p), var(p)
end

function meanvar(graph::MetaGraph, x::Float64)
    "If variable is clamped to a value, then return that value with variance 0."
    return x, 0
end

function meanvar(graph::MetaGraph, x::Symbol)
    "Extract marginal from variable in graph and return mean and variance."
    return meanvar(graph[graph[x, :id], :node].marginal)
end

function *(a::Symbol, b::Integer)
    "Concatenate symbol and integer"
    return Symbol(string(a)*string(b))
end

function *(a::Symbol, b::String)
    "Concatenate symbol and integer"
    return Symbol(string(a)*b)
end

function *(px::Normal{Float64}, qx::Normal{Float64})
    "Multiplication of two normal distributions of the same variable."

    # Extract parameters
    μ_p, σ_p = params(px)
    μ_q, σ_q = params(qx)

    # Compute precisions
    τ_p = inv(σ_p^2)
    τ_q = inv(σ_q^2)

    # Add precisions
    τ = τ_p + τ_q

    # Add precision-weighted means
    τμ = τ_p*μ_p + τ_q*μ_q

    # Compute new variance
    σ2 = inv(τ)

    # Return normal distribution in mean-stddev parameterization
    return Normal(σ2*τμ, sqrt(σ2))
end

function *(px::Gamma{Float64}, qx::Gamma{Float64})
    "Multiplication of two Gamma distributions of the same variable."

    # Extract parameters
    shape_p, scale_p = params(px)
    shape_q, scale_q = params(qx)

    # Add shapes
    new_shape = shape_p + shape_q - 1.

    # Add inverse scales
    new_scale = inv(inv(scale_p) + inv(scale_q))

    # Return gamma distribution in shape-scale parameterization
    return Gamma(new_shape, new_scale)
end

function *(px::Flat, qx::Flat)
    return Flat()
end

function *(px::Union{Gamma{Float64},Normal{Float64}}, qx::Flat)
    "Multiplication of a distribution with a flat distribution."
    return px
end

function *(qx::Flat, px::Union{Gamma{Float64},Normal{Float64}})
    "Multiplication of a distribution with a flat distribution."
    return px
end

function expectation(q::Normal{Float64}, ν::Normal{Float64})
    "Compute the expectation E_q(x) [log ν(x)]"

    # Parameters
    μ_q, σ_q = params(q)
    μ_ν, σ_ν = params(ν)

    return -log(σ_ν) - 1/2*log(2*π) -1/(2*σ_ν^2)*(σ_q^2 + μ_q^2 -2*μ_q*μ_ν + μ_ν^2)
end

function expectation(q::Gamma{Float64}, ν::Gamma{Float64})
    "Compute the expectation E_q(x) [log ν(x)]"

    # Parameters
    α_q, θ_q = params(q)
    α_ν, θ_ν = params(ν)

    return -log(gamma(α_ν)) - α_ν*log(θ_ν) + (α_ν-1)*(digamma(α_q) + log(θ_q)) - (α_q*θ_q)/θ_ν
end

function expectation(q::Union{Gamma{Float64},Normal{Float64}}, ν::Flat)
    "If the message is a flat distribution, the integral E_q(x) [log ν(x)] diverges."
    return Inf
end

function expectation(ν::Flat, q::Union{Gamma{Float64},Normal{Float64}})
    "If the message is a flat distribution, the integral E_q(x) [log ν(x)] diverges."
    return Inf
end

function grad_expectation(q::Normal{Float64}, ν::Normal{Float64})
    "Compute the gradient of the expectation E_q(x) [log ν(x)] w.r.t. q(x)'s params"

    # Parameters
    μ_q, σ_q = params(q)
    μ_ν, σ_ν = params(ν)

    # Partial derivative w.r.t μ_q
    partial_μ = -(μ_q - μ_ν)/(σ_ν^2)

    # Partial derivative w.r.t σ_q
    partial_σ = - σ_q / σ_ν

    return (partial_μ, partial_σ)
end

function grad_expectation(q::Gamma{Float64}, ν::Gamma{Float64})
    "Compute the gradient of the expectation E_q(x) [log ν(x)] w.r.t. q(x)'s params"

    # Parameters
    α_q, θ_q = params(q)
    α_ν, θ_ν = params(ν)

    # Partial derivative w.r.t α_q
    partial_α = (α_ν-1)*polygamma(1, α_q) - θ_q / θ_ν

    # Partial derivative w.r.t σ_q
    partial_θ = (α_ν-1)/θ_q - α_q / θ_ν

    return (partial_α, partial_θ)
end

function grad_expectation(q::Union{Gamma{Float64},Normal{Float64}}, ν::Flat)
    "The gradient of a diverging integral, diverges."
    return (Inf, Inf)
end

function grad_expectation(ν::Flat, q::Union{Gamma{Float64},Normal{Float64}})
    "The gradient of a diverging integral, diverges."
    return (Inf, Inf)
end
