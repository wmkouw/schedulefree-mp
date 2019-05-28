# Script to run a small-scale experiment.
# Wouter Kouw
# 15-05-2019
#
# Consider an FFG with 3 nodes:
# f_0(x) = d(x - x)
# f_1(x, mu) = N(x | mu, sigma⁻¹)
# f_2(mu) = N(mu | u, t⁻¹)
#
# with
# q(mu) = N(mu | m_mu, V_mu)
#
# We want the node f_1 to appropriately react to incoming messages.

using Revise
using Printf
using Distributions
using PyCall
plt = pyimport("matplotlib.pyplot")

# Import energy functions
include("free_energies.jl")
include("update_rules.jl")

# Prefixes for saving files
vizprf = pwd() * "/viz/"

# Parameters
u =-1.0
s = 1.0
sigma = 0.5

# Recognition distributions
q_x(m, v) = Normal(m, v)
q_mu(m, v) = Normal(m, v)

### Before observation, the system passes messages to attain equilibrium.

# Number of iterations to run
N = 10

# Preallocate arrays
Fqx = zeros(1,N*2)
Fqmu = zeros(1,N*2)
DFqx = zeros(1,N*2)
DFqmu = zeros(1,N*2)

m_x = zeros(1,N*2)
V_x = ones(1,N*2)
m_mu = zeros(1,N*2)
V_mu = ones(1,N*2)

for i = 2:N

    # Messages to edge x
    nu0 = nu_f0x(NaN)
    nu1 = nu_f1x(m_mu[i-1], sigma)

    # Update q(x)
    m_x[i], V_x[i] = update_qx(m_x[i-1], V_x[i-1], nu0, nu1, observed=false)

    # Compute free energy of edge x
    Fqx[i] = Fq_x(m_x[i], V_x[i], m_mu[i-1], V_mu[i-1], sigma)
    DFqx[i] = DFq_x(m_x[i], V_x[i], m_x[i-1], V_x[i-1], m_mu[i-1], V_mu[i-1], sigma)

    # Messages to edge mu
    nu2 = nu_f1mu(m_x[i], sigma)
    nu3 = nu_f2mu(u, s)

    # Update q(mu)
    m_mu[i], V_mu[i] = update_qmu(m_mu[i-1], V_mu[i-1], nu2, nu3)

    # Compute free energy of edge mu
    Fqmu[i] = Fq_mu(m_mu[i], V_mu[i], m_x[i], V_x[i], sigma, u, s)
    DFqmu[i] = DFq_mu(m_mu[i], V_mu[i], m_mu[i-1], V_mu[i-1], m_x[i], V_x[i], sigma, u, s)

end

### After observation, the system is disturbed from equilibrium
x_hat = 1.0

for i = N+1:N*2

    # Messages to edge x
    nu0 = nu_f0x(x_hat)
    nu1 = nu_f1x(m_mu[i-1], sigma)

    # Update q(x)
    m_x[i], V_x[i] = update_qx(m_x[i-1], V_x[i-1], nu0, nu1, observed=true)

    # Compute free energy of edge x
    Fqx[i] = Fq_x(m_x[i], V_x[i], m_mu[i-1], V_mu[i-1], sigma)
    DFqx[i] = DFq_x(m_x[i], V_x[i], m_x[i-1], V_x[i-1], m_mu[i-1], V_mu[i-1], sigma)

    # Messages to edge mu
    nu2 = nu_f1mu(m_x[i], sigma)
    nu3 = nu_f2mu(u, s)

    # Update q(mu)
    m_mu[i], V_mu[i] = update_qmu(m_mu[i-1], V_mu[i-1], nu2, nu3)

    # Compute free energy of edge mu
    Fqmu[i] = Fq_mu(m_mu[i], V_mu[i], m_x[i], V_x[i], sigma, u, s)
    DFqmu[i] = DFq_mu(m_mu[i], V_mu[i], m_mu[i-1], V_mu[i-1], m_x[i], V_x[i], sigma, u, s)

end

# Plot free energy minimization
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(2:N*2, Fqx[2:end]', color="red", label="F[q(x)]")
ax.semilogy(2:N*2, Fqmu[2:end]', color="blue", label="F[q(mu)]")
# ax.set_ylim([1e-2, 1e1])
ax.legend()
ax.set_xlabel("iterations")
ax.set_ylabel("Free energy")
fig.savefig(vizprf * "FE_xhat" * string(x_hat) * ".png",
            bbox_inches="tight",
            padding=0.0)

# Plot delta free energy
fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(2:N, DFqx[2:end]', color="red", label="ΔF[q(x)]")
ax.plot(2:N*2, DFqmu[2:end]', color="blue", label="ΔF[q(mu)]")
ax.legend()
ax.set_xlabel("iterations")
ax.set_ylabel("Change in free energy")
fig.savefig(vizprf * "DFE_xhat" * string(x_hat) * ".png",
            bbox_inches="tight",
            padding=0.0)

# Plot variational distributions
fig, ax = plt.subplots(figsize=(6,5))
ax.fill_between(1:N*2, (m_x .- V_x.*(1/2))'[:,1], (m_x .+ V_x.*(1/2))'[:,1], facecolor="red", alpha=0.2)
ax.plot(1:N*2, m_x', color="red", linestyle="dashed", label="m_x")
ax.fill_between(1:N*2, (m_mu .- V_mu.*(1/2))'[:,1], (m_mu .+ V_mu.*(1/2))'[:,1], facecolor="blue", alpha=0.2)
ax.plot(1:N*2, m_mu', color="blue", linestyle="dotted", label="m_μ")
ax.legend()
ax.set_xlabel("iterations")
ax.set_ylabel("Values variational params")
fig.savefig(vizprf * "varparams_xhat" * string(x_hat) * ".png",
            bbox_inches="tight",
            padding=0.0)

plt.close("all")
