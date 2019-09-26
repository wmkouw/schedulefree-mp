using Distributions

function nu_f0x(x_hat)
    "
    Message from f_0 to edge x.

    Before observation, the message is 1 (i.e. delta(x - x)). After observation,
    it corresponds to the delta of the observation (i.e. delta(x - x_hat)).
    "
    if isnan(x_hat)
        return 1
    else
        return x_hat
    end
end

function nu_f1x(m_mu, sigma)
    "Message from f_1 to edge x."
    return Normal(m_mu, sigma)
end

function nu_f1mu(m_x, sigma)
    "Message from f_1 to edge mu."
    return Normal(m_x, sigma)
end

function nu_f2mu(u, s)
    "Message from f_2 to edge mu."
    return Normal(u, s)
end

function update_qx(m_x, V_x, nu_f0x, nu_f1x; observed=false)
    "
    Update rule for recognition distribution of x.

    Accepts variational parameters and two messages. Computation differs before
    and after observation.
    "
    if observed
        # Recognition distribution centers at observation, with zero variance.
        m_x = nu_f0x
        V_x = 1e-12
    else
        # Recognition distribution becomes message from f_1
        m_x = mean(nu_f1x)
        V_x = var(nu_f1x)
    end
    # Return recognition distributions parameters
    return (m_x, V_x)
end

function update_qmu(m_mu, V_mu, nu_f1mu, nu_f2mu)
    "
    Update rule for recognition distribution of mu.

    Accepts variational parameters and two messages.
    Both messages are normal distributions and their product returns:
        Wx = ->Wx + <-Wx
        Wx*mx = ->Wx*->mx + <-Wx*<-mx
    "
    # Update precisions
    W_mu = inv(var(nu_f1mu)) + inv(var(nu_f2mu))

    # Compute variance
    V_mu = inv(W_mu)

    # Update means
    m_mu = V_mu * (var(nu_f1mu) * mean(nu_f1mu) + var(nu_f2mu) * mean(nu_f2mu))

    return (m_mu, V_mu)
end
