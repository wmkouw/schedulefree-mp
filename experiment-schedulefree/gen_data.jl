function gen_data_randomwalk(transition_precision,
                             emission_precision,
                             m0, W0;
                             time_horizon=100)
    "Draw a time-series according to a generative model"

    # Dimensionality
    d = size(transition_precision, 1)

    # Preallocate
    y = zeros(time_horizon, d)
    x = zeros(time_horizon+1, d)

    # Initialize state
    x[1, :] = randn(d).*sqrt(W0) .+ m0

    for t = 1:T

        # Evolve state
        x[t+1, :] = randn(d)./sqrt(transition_precision) + x[t, :]

        # Observe
        y[t, :] = randn(d)./sqrt(emission_precision) + x[t+1, :]

    end
    return y, x
end

function gen_data_LGDS(transition_matrix,
                       emission_matrix,
                       transition_precision,
                       emission_precision,
                       m0, W0;
                       time_horizon=100)
    "Draw a time-series according to a generative model"

    # Dimensionality
    d = size(transition_matrix, 1)

    # Preallocate
    y = zeros(T, d)
    x = zeros(T+1, d)

    # Initialize state
    x[1, :] = randn(d)./sqrt(W0) .+ m0

    for t = 1:time_horizon

        # Evolve state
        x[t+1, :] = randn(d)./sqrt(transition_precision) + transition_matrix*x[t, :]

        # Observe
        y[t, :] = randn(d)./sqrt(emission_precision) + emission_matrix*x[t+1, :]

    end
    return y, x
end
