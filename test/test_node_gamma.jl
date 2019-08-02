using Test

include("../nodes/node_gamma.jl")

@testset "constructor" begin

    # Edge variable
    x = Gamma(1,1)

    # Initialize node
    h = NodeGamma(:x, 1.,1., "h")

    @test h.node_id == "h"
    @test h.params == 1.0
    @test h.rate == 1.0
    @test h.outcomes_edge_id == :x

    #TODO: test invalid inputs
end

@testset "energy" begin

    # Edge variable
    x = Gamma(1.,1.)

    # Initialize node
    h = NodeGamma(:x, 1.,1., "h")

    @test energy(h) == -1.0
end
