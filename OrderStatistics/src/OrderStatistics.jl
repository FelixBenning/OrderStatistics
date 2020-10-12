
module OrderStatistics

include("rngSquares.jl")
import Test: @test
import rngSquares

function helloWorld()
    rngSquares.squares_rng(0, rngSquares.key)

end

end # module
