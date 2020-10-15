import Random: AbstractRNG


import Test: @test, @testset
import Statistics: mean, var


export key
export squares_rng
# Implementation of Widynski, Bernard (2020). 
# "Squares: A Fast Counter-Based RNG". arXiv:2004.06278v2

# key (seed) taken from keys.h (line 2193) in the distribution of Squares
# see https://squaresrng.wixsite.com/rand

# this distribution also includes a generator for these keys - eventually
# this hardcoded key should be replaced with such a generator 
# (one key generates 2^64 random numbers)
key = 0x86d47f132b79acfd

@inline function squares_rng(counter::UInt64, seed::UInt64)::UInt32
    yy = counter * seed
    z = yy + seed
    xx = yy * (yy+1)
    # >> arithmetic rightshift, >>> logical rightshift 
    # (most C Impl.: >> arithm on signed, logical on unsigned)
    # << logical/arithmetic leftshift
    xx = (xx >>> 32) | (xx << 32) 
    xx = xx*xx + z
    xx = (xx >>> 32) | (xx << 32)
    return UInt32((xx*xx + yy) >> 32)
end

@inline function uniform(number::UInt32, seed::UInt64)
    return [Float32(squares_rng(UInt64(ctr), seed))/typemax(UInt32) for ctr in 1:number]
end

struct RVGenerator
    pseudoInverse
    stop::UInt64
end

function Base.iterate(rvg::RVGenerator, state::UInt64=UInt64(0))
    if rvg.stop >= state
        return (rvg.pseudoInverse(Float32(squares_rng(state, key))/typemax(UInt32)), state+1)
    else
        return nothing
    end
end

@testset "Staticial Sanity Check" begin
    rv = uniform(UInt32(10^8), key)
    @test isapprox(mean(rv), 0.5)
    @test isapprox(var(rv), 1/12)
end

# @testset "Iterator" begin
#     rvs = RVGenerator(x->x, 10)
#     for rv = rvs
#         print(rv)
#     end
# end

