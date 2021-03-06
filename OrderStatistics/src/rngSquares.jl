import Random: AbstractRNG


import Test: @test, @testset
import Statistics: mean, var


export key
export squares_rng
# Implementation of Widynski, Bernard (2020). 
# "Squares: A Fast Counter-Based RNG". arXiv:2004.06278v2

# key (seed) taken from keys.h (line 2193) in the distribution of Squares
# see https://squaresrng.wixsite.com/rand

@inline function get_seed()::UInt64
    # this distribution also includes a generator for these keys - eventually
    # this hardcoded key should be replaced with such a generator 
    # (one key generates 2^64 random numbers)
    return 0x86d47f132b79acfd
end

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

abstract type RNG end

struct unifWithPseudoInvGen{T} <: RNG
    pseudoInverse::T
    start::UInt64
    stride::UInt64
    stop::UInt64
    seed::UInt64
end

function Base.iterate(rvg::unifWithPseudoInvGen, state::Tuple{UInt64,UInt64}=(UInt64(0),UInt64(0)))
    if rvg.stop >= state[1]
        attempt = state[2]
        while true
            rv, accepted = rvg.pseudoInverse(Float32(squares_rng(rvg.start + rvg.stride * attempt, rvg.seed))/typemax(UInt32))
            attempt += 1 # monte carlo attempts
            if accepted
                return rv, (state[1]+1,attempt)
            end
        end
    else
        return nothing
    end
end

struct bufferedNormWithTransformation{T} <: RNG
    pseudoInverse::T
    start::UInt64
    stride::UInt64
    stop::UInt64
    seed::UInt64
end

function box_muller(x)
    scalar = sqrt(-2*log(x[1]))
    return (scalar*cos(2*pi*x[2]), scalar*sin(2*pi*x[2]))
end

function Base.iterate(rng::bufferedNormWithTransformation, state=(UInt64(0), Float32(0.0)))
    if rng.stop >= state[1]
        if state % 2 # buffer full
            return state[2]
        else # fill buffer and return first elements
            buffer = [
                Float32(squares_rng(
                    rng.start + rng.stride * (state[1] + UInt64(idx)), 
                    rng.seed)
                )/typemax(UInt32) 
                for idx in 0:1
            ]

            rv, buffer = rng.pseudoInverse.(box_muller(buffer))
            return rv, [state+1, buffer]
        end
    else
        return nothing
    end
end

@testset "Staticial Sanity Check" begin
    rv = uniform(UInt32(10^8), get_seed())
    @test isapprox(mean(rv), 0.5)
    @test isapprox(var(rv), 1/12)
end

# @testset "Iterator" begin
#     rvs = RVGenerator(x->x, 10)
#     for rv = rvs
#         print(rv)
#     end
# end

