
module OrderStatistics

import CUDA
import CUDA: @cuda

# Implementation of Widynski, Bernard (2020). 
# "Squares: A Fast Counter-Based RNG". arXiv:2004.06278v2

# key (seed) taken from keys.h (line 2193) in the distribution of Squares
# see https://squaresrng.wixsite.com/rand

# this distribution also includes a generator for these keys - eventually
# this hardcoded key should be replaced with such a generator 
# (one key generates 2^64 random numbers)
@inline function squares_rng(counter::UInt64)::UInt32
    seed::UInt64 = 0x86d47f132b79acfd
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

function gpu_parallel!(results, sampleSize, pseudoInverse)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for thread in index:stride:length(results)
        result = 0
        for idx in 1:sampleSize
            result = max(result, pseudoInverse(Float32(squares_rng(UInt64(idx)))/typemax(UInt32)))
        end
        results[thread] = result 
    end
end

function sample_extreme_values(sampleSize, superSampleSize, pseudoInverse)::Array{Float32,1}
    numblocks = ceil(Int, superSampleSize/256)
    gpu_res = CUDA.CuArray{Float32}(undef, superSampleSize)
    cpu_res = Array{Float32}(undef, superSampleSize)
    @cuda threads=256 blocks=numblocks gpu_parallel!(gpu_res, sampleSize, pseudoInverse)
    copyto!(cpu_res, gpu_res)
    return cpu_res
end

function largest(generator)
    largest = Float32(0)
    for rv = generator
        largest = max(largest, rv)
    end
    return largest
end

end # module
