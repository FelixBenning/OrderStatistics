
module OrderStatistics

import CUDA
import CUDA: @cuda

include("rngSquares.jl")

function gpu_parallel!(results, sampleSize, pseudoInverse, seed::UInt64, callable, rng::RNG)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for thread in index:stride:length(results)
        rvg = rng(pseudoInverse, UInt64(thread), UInt64(length(results)), UInt64(sampleSize), seed)
        results[thread] = callable(rvg)
    end
end

function sample_extreme_values(sampleSize, superSampleSize, pseudoInverse; callable=largest, seed=get_seed(), rng::RNG=unifWithPseudoInvGen)::Array{Float32,1}
    if typeof(pseudoInverse(0.5)) <: Tuple
        # monte carlo pseudoInverse (can fail -> Tuple{rv, accepted::Bool})
        fixedPseudoInverse = pseudoInverse
    else
        # normal pseudoInverse (can not fail -> always return accepted=true)
        fixedPseudoInverse = (x -> (pseudoInverse(x), true))
    end
    numblocks = ceil(Int, superSampleSize/256)
    gpu_res = CUDA.CuArray{Float32}(undef, superSampleSize)
    cpu_res = Array{Float32}(undef, superSampleSize)
    @cuda threads=256 blocks=numblocks gpu_parallel!(gpu_res, sampleSize, fixedPseudoInverse, seed, callable, rng)
    copyto!(cpu_res, gpu_res)
    return cpu_res
end

function largest(generator::unifWithPseudoInvGen{T}) where T
    largest = Float32(0)
    for rv = generator
        largest = max(largest, rv)
    end
    return largest
end

function diff_largest_second(generator::unifWithPseudoInvGen{T}) where T
    largest = Float32(0)
    second = Float32(0)
    for rv in generator
        if rv > second
            second = min(largest, rv)
            largest = max(largest, rv)
        end
    end
    return largest - second
end

end # module
