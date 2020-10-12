
module OrderStatistics

include("rngSquares.jl")
import CUDA

function gpu_parallel!(results, callable, sampleSize)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * gridDim().x
    for thread  = index:stride:length(results)
        results[thread] = callable(thread)
    end
end

function sample_extreme_values(sampleSize, superSampleSize, pseudoInverse)::T where T
    numblocks = ceil(Int, sampleSize/256)
    gpu_res = CUDA.CuArray{T}(undef, superSampleSize)
    cpu_res = Array{T}(undef, superSampleSize)
    @cuda threads=256 blocks=numblocks gpu_parallel!(gpu_res, pseudoInverse, sampleSize)
    copyto!(cpu_res, gpu_res)
    return cpu_res
end

end # module
