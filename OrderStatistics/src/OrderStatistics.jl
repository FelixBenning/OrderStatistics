
module OrderStatistics

import CUDA
import CUDA: @cuda

include("rngSquares.jl")

function gpu_parallel!(results, sampleSize, pseudoInverse)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for thread in index:stride:length(results)
        rvg = RVGenerator(pseudoInverse, UInt64(sampleSize))
        results[thread] = largest(rvg)
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

function largest(generator::RVGenerator{T}) where T
    largest = Float32(0)
    for rv = generator
        largest = max(largest, rv)
    end
    return largest
end

end # module
