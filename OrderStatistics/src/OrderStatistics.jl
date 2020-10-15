
module OrderStatistics

include("rngSquares.jl")
import CUDA
import CUDA: @cuda

function gpu_parallel!(results, pseudoinverse, sampleSize)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    for thread  = index:stride:length(results)
        generator = RVGenerator(pseudoinverse, sampleSize)
        results[thread] = largest(generator)
    end
end

function sample_extreme_values(sampleSize, superSampleSize, pseudoInverse)::Float32
    numblocks = ceil(Int, superSampleSize/256)
    gpu_res = CUDA.CuArray{Float32}(undef, superSampleSize)
    cpu_res = Array{Float32}(undef, superSampleSize)
    @cuda threads=256 blocks=numblocks gpu_parallel!(gpu_res, pseudoInverse, sampleSize)
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
