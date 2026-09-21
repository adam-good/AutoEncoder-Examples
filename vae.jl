using Base: dataids


using Flux
using MLDatasets
using Statistics
using ProgressMeter

function getData()
    data = rand(Float32, 2, 1000)
    truth = [xor(col[1] > 0.5, col[2] > 0.5 for col in eachcol(data))]
    return (
        data,
        truth
    )
end

function newModel()
    return Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )
end
