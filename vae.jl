using Flux
using MLDatasets
using Statistics
using ProgressMeter
using Plots

function getData()
    data = rand(Float32, 2, 1000)
    classes = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(data)]
    return (
        data,
        classes
    )
end

function plotData(datapoints::Matrix, classes::Vector{Bool})
    x,y = eachrow(datapoints) 
    scatter(x,y, zcolor=classes)
end

function newModel()
    return Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )
end


