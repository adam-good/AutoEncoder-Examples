
using Plots

xor2d(x::Number, y::Number)::Bool = xor(x > 0.5, y > 0.5)

struct DataSample
    N::Integer
    features::Matrix{Float32}
    classes::Vector{Bool}
end
length(x::DataSample) = x.N
features(x::DataSample) = x.features
classes(x::DataSample) = x.classes
ith_feature(i::Integer, x::DataSample) = eachcol(x.features)[i]
ith_class(i::Integer, x::DataSample) = x.classes[i]

function new_sample(N::Integer=1000)::DataSample
    data = rand(Float32, 2, N)
    DataSample(
        N,
        data,
        [xor2d(col[1], col[2]) for col in eachcol(data)]
    )
end

function dataloader(data::DataSample)
    Flux.DataLoader(
        (data.features, Flux.onehotbatch(data.classes, [true, false])),
        batchsize=64, shuffle=true
    )
end

function plot_data(features, classes)
    x, y = eachrow(features)
    scatter(x, y, group=classes,
        xlims=(0, 1),
        ylims=(0, 1))
end


