module DataUtils

using Flux
using Plots

include("datasets.jl")

using .Datasets: DataSample

export new_dataloader, plot_data

function new_dataloader(data::DataSample)::Flux.DataLoader
    Flux.DataLoader(
        (data.features, Flux.onehotbatch(data.classes, XOR_CLASSES)),
        batchsize=64, shuffle=true
    )
end

function plot_data(features, z)
    x, y = eachrow(features)
    scatter(x, y, marker_z=z,
        color=get(cgrad(:heat), z),
        xlims=(0, 1),
        ylims=(0, 1))
end

end
