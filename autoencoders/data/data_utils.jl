module DataUtils

using Flux

include("datasets.jl")

using .Datasets: DataSample

export new_dataloader

function new_dataloader(data::DataSample)::Flux.DataLoader
    Flux.DataLoader(
        (data.features, Flux.onehotbatch(data.classes, XOR_CLASSES)),
        batchsize=64, shuffle=true
    )
end

end
