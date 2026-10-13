module Data
include("data_utils.jl")
include("datasets.jl")

using .DataUtils: new_dataloader, plot_data
export new_dataloader, plot_data

using .Datasets: length, features, classes, sample, Xor
export length, features, classes, sample, Xor

using .Datasets.Xor: XorDataset
export XorDataset

end
