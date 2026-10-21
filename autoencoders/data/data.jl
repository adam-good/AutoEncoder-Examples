module Data
include("data_utils.jl")
include("datasets.jl")

using .DataUtils: new_dataloader
export new_dataloader

using .Datasets: length, features, targets, sample, plot_data
export length, features, targets, sample, plot_data

using .Datasets: XorDataset, SinusoidDataset
export XorDataset, SinusoidDataset

end
