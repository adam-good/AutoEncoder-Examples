module Data
include("data_utils.jl")
include("datasets.jl")

using .DataUtils: new_dataloader, plot_data
export new_dataloader, plot_data

using .Datasets: length, features, classes, sample
export length, features, classes, sample

using .Datasets: XorDataset, SinusoidDataset
export XorDataset, SinusoidDataset

end
