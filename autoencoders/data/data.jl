module Data

include("datasets.jl")

using .Datasets: length, features, targets, sample, plot_data, DataLoader
export length, features, targets, sample, plot_data, DataLoader

using .Datasets: XorDataset, SinusoidDataset
export XorDataset, SinusoidDataset

end
