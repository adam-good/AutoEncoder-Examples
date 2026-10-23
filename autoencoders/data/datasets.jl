module Datasets

export length, features, targets, sample, plot_data, DataLoader

struct DatasetMetadata
    Name::String
    feature_types::Vector{Type}
    target_types::Vector{Type}
end

abstract type DataSample end
length(::DataSample) = error("DataSample Subtypes Must Implement length()")
features(::DataSample) = error("DataSample Subtypes Must Implement features()")
targets(::DataSample) = error("DataSample Subtypes Must Implement targets()")
plot_data(::DataSample) = error("DataSample Subtypes Must Implement plot_data")

abstract type Dataset end
function sample end
sample(::Dataset, ::Any)::DataSample = error("Dataset Subtypes Must Implement sample()")

include("datasets/xor.jl")
include("datasets/sinusoid.jl")

end
