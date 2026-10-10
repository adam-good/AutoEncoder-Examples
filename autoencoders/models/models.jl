module Models

using Flux

include("basic_nn.jl")
include("metrics.jl")

using .Metrics

abstract type Model <: Flux.Chain end
new()::Model = error("new()::Model must be implemented for each subtype")

probabilities(model::Model, data) = model(data) |> softmax
predict(probabilities, classes) = begin
    probabilities |>
    eachcol .|>
    argmax .|>
    idx -> classes[idx]
end

end
