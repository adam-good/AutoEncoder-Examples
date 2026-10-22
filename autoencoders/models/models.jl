module Models

export Model, BasicNN

using Flux

include("metrics.jl")
using .Metrics

abstract type Model end
new()::Model = error("new()::Model must be implemented for each subtype")
(::Model)(::AbstractVector) = error("Model must be implemented for each subtype")

probabilities(model::Model, data) = model(data) |> softmax
predict(probabilities, classes) = begin
    probabilities |>
        eachcol .|>
        argmax .|>
        idx -> classes[idx]
end

include("basic_nn.jl")

end
