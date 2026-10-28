module Models

export Model, BasicNN, probabilities, candidates, predict

using Flux

include("metrics.jl")
using .Metrics
export Metrics

abstract type Model end
(::Model)(::AbstractVector) = error("Model must be implemented for each subtype")

probabilities(scores) = scores |> softmax
candidates(probs) = probs |> eachcol .|> argmax
predict(candidates, classes) = candidates |> idx -> classes[idx]

include("basic_nn.jl")

end
