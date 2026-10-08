module ModelUtils 
    using Flux

    export probabilities, predict

    probabilities(model::Flux.Chain, data) = model(data) |> softmax
    predict(probabilities) = probabilities |> eachcol .|> argmax 
end
