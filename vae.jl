using Flux
using Flux.Optimisers
using Flux.Losses
using MLUtils
using ProgressMeter
using Plots

function getData(N=1000)
    data = rand(Float32, 2, N)
    classes = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(data)]
    return (
        data,
        classes
    )
end

function plotData(datapoints::Matrix, classes::Vector{Bool})
    x, y = eachrow(datapoints)
    scatter(x, y, marker_z=classes,
        xlims=(0, 1),
        ylims=(0, 1))
end

function newModel()
    return Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )
end

function probs(model, data)
    model(data) |> softmax
end

function predict(probs)
    probs |> eachcol .|> argmax
end

function update!(model, opt, x, y)
    loss, grads = Flux.withgradient(model) do m
        Losses.logitcrossentropy(m(x), y)
    end
    Flux.update!(opt, model, grads[1])
    loss
end

function train!(model, dataloader, epochs)
    opt_state = Flux.setup(Optimisers.Adam(0.01), model)
    losses = []
    @showprogress for epoch in 1:epochs
        for (x,y) in dataloader
            loss = update!(model, opt_state, x, y)
            push!(losses, loss)
        end
    end

    opt_state
end
