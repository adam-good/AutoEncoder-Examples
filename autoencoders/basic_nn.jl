using Flux
using Flux.Optimisers
using Flux.Losses
using MLUtils
using ProgressMeter
using Plots

xor2d(x::Number, y::Number)::Bool = xor(x > 0.5, y > 0.5)

struct DataObservation
    x::Float32
    y::Float32
    class::Bool
end
struct DataSample
    N::Integer
    features::Matrix{Float32}
    classes::Vector{Bool}
end
length(x::DataSample) = x.N
features(x::DataSample) = x.features
classes(x::DataSample) = x.classes
# TODO: These should error on overflow
ith_feature(i::Integer, x::DataSample) = eachcol(x.features)[i] 
ith_class(i::Integer, x::DataSample) = x.classes[i] 
ith_observation(i::Integer, x::DataSample) = begin 
    x,y = ith_feature(i,x)
    DataObservation(x,y,ith_class(i,x))
end

function get_data(N::Integer=1000)::DataSample
    data = rand(Float32, 2, N)
    DataSample(
        N,
        data, 
        [xor2d(col[1], col[2]) for col in eachcol(data)]
    )
end

function plot_data(sample::DataSample)
    x, y = eachrow(features(sample))
    scatter(x, y, marker_z=classes(sample),
        xlims=(0, 1),
        ylims=(0, 1))
end

# TODO: Add some parameters as args so this is more flexible
function new_model()
    Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )
end
probs(model, data) = model(data) |> softmax
predict(probs) = probs |> eachcol .|> argmax

struct ModelTrainer
    model::Flux.Chain
    opt_state::Optimisers
end
function new_trainer(model, opt)
    ModelTrainer(model, Flux.setup(opt, model))
end

function update!(trainer, x, y)
    loss, grads = Flux.withgradient(trainer.model) do m
        Losses.logitcrossentropy(m(x), y)
    end
    Flux.update!(trainer.opt_state, trainer.model, grads[1])
    loss
end

function train!(model, dataloader, epochs)
    opt_state = Flux.setup(Optimisers.Adam(0.01), model)
    losses = []
    @showprogress for epoch in 1:epochs
        for (x, y) in dataloader
            loss = update!(model, opt_state, x, y)
            push!(losses, loss)
        end
    end

    opt_state
end
