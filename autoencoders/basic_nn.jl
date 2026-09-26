using Flux
using Flux.Optimisers
using Flux.Losses
using MLUtils
using ProgressMeter

# TODO: Add some parameters as args so this is more flexible
function new_model()
    Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )
end
probs(model, data) = model(data) |> softmax
predict(probs) = probs |> eachcol .|> argmax .|> x -> x - 1

struct ModelTrainer
    model       # TODO: What type should this be
    opt_state   # TODO: What type should this be
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

function train!(trainer, dataloader, epochs)
    losses = []
    @showprogress for _ in 1:epochs
        for (x, y) in dataloader
            loss = update!(trainer, x, y)
            push!(losses, loss)
        end
    end
end
