module ModelTraining

export ModelTrainer, train!

using Flux
using Flux.Optimisers
using Flux.Losses
using ProgressMeter

using ..Models

const DEFAULT_OPT = Optimisers.Adam(0.01)
struct ModelTrainer
    model::Flux.Chain   # TODO: Generalize
    opt_state           # TODO: Figure out a type for this
end
ModelTrainer(model::Model, opt::Optimisers.AbstractRule=DEFAULT_OPT) = ModelTrainer(
    model.inner,
    Flux.setup(opt, model.inner) # TODO: This won't work if we expand past Flux
)
model(trainer::ModelTrainer)::Flux.Chain = trainer.model

function update!(trainer, x, y)
    loss, grads = Flux.withgradient(trainer.model) do m
        Losses.logitcrossentropy(m(x), y)
    end
    Flux.update!(trainer.opt_state, trainer.model, grads[1])
    loss
end

function train!(trainer, dataloader, epochs; callbacks=Dict())
    losses = []
    cbs = []
    @showprogress for _ in 1:epochs
        counter = 0
        batch_loss = 0
        for (x,y) in dataloader
            loss = update!(trainer, x, y)
            batch_loss += loss
            counter += 1
        end
        batch_loss /= counter
        cb = Dict(key => fn() for (key,fn) in callbacks)
        push!(losses, batch_loss)
        push!(cbs, cb)
    end
    losses, cbs
end

function validate(model, x, y; metrics::Dict{Symbol,Function}=MLMetrics.DEFAULT_METRICS)
    y_pred = model(x) |> softmax |> predict
    conf_mat = MLMetrics.confusion_matrix(y_pred, y)

    Dict(metric => fn(conf_mat) for (metric, fn) = metrics)

end

end
