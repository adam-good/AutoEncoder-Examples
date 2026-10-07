module MLMetrics

# TODO: generalize
struct ConfusionMatrix
    true_pos::Int
    true_neg::Int
    false_pos::Int
    false_neg::Int
end

true_positive(y_pred, y_true) = sum(y_pred .== true .&& y_true .== true)
true_negative(y_pred, y_true) = sum(y_pred .== false .&& y_true .== false)
false_positive(y_pred, y_true) = sum(y_pred .== true .&& y_true .== false)
false_negative(y_pred, y_true) = sum(y_pred .== false .&& y_true .== true)

confusion_matrix(y_pred, y_true) = ConfusionMatrix(
    true_positive(y_pred, y_true),
    true_negative(y_pred, y_true),
    false_positive(y_pred, y_true),
    false_negative(y_pred, y_true)
)

accuracy(conf_mat::ConfusionMatrix) = accuracy(
    conf_mat.true_pos,
    conf_mat.true_neg,
    conf_mat.false_pos,
    conf_mat.false_neg
)
accuracy(tp, tn, fp, fn) = (tp + tn) / (tp + tn + fp + fn)

precision(conf_mat::ConfusionMatrix) = precision(
    conf_mat.true_pos,
    conf_mat.false_pos,
)
precision(tp, fp) = tp / (tp + fp)

recall(conf_mat::ConfusionMatrix) = recall(
    conf_mat.true_pos,
    conf_mat.false_neg
)
recall(tp, fn) = tp / (tp + fn)

f1(conf_mat::ConfusionMatrix) = f1(
    conf_mat.true_pos,
    conf_mat.false_pos,
    conf_mat.false_neg
)
f1(tp, fp, fn) = begin
    prc = precision(tp, fp)
    rcl = recall(tp, fn)
    2 * prc * rcl / (prc + rcl)
end

const DEFAULT_METRICS = Dict(
    :Accuracy => accuracy,
    :Precision => precision,
    :Recall => recall,
    :F1 => f1
)

end
