from sklearn.metrics import mean_absolute_error


def evaluate_on_mea(valid_labels, pred_labels):
    mae = mean_absolute_error(valid_labels, pred_labels)
    print(mae)
