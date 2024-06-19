def estimate_cost(predictions, annotation: [str, float, float]) -> float:
    """
    :param predictions: of shape ((prediction, pred_start_frame, pred_end_frame),)
    :param annotation: of shape ((command, start_second, end_second),)
    :return: estimated cost
    """
    command, start, end = annotation
    for prediction in predictions:
        prediction, pred_start_frame, pred_end_frame = prediction
        pred_start = pred_start_frame / 39
        pred_end = pred_end_frame / 39 + 1
        # TP
        if prediction == command and pred_start >= start and pred_end <= end:
            return -1
        # FP
        if prediction != command:
            if pred
    return 0.5