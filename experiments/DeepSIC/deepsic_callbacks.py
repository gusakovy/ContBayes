from typing import Generator

def bayesian_deepsic_loss_callback(i, bnn, e: float, user: int, layer: int, losses:list[list[list]],
                                   *args, **kwargs):
    losses[user][layer].append(e)


def deepsic_loss_callback(epoch_num, net, loss: float, user_num: int, layer_num: int, losses:list[list[list]],
                          *args, **kwargs):
    losses[user_num][layer_num].append(loss)


def per_frame_metrics_callback(iteration_num, detector, inputs, outputs, errors: list, confidences: list,
                                test_generator: Generator, *args, **kwargs):
    rx, labels = next(test_generator)
    error, confidence = detector.test_model(rx=rx, labels=labels)
    errors.append(error.item())
    confidences.append(confidence.item())
