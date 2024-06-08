def bayesian_deepsic_loss_callback(i, bnn, e: float, user: int, layer: int, losses:list[list[list]],
                                   *args, **kwargs):
    losses[user][layer].append(e)


def deepsic_loss_callback(epoch_num, net, loss: float, user_num: int, layer_num: int, losses:list[list[list]],
                          *args, **kwargs):
    losses[user_num][layer_num].append(loss)