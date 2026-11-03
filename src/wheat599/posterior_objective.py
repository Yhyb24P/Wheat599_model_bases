"""Common denoised query targets for synthetic posterior training."""

import torch


@torch.no_grad()
def query_targets(inference, generating, context_y, true_effects, query_z, rbf_cross):
    """Construct both direct and correction targets under the generating model.

    The generating operator must use the synthetic covariance and integrates
    the unknown intercept under a flat prior. If the simulator instead fixes a
    known intercept, this target is not its exact conditional expectation.
    Inference may use a different covariance, for example one estimated from
    context. Both operators must use the same synthetic response scale.
    """
    mean = inference.gaussian_posterior_mean(context_y)
    baseline = inference.predict_given_weights(context_y, mean, query_z, rbf_cross)
    target = generating.predict_given_weights(context_y, true_effects, query_z, rbf_cross)
    return mean, baseline, target, target - baseline


def correction_loss(inference, correction, query_z, rbf_cross, target_correction):
    prediction = inference.prediction_correction(correction, query_z, rbf_cross)
    return (prediction - target_correction).square().mean()


def direct_loss(prediction, target):
    return (prediction - target).square().mean()
