"""TabM mechanism core (WP-B). Math structure + training method, no experiment orchestration.

Contract:
- backbone: GenomicTabM, EnvironmentLinearBatchEnsemble, SharedTabM, IndependentTabM,
  EnvironmentTabM, InducedFTTransformer, PreLNCrossAttention, build_induced_model.
- quantile: quantile_loss, build_model, update_epoch, train (variant dispatcher), mixture_scores,
  QueryQuantile, ConditionalQuantile, entmax15.
- shape: ShapeTransform.
- interactions: InteractionBackbone, LayerInteractionBackbone, make_optimizers,
  MemberInteractionBackbone, ConnectionBasis, basis_loss.

run()/_fit_worker/_init_fit_worker live in genopred.experiments.wheat599.quantile_tabm_dev (WP-C).
"""