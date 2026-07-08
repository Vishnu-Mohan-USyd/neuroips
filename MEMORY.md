# Project Memory

- Response shapes are validation outcomes only. Training objectives may differ
  only by naturalistic task, energy, and precision demands, with minimal
  mechanism differences between sharpen and dampen networks. Never add explicit
  dampen/sharpen shape losses, center/flank objectives, local annulus terms, or
  target curve templates.
- Expected-vs-unexpected energy and shape comparisons are validation outcomes
  only. Never add expected-vs-unexpected contrast losses, or any training loss
  that knows a stimulus is expected or unexpected by construction. Energy
  pressure must be general metabolic or homeostatic pressure that applies under
  all relevant circumstances, not selectively to expected cases.
- Do not use regime-specific circuit or gain parameters that directly make
  feedback suppress more or less, including regime-specific
  `pred_feature_supp_strength`. Sharpen and dampen may share the same
  architecture, initial form, and circuit parameters; learned weights may
  diverge only through training under different objective pressures. Allowed
  regime differences are objective weights such as energy, task, precision, and
  homeostasis weights.
