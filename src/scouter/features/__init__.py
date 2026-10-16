"""Dictionary learning: find and steer interpretable features (sparse autoencoder).

Anthropic's "Scaling Monosemanticity" trained sparse autoencoders on Claude 3
Sonnet's residual stream and found features like the Golden Gate Bridge one,
then clamped a feature on to steer the model. This subsystem does the same for
any HuggingFace model, at one layer:

  * ``dictionary`` - FeatureDictionary (encode / top_features / as_axis / save / load).
  * ``train``      - train_sae and label_features (top-activating tokens).
  * ``generate``   - the ``scouter-features`` CLI.
"""

from .dictionary import FeatureDictionary, FeatureHit
from .generate import build_feature_dictionary, drop_tokens, load_corpus, main
from .train import FeatureExample, SAEConfig, label_features, train_sae

__all__ = [
    "FeatureDictionary",
    "FeatureHit",
    "FeatureExample",
    "SAEConfig",
    "train_sae",
    "label_features",
    "build_feature_dictionary",
    "drop_tokens",
    "load_corpus",
    "main",
]
