"""Subject-invariant EEG representation learning for emotion recognition.

TensorFlow 2 / Keras 3 re-implementation of SI-EEG and SIDA-EEG
(Rayatdoost, Yin, Rudrauf, Soleymani, ICASSP 2021) and the baselines compared
in the paper (EEG CNN, SINN, DANN, SIDANN).

Heavy dependencies (TensorFlow, MNE) are imported lazily by the modules that
need them, so ``import sieeg`` stays cheap.
"""

__version__ = "0.1.0"
