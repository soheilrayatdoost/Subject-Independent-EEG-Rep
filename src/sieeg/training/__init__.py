"""Training: method registry (no TensorFlow import) and the adversarial trainer."""

from .methods import ADVERSARY_KINDS, AdversarySpec, MethodSpec, available_methods, get_method

__all__ = ["ADVERSARY_KINDS", "AdversarySpec", "MethodSpec", "available_methods", "get_method"]
