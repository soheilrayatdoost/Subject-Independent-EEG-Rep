"""The six methods of Table 1, expressed as an emotion classifier plus adversaries.

Every method shares the same encoder and emotion classifier; they differ only in
which gradient-reversal adversaries are attached to the 128-d representation:

* ``group``    - binary subject discriminator: in-group (training subjects, label 0)
                 vs out-group (test subjects, label 1). This is the paper's contribution.
                 Across databases the untested target subjects are in-group too, as the
                 subject term of Eq. 3 covers source and target samples (D14).
* ``identity`` - multi-class subject-identity discriminator (SINN / SIDANN, Yin et al. 2020);
                 pool samples are weighted so that every subject counts equally (D15).
* ``domain``   - binary database discriminator: source (0) vs target (1) (DANN, Ganin et al. 2016).
"""

from __future__ import annotations

from dataclasses import dataclass

ADVERSARY_KINDS = ("group", "identity", "domain")


@dataclass(frozen=True)
class AdversarySpec:
    kind: str  # one of ADVERSARY_KINDS
    lam: float  # gradient-reversal coefficient (lambda in Eq. 1 / lambda_s, lambda_d in Eq. 3)

    def __post_init__(self) -> None:
        if self.kind not in ADVERSARY_KINDS:
            raise ValueError(f"Unknown adversary kind {self.kind!r}; expected one of {ADVERSARY_KINDS}")
        if self.lam < 0:
            raise ValueError(f"lambda must be >= 0, got {self.lam}")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str
    adversaries: tuple[AdversarySpec, ...] = ()

    def has(self, kind: str) -> bool:
        return any(a.kind == kind for a in self.adversaries)

    def adversary(self, kind: str) -> AdversarySpec | None:
        return next((a for a in self.adversaries if a.kind == kind), None)

    @property
    def requires_target_domain(self) -> bool:
        """Needs unlabeled data from a second database (only valid cross-database)."""
        return self.has("domain")


_METHODS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cnn": ("EEG CNN [7]: encoder + emotion classifier, no adversary", ()),
    "sinn": ("SINN [8]: multi-class subject-identity adversary", ("identity",)),
    "si_eeg": ("SI-EEG (Eq. 1): in-group vs out-group subject adversary", ("group",)),
    "dann": ("DANN [9]: source vs target database adversary", ("domain",)),
    "sidann": ("SIDANN [8]: database adversary + subject-identity adversary", ("domain", "identity")),
    "sida_eeg": ("SIDA-EEG (Eq. 3): database adversary + in-group vs out-group subject adversary", ("domain", "group")),
}

#: Gradient-reversal weights. SI-EEG (0.2) and SIDA-EEG (0.15 subject / 0.1 domain)
#: are the best values reported in the paper (Sec. 6); the baselines reuse them so
#: that every method is compared at the same adversarial strength.
DEFAULT_LAMBDAS: dict[str, dict[str, float]] = {
    "cnn": {},
    "sinn": {"subject": 0.2},
    "si_eeg": {"subject": 0.2},
    "dann": {"domain": 0.1},
    "sidann": {"subject": 0.15, "domain": 0.1},
    "sida_eeg": {"subject": 0.15, "domain": 0.1},
}


def available_methods() -> tuple[str, ...]:
    return tuple(_METHODS)


def get_method(name: str, *, lambda_subject: float | None = None, lambda_domain: float | None = None) -> MethodSpec:
    """Build a :class:`MethodSpec`; ``None`` lambdas fall back to :data:`DEFAULT_LAMBDAS`.

    ``lambda_subject`` applies to ``group`` and ``identity`` adversaries,
    ``lambda_domain`` to the ``domain`` adversary.
    """
    key = name.lower().replace("-", "_")
    if key not in _METHODS:
        raise ValueError(f"Unknown method {name!r}; choose from {available_methods()}")
    description, kinds = _METHODS[key]
    defaults = DEFAULT_LAMBDAS[key]
    adversaries = []
    for kind in kinds:
        if kind == "domain":
            lam = lambda_domain if lambda_domain is not None else defaults["domain"]
        else:
            lam = lambda_subject if lambda_subject is not None else defaults["subject"]
        adversaries.append(AdversarySpec(kind=kind, lam=float(lam)))
    return MethodSpec(name=key, description=description, adversaries=tuple(adversaries))
