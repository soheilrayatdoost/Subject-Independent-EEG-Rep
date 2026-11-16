"""The method registry of Table 1 (sieeg.training.methods)."""

from __future__ import annotations

import math

import pytest

from sieeg.training.methods import ADVERSARY_KINDS, AdversarySpec, available_methods, get_method


def test_registry_lists_the_six_methods_of_table_1_in_paper_order():
    assert available_methods() == ("cnn", "sinn", "si_eeg", "dann", "sidann", "sida_eeg")


@pytest.mark.parametrize(
    "name, adversaries",
    [
        ("cnn", []),
        ("sinn", [("identity", 0.2)]),
        ("si_eeg", [("group", 0.2)]),  # paper Sec. 6: best lambda 0.2
        ("dann", [("domain", 0.1)]),
        ("sidann", [("domain", 0.1), ("identity", 0.15)]),
        ("sida_eeg", [("domain", 0.1), ("group", 0.15)]),  # paper Sec. 6: lambda_s 0.15, lambda_d 0.1
    ],
)
def test_default_lambdas(name, adversaries):
    assert [(a.kind, a.lam) for a in get_method(name).adversaries] == adversaries


def test_lambda_overrides_apply_per_adversary_kind_and_names_are_normalised():
    spec = get_method("SIDA-EEG", lambda_subject=0.3, lambda_domain=0.05)
    assert spec.name == "sida_eeg"
    assert [(a.kind, a.lam) for a in spec.adversaries] == [("domain", 0.05), ("group", 0.3)]
    assert spec.requires_target_domain and spec.has("group") and not spec.has("identity")
    assert get_method("si_eeg", lambda_subject=0).adversary("group").lam == 0.0  # lambda 0 is a valid ablation


@pytest.mark.parametrize("lam", [math.nan, math.inf, -math.inf, -0.1])
def test_lambda_must_be_finite_and_non_negative(lam):
    with pytest.raises(ValueError, match=r"lambda of the group adversary must be a finite number >= 0"):
        AdversarySpec("group", lam)
    with pytest.raises(ValueError, match=r"lambda of the domain adversary"):
        get_method("dann", lambda_domain=lam)


def test_unknown_kind_and_method_are_rejected():
    with pytest.raises(ValueError, match="Unknown adversary kind"):
        AdversarySpec("speaker", 0.2)
    with pytest.raises(ValueError, match="Unknown method 'adda'"):
        get_method("adda")
    assert set(ADVERSARY_KINDS) == {"group", "identity", "domain"}
