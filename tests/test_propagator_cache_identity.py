"""Native disk-table identities and conservative callback admission."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.workflow._propagator_identity import (
    native_closed_form_identity,
    native_propagator_identity,
)
from sft_wick.workflow import cache as disk_cache
from sft_wick.workflow import specs
from sft_wick.workflow import closed_forms


def _system(gamma=None, *, kappa2=None, sigma2=None, explicit_R=None):
    if gamma is None:
        gamma = [1.0]
    if kappa2 is None:
        kappa2 = sw.SeparableTranslation(
            sw.ExponentialTemporal(1.0, 1.0), sw.ExponentialSpatial(1.0)
        )
    return sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma),
        noise=sw.GaussianNoise(kappa2, sigma2),
        explicit_R=explicit_R,
    )


def _props(system, cache_path):
    return system.propagators(
        t_max=1.0,
        n_grid_t=4,
        c_closed_form=None,
        c_method="gauss_legendre",
        c_n_gauss=2,
        cache_path=cache_path,
    )


@pytest.mark.parametrize(
    "kernel",
    [
        sw.SeparableTranslation(
            sw.ExponentialTemporal(1.0, 1.0), sw.ExponentialSpatial(1.0)
        ),
        sw.SeparableTranslation(sw.GaussianTemporal(1.0, 1.0), sw.GaussianSpatial(1.0)),
        sw.SeparableRotation(
            sw.ExponentialTemporal(1.0, 1.0), sw.LegendreAngular([1.0, 0.2])
        ),
        sw.SeparableRotation(
            sw.GaussianTemporal(1.0, 1.0), sw.LegendreAngular(np.array([1.0, 0.2]))
        ),
    ],
)
@pytest.mark.parametrize(
    "impulse", [None, sw.ConstantImpulse(0.1), sw.ConstantImpulse(np.eye(1))]
)
def test_native_family_has_copy_stable_identity(kernel, impulse):
    first = _system(np.array([1.0]), kappa2=kernel, sigma2=impulse)
    copied = replace(first, linear=sw.DiagonalA(np.array([1.0])))
    assert native_propagator_identity(first) is not None
    assert native_propagator_identity(first)[1] == "3"
    assert native_propagator_identity(first) == native_propagator_identity(copied)


def test_internal_array_entry_cannot_hide_behind_truncated_repr():
    gamma = np.ones(64)
    changed = gamma.copy()
    changed[31] = 2.0
    left, right = _system(gamma), _system(changed)
    with np.printoptions(threshold=2):
        assert repr(left.linear) == repr(right.linear)
        assert native_propagator_identity(left) != native_propagator_identity(right)


@pytest.mark.parametrize("gamma", [np.array([1.0]), [1.0]])
def test_mutated_numeric_storage_changes_identity(gamma):
    system = _system(gamma)
    original = native_propagator_identity(system)
    gamma[0] = np.nextafter(1.0, 2.0)
    assert native_propagator_identity(system) != original


def test_complex_array_complete_phase_is_in_identity():
    first = _system(np.array([1.0 + 2.0j], dtype=np.complex128))
    second = _system(np.array([1.0 + 3.0j], dtype=np.complex128))
    assert native_propagator_identity(first) != native_propagator_identity(second)


@pytest.mark.parametrize(
    "left,right",
    [
        (np.int64, np.longlong),
        (np.float64, np.longdouble),
        (np.complex128, np.clongdouble),
    ],
)
def test_native_dtype_number_distinguishes_equal_dtype_aliases(left, right):
    if np.dtype(left).num == np.dtype(right).num:
        pytest.skip("platform exposes the same dtype number")
    assert native_propagator_identity(
        _system(np.array([1], dtype=left))
    ) != native_propagator_identity(_system(np.array([1], dtype=right)))


def test_symbolic_field_name_and_interactions_do_not_determine_tables():
    original = _system()
    with_vertices = replace(
        original,
        field=sw.FieldSpec("density", 1),
        vertices=(sw.LocalVertex("F", coupling=np.ones((1, 1, 1))),),
        nonlocal_vertices=(sw.NonLocalVertex("K", order=3, coupling=0.2),),
    )
    assert native_propagator_identity(with_vertices) == native_propagator_identity(
        original
    )


@pytest.mark.parametrize("changed", ["linear", "noise", "t_min", "n_components"])
def test_each_physical_input_changes_identity(changed):
    original = _system()
    if changed == "linear":
        altered = replace(original, linear=sw.DiagonalA([2.0]))
    elif changed == "noise":
        altered = replace(
            original,
            noise=sw.GaussianNoise(
                sw.SeparableTranslation(
                    sw.ExponentialTemporal(2.0, 1.0), sw.ExponentialSpatial(1.0)
                )
            ),
        )
    elif changed == "t_min":
        altered = replace(original, t_min=-1.0)
    else:
        altered = replace(original, field=sw.FieldSpec("phi", 2))
    assert native_propagator_identity(altered) != native_propagator_identity(original)


def test_static_native_effective_linear_cache_is_allowed():
    system = _system()
    expected = native_propagator_identity(system)
    assert system._effective_linear is system.linear
    assert native_propagator_identity(system) == expected
    object.__setattr__(system, "_effective_linear", sw.DiagonalA([2.0]))
    assert native_propagator_identity(system) is None


class _Opaque:
    def __repr__(self):
        raise AssertionError("identity must not call user repr")

    def __eq__(self, _other):
        raise AssertionError("identity must not call user eq")

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("identity must not call user array conversion")


class _OpaqueCallable(_Opaque):
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("identity must not invoke callbacks")


def _opaque_array():
    # Construct the fixture without invoking _Opaque.__array__ itself.
    array = np.empty(1, dtype=object)
    array[0] = _Opaque()
    return array


@pytest.mark.parametrize(
    "system",
    [
        _system(_Opaque()),
        _system(_OpaqueCallable()),
        _system(explicit_R=_OpaqueCallable()),
        _system(kappa2=sw.GeneralKappa2(_OpaqueCallable())),
        _system(
            kappa2=sw.SeparableTranslation(
                sw.CustomKernel(_OpaqueCallable()), sw.ExponentialSpatial(1.0)
            )
        ),
        _system(sigma2=sw.CustomImpulse(_OpaqueCallable())),
        _system(_opaque_array()),
    ],
)
def test_unknown_protocols_bypass_without_user_operations(system):
    assert native_propagator_identity(system) is None


def test_cycles_and_array_subclasses_bypass():
    cyclic = []
    cyclic.append(cyclic)
    assert native_propagator_identity(_system(cyclic)) is None

    class _Array(np.ndarray):
        pass

    assert native_propagator_identity(_system(np.array([1.0]).view(_Array))) is None


def test_custom_metaclass_is_not_hashed_or_compared():
    class _Meta(type):
        def __hash__(cls):
            raise AssertionError("admission must not hash a user class")

        def __eq__(cls, _other):
            raise AssertionError("admission must not compare a user class")

    class _Value(metaclass=_Meta):
        pass

    assert native_propagator_identity(_system(_Value())) is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_native_values_bypass(value):
    assert native_propagator_identity(_system([value])) is None


def test_spec_subclass_and_instance_overrides_bypass():
    class _Linear(sw.DiagonalA):
        pass

    system = _system()
    assert native_propagator_identity(replace(system, linear=_Linear([1.0]))) is None
    object.__setattr__(system.linear, "build_R_callable", _OpaqueCallable())
    assert native_propagator_identity(system) is None


@pytest.mark.parametrize(
    "cls,name",
    [
        (sw.DiagonalA, "build_R_callable"),
        (sw.ExponentialSpatial, "__call__"),
        (specs._StaticIsoR, "__call__"),
    ],
)
def test_replaced_native_class_method_bypasses(monkeypatch, cls, name):
    system = _system()
    assert native_propagator_identity(system) is not None
    monkeypatch.setattr(cls, name, _OpaqueCallable())
    assert native_propagator_identity(system) is None


def test_native_function_code_mutation_bypasses(monkeypatch):
    function = sw.ExponentialSpatial.__call__

    def replacement(self, dr):
        raise AssertionError("mutated protocol must not run")

    monkeypatch.setattr(function, "__code__", replacement.__code__)
    assert native_propagator_identity(_system()) is None


@pytest.mark.parametrize(
    "attribute,value", [("__defaults__", (1.0,)), ("__kwdefaults__", {"dr": 1.0})]
)
def test_native_function_defaults_mutation_bypasses(monkeypatch, attribute, value):
    monkeypatch.setattr(sw.ExponentialSpatial.__call__, attribute, value)
    assert native_propagator_identity(_system()) is None


def test_runtime_keyword_metadata_rejects_hostile_keys_before_lookup(monkeypatch):
    class _Key:
        def __hash__(self):
            return hash("diag_C")

        def __eq__(self, _other):
            raise AssertionError("identity must not compare a user metadata key")

    defaults = dict(sw.System.propagators.__kwdefaults__)
    original = defaults.pop("diag_C")
    defaults[_Key()] = original
    monkeypatch.setattr(sw.System.propagators, "__kwdefaults__", defaults)
    assert native_propagator_identity(_system()) is None


def test_modified_native_system_getter_bypasses_without_invocation(monkeypatch):
    system = _system()
    monkeypatch.setattr(sw.System, "iso_R", property(_OpaqueCallable()))
    assert native_propagator_identity(system) is None


@pytest.mark.parametrize("target", [sw.System, sw.DiagonalA])
@pytest.mark.parametrize("attribute", ["func", "attrname"])
def test_cached_property_mutation_invalidates_admission(monkeypatch, target, attribute):
    name = "_effective_linear" if target is sw.System else "_gamma_grid"
    descriptor = vars(target)[name]
    assert native_propagator_identity(_system()) is not None
    replacement = _OpaqueCallable() if attribute == "func" else "different_cache_slot"
    monkeypatch.setattr(descriptor, attribute, replacement)
    assert native_propagator_identity(_system()) is None


def test_mutable_class_display_name_cannot_merge_kernel_families():
    exponential = _system(
        kappa2=sw.SeparableTranslation(
            sw.ExponentialTemporal(1.0, 1.0, is_even=True, has_diagonal_cusp=False),
            sw.ExponentialSpatial(1.0),
        )
    )
    gaussian = _system(
        kappa2=sw.SeparableTranslation(
            sw.GaussianTemporal(1.0, 1.0, is_even=True, has_diagonal_cusp=False),
            sw.ExponentialSpatial(1.0),
        )
    )
    original = native_propagator_identity(gaussian)
    original_name = sw.GaussianTemporal.__name__
    sw.GaussianTemporal.__name__ = "ExponentialTemporal"
    try:
        assert native_propagator_identity(gaussian) == original
        assert native_propagator_identity(gaussian) != native_propagator_identity(
            exponential
        )
    finally:
        sw.GaussianTemporal.__name__ = original_name


@pytest.mark.parametrize(
    "mutation",
    [
        "replacement",
        "code",
        "property",
        "staticmethod",
        "cached_property",
        "closed_form_replacement",
        "closed_form_code",
        "kwdefaults_subclass",
        "kwdefaults_non_string",
    ],
)
def test_first_identity_import_rejects_prepatched_protocol(mutation, tmp_path):
    # Each interpreter imports the identity module only after the patch, so
    # this cannot pass merely by seeing an earlier runtime snapshot.
    script = textwrap.dedent("""
        import sys
        import sft_wick as sw
        from sft_wick.workflow import closed_forms
        from functools import cached_property
        assert "sft_wick.workflow._propagator_identity" not in sys.modules
        system = sw.System(
            field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA([1.0]),
            noise=sw.GaussianNoise(sw.SeparableTranslation(
                sw.ExponentialTemporal(1.0, 1.0), sw.ExponentialSpatial(1.0))))
        class Opaque:
            def __bool__(self): raise AssertionError("truth callback")
            def __getattribute__(self, name):
                if name in ("__code__", "__defaults__", "__kwdefaults__"):
                    raise AssertionError("attribute callback")
                return object.__getattribute__(self, name)
            def __call__(self, *args): raise AssertionError("call callback")
        class OpaqueDict(dict):
            def __getattribute__(self, name):
                if name == "items": raise AssertionError("metadata getter callback")
                return super().__getattribute__(name)
            def __iter__(self): raise AssertionError("metadata iterator callback")
        def foreign(self): raise AssertionError("foreign callback")
        # Changing .__code__ deliberately retains the original native
        # function object and .__module__, defeating module-only checks.
        mutation = sys.argv[1]
        if mutation == "replacement":
            foreign.__module__ = "sft_wick.workflow.specs"
            sw.DiagonalA.build_R_callable = foreign
        elif mutation == "code":
            sw.DiagonalA.build_R_callable.__code__ = foreign.__code__
        elif mutation == "property":
            sw.DiagonalA.is_iso_R = property(Opaque())
        elif mutation == "staticmethod":
            sw.DiagonalA._iso_from_array = staticmethod(Opaque())
        elif mutation == "cached_property":
            vars(sw.System)["_effective_linear"].func = Opaque()
        elif mutation == "closed_form_replacement":
            closed_forms.builtin_closed_form_for = foreign
        elif mutation == "kwdefaults_subclass":
            sw.System.build_propagator_model.__kwdefaults__ = OpaqueDict(diag_C=True)
        elif mutation == "kwdefaults_non_string":
            sw.System.build_propagator_model.__kwdefaults__ = {object(): True}
        else:
            closed_forms.builtin_closed_form_for.__code__ = foreign.__code__
        from sft_wick.workflow._propagator_identity import (
            native_propagator_identity, native_closed_form_identity)
        assert native_propagator_identity(system) is None
        assert native_closed_form_identity(None) is None
    """)
    root = Path(sw.__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(tmp_path / "bytecode"),
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, mutation],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("gamma,n_components", [([1.0], 1), ([1.0, 2.0], 2)])
def test_native_response_pickle_preserves_model_identity(gamma, n_components):
    import pickle

    system = replace(_system(gamma), field=sw.FieldSpec("phi", n_components))
    expected = native_propagator_identity(system)
    assert expected is not None
    response = system.linear.build_R_callable()
    pickle.loads(pickle.dumps(response))
    assert native_propagator_identity(system) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "code_constants",
        "code_opaque_constant",
        "code_string_subclass",
        "generated_code",
        "wrapped_repr_code",
        "cached_property_code",
        "property_code",
        "module_math_code",
        "defaults",
        "keyword_default",
        "function_metadata_key",
        "keyword_metadata_key",
    ],
)
def test_first_identity_import_rejects_definition_preserving_mutation(
    mutation, tmp_path
):
    script = textwrap.dedent("""
        import sys
        import sft_wick as sw
        from sft_wick.workflow import closed_forms
        assert "sft_wick.workflow._propagator_identity" not in sys.modules
        system = sw.System(
            field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA([1.0]),
            noise=sw.GaussianNoise(sw.SeparableTranslation(
                sw.GaussianTemporal(1.0, 1.0), sw.ExponentialSpatial(1.0))))
        class Opaque:
            def __eq__(self, other): raise AssertionError("equality callback")
            def __bool__(self): raise AssertionError("truth callback")
        class OpaqueStr(str):
            __hash__ = str.__hash__
            def __eq__(self, other): raise AssertionError("string equality callback")
        mutation = sys.argv[1]
        function = sw.GaussianTemporal.__call__
        if mutation == "code_constants":
            before = sw.GaussianTemporal(1.0, 1.0)(1.0)
            old = function.__code__
            function.__code__ = old.replace(co_consts=tuple(
                4.0 if type(value) is float and value == 2.0 else value
                for value in old.co_consts))
            assert function.__code__.co_filename == old.co_filename
            assert function.__code__.co_firstlineno == old.co_firstlineno
            assert sw.GaussianTemporal(1.0, 1.0)(1.0) != before
        elif mutation == "code_string_subclass":
            function.__code__ = function.__code__.replace(
                co_name=OpaqueStr(function.__code__.co_name))
        elif mutation == "defaults":
            function.__defaults__ = (Opaque(),)
        elif mutation == "keyword_default":
            sw.System.build_propagator_model.__kwdefaults__ = {"diag_C": Opaque()}
        elif mutation == "function_metadata_key":
            function.__dict__[OpaqueStr("metadata")] = None
        elif mutation == "keyword_metadata_key":
            sw.System.build_propagator_model.__kwdefaults__ = {OpaqueStr("diag_C"): None}
        else:
            if mutation == "generated_code":
                function = sw.FieldSpec.__init__
            elif mutation == "wrapped_repr_code":
                function = vars(sw.FieldSpec.__repr__)["__wrapped__"]
            elif mutation == "cached_property_code":
                function = vars(sw.System)["_effective_linear"].func
            elif mutation == "property_code":
                function = vars(sw.System)["n_components"].fget
            elif mutation == "module_math_code":
                function = closed_forms.ou_white_psi
            old = function.__code__
            function.__code__ = old.replace(co_consts=(Opaque(), *old.co_consts[1:]))
        from sft_wick.workflow._propagator_identity import (
            native_propagator_identity, native_closed_form_identity)
        assert native_propagator_identity(system) is None
        assert native_closed_form_identity(None) is None
        """)
    root = Path(sw.__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(tmp_path / "bytecode"),
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, mutation],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mode",
    ["pickle", "all_slots", "nonempty", "tuple", "list_subclass"],
)
def test_first_identity_import_handles_native_pickle_slot_cache(mode, tmp_path):
    script = textwrap.dedent("""
        import copyreg
        import pickle
        import sys
        import sft_wick as sw
        from sft_wick.workflow import specs, closed_forms

        assert "sft_wick.workflow._propagator_identity" not in sys.modules
        system = sw.System(
            field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA([1.0]),
            noise=sw.GaussianNoise(sw.SeparableTranslation(
                sw.ExponentialTemporal(1.0, 1.0), sw.ExponentialSpatial(1.0))))
        mode = sys.argv[1]
        if mode == "pickle":
            restored = pickle.loads(pickle.dumps(system))
            assert vars(sw.System)["__slotnames__"] == []
        elif mode == "all_slots":
            names = (
                "FieldSpec", "DiagonalA", "GaussianNoise",
                "SeparableTranslation", "SeparableRotation",
                "ExponentialTemporal", "GaussianTemporal", "ExponentialSpatial",
                "GaussianSpatial", "LegendreAngular", "ConstantImpulse",
                "LinearOp", "Kappa2", "Sigma2", "_StaticIsoR", "_StaticMatR",
                "_SeparableTranslationKappa2", "_SeparableRotationKappa2",
                "_ConstantImpulseIso", "_ConstantImpulseMat",
            )
            for cls in (sw.System, closed_forms.ClosedFormC,
                        *(vars(specs)[name] for name in names)):
                assert copyreg._slotnames(cls) == list(vars(cls).get("__slots__", ()))
        else:
            class HostileList(list):
                def __bool__(self): raise AssertionError("truth callback")
                def __iter__(self): raise AssertionError("iterator callback")
            sw.System.__slotnames__ = {
                "nonempty": ["field"], "tuple": (), "list_subclass": HostileList()
            }[mode]
        from sft_wick.workflow import _propagator_identity as identity
        if mode in ("pickle", "all_slots"):
            key = identity.native_propagator_identity(system)
            assert key is not None
            assert identity._ORIGIN_NATIVE
            sw.System.__slotnames__.append("field")
            assert identity.native_propagator_identity(system) is None
            sw.System.__slotnames__ = []
            assert identity.native_propagator_identity(system) == key
        else:
            assert not identity._ORIGIN_NATIVE
            assert identity.native_propagator_identity(system) is None
        """)
    root = Path(sw.__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(tmp_path / "bytecode"),
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, mode],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_rebound_native_lowering_class_bypasses(monkeypatch):
    monkeypatch.setattr(specs, "_StaticIsoR", _OpaqueCallable)
    assert native_propagator_identity(_system()) is None


def test_builtin_closed_form_has_complete_native_identity():
    first = closed_forms.builtin_closed_form_for(_system())
    second = closed_forms.builtin_closed_form_for(_system([2.0]))
    assert native_closed_form_identity(None) == ("none",)
    assert native_closed_form_identity(first) is not None
    assert native_closed_form_identity(first) != native_closed_form_identity(second)
    assert native_closed_form_identity(_OpaqueCallable()) is None


def test_modified_builtin_closed_form_implementation_bypasses(monkeypatch):
    first = closed_forms.builtin_closed_form_for(_system())
    monkeypatch.setattr(closed_forms.ClosedFormC, "__call__", _OpaqueCallable())
    assert native_closed_form_identity(first) is None


def test_explicit_R_same_disk_path_never_reuses_another_model(tmp_path):
    first = _system(
        explicit_R=lambda t1, t2: float(np.exp(-(t1 - t2))) if t1 >= t2 else 0.0
    )
    second = _system(
        explicit_R=lambda t1, t2: float(np.exp(-2 * (t1 - t2))) if t1 >= t2 else 0.0
    )
    path = tmp_path / "same_tables"
    left, right = _props(first, path), _props(second, path)
    assert left.cache.model.R_time(1.0, 0.0) == np.exp(-1.0)
    assert right.cache.model.R_time(1.0, 0.0) == np.exp(-2.0)
    assert not path.exists()


@pytest.mark.parametrize("custom", [lambda x1, t1, x2, t2: np.eye(1)])
def test_user_closed_form_does_not_enter_shared_disk_cache(
    monkeypatch, tmp_path, custom
):
    paths = []
    original = disk_cache.load_or_compute

    def spy(path, *args, **kwargs):
        paths.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(disk_cache, "load_or_compute", spy)
    _system().propagators(
        t_max=1.0, n_grid_t=4, c_closed_form=custom, cache_path=tmp_path / "custom"
    )
    assert paths == []


def test_same_native_model_reuses_directory_cache(monkeypatch, tmp_path):
    system = _system()
    original = sw.System.build_propagator_model
    calls = []

    # Instrument the compute callback instead of the guarded model protocol.
    original_load = disk_cache.load_or_compute

    def spy(path, spec, compute, **kwargs):
        def counted():
            calls.append(True)
            return compute()

        return original_load(path, spec, counted, **kwargs)

    monkeypatch.setattr(disk_cache, "load_or_compute", spy)
    first = _props(system, tmp_path / "native")
    second = _props(replace(system, linear=sw.DiagonalA([1.0])), tmp_path / "native")
    assert original is sw.System.build_propagator_model
    assert calls == [True]
    assert first.cache.model.R_time(1.0, 0.0) == second.cache.model.R_time(1.0, 0.0)
    assert list((tmp_path / "native").glob("*.joblib"))
