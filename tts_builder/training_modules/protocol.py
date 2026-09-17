from __future__ import annotations

import math
import operator
from typing import Callable

from .models import FieldDescriptor, FrameworkDescriptor, ModuleDescriptor


_LOCALES = {"zh_CN", "en", "ja"}
_CAPABILITIES = {"preprocess", "train", "evaluate", "listen", "promote"}
_KINDS = {"string", "integer", "number", "boolean", "enum", "path"}
_ROOT_FIELDS = {"protocol_version", "module_id", "module_version", "frameworks"}
_FRAMEWORK_FIELDS = {"id", "display_name", "capabilities", "fields"}
_FIELD_FIELDS = {"key", "kind", "default", "constraints", "labels"}
_CONSTRAINT_FIELDS = {
    "string": set(),
    "boolean": set(),
    "path": set(),
    "integer": {"minimum", "maximum", "exclusive_minimum", "exclusive_maximum"},
    "number": {"minimum", "maximum", "exclusive_minimum", "exclusive_maximum"},
    "enum": {"choices"},
}
_NUMERIC_RULES = {
    "minimum": operator.ge,
    "maximum": operator.le,
    "exclusive_minimum": operator.gt,
    "exclusive_maximum": operator.lt,
}


def _is_string(value: object) -> bool:
    return isinstance(value, str)


def _is_integer(value: object) -> bool:
    return type(value) is int


def _is_number(value: object) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _is_boolean(value: object) -> bool:
    return type(value) is bool


_DEFAULT_VALIDATORS: dict[str, Callable[[object], bool]] = {
    "string": _is_string,
    "integer": _is_integer,
    "number": _is_number,
    "boolean": _is_boolean,
    "enum": _is_string,
    "path": _is_string,
}


def parse_descriptor(payload: object) -> ModuleDescriptor:
    root = _object(payload, _ROOT_FIELDS, "descriptor")
    if type(root["protocol_version"]) is not int or root["protocol_version"] != 1:
        raise ValueError("protocol_version must be integer 1")
    frameworks = _nonempty_list(root["frameworks"], "frameworks")
    parsed = tuple(_parse_framework(value) for value in frameworks)
    _unique((framework.id for framework in parsed), "framework id")
    return ModuleDescriptor(
        protocol_version=1,
        module_id=_text(root["module_id"], "module_id"),
        module_version=_text(root["module_version"], "module_version"),
        frameworks=parsed,
    )


def _parse_framework(payload: object) -> FrameworkDescriptor:
    value = _object(payload, _FRAMEWORK_FIELDS, "framework")
    capabilities = tuple(_nonempty_list(value["capabilities"], "capabilities"))
    if any(not isinstance(item, str) or item not in _CAPABILITIES for item in capabilities):
        raise ValueError("unsupported capability")
    _unique(capabilities, "capability")
    fields = tuple(
        _parse_field(field) for field in _list(value["fields"], "framework fields")
    )
    _unique((field.key for field in fields), "field key")
    return FrameworkDescriptor(
        id=_text(value["id"], "framework id"),
        display_name=_text(value["display_name"], "framework display_name"),
        capabilities=capabilities,
        fields=fields,
    )


def _parse_field(payload: object) -> FieldDescriptor:
    value = _object(payload, _FIELD_FIELDS, "field")
    kind = value["kind"]
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("unsupported field kind")
    default = value["default"]
    if not _DEFAULT_VALIDATORS[kind](default):
        raise ValueError(f"field default must be {kind}")
    constraints = _object(
        value["constraints"], _CONSTRAINT_FIELDS[kind], "constraints", require_all=False
    )
    labels = _object(value["labels"], _LOCALES, "labels")
    parsed_labels = {locale: _text(labels[locale], f"labels.{locale}") for locale in _LOCALES}
    _validate_constraints(kind, default, constraints)
    return FieldDescriptor(
        key=_text(value["key"], "field key"),
        kind=kind,
        default=default,
        constraints=dict(constraints),
        labels=parsed_labels,
    )


def _validate_constraints(kind: str, default: object, constraints: dict) -> None:
    if kind in {"integer", "number"} and any(
        not _is_number(value) for value in constraints.values()
    ):
        raise ValueError(f"{kind} constraints must be finite numbers")
    if kind in {"integer", "number"} and any(
        not _NUMERIC_RULES[name](default, boundary)
        for name, boundary in constraints.items()
    ):
        raise ValueError(f"{kind} default violates constraints")
    if kind == "enum":
        choices = constraints.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or any(not isinstance(choice, str) for choice in choices)
            or len(set(choices)) != len(choices)
            or default not in choices
        ):
            raise ValueError("enum choices must be unique strings containing the default")


def _object(
    value: object,
    fields: set[str],
    name: str,
    *,
    require_all: bool = True,
) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value) if require_all else set()
    if unknown:
        raise ValueError(f"unknown {name} field: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {name} field: {', '.join(sorted(missing))}")
    return value


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _nonempty_list(value: object, name: str) -> list:
    values = _list(value, name)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _unique(values, name: str) -> None:
    values = tuple(values)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")
