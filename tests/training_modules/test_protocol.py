from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from tts_builder.training_modules.protocol import parse_descriptor


def descriptor_payload() -> dict:
    return {
        "protocol_version": 2,
        "module_id": "gpt-sovits-v2proplus",
        "module_version": "0.1.0",
        "frameworks": [
            {
                "id": "v2ProPlus",
                "display_name": "GPT-SoVITS v2ProPlus",
                "capabilities": [
                    "preprocess",
                    "train",
                    "evaluate",
                    "listen",
                    "promote",
                    "infer",
                ],
                "training_data": {"kind": "file", "extensions": [".list"]},
                "fields": [
                    {
                        "key": "s2.batch_size",
                        "kind": "integer",
                        "default": 2,
                        "constraints": {"minimum": 1},
                        "labels": {
                            "zh_CN": "S2 批大小",
                            "en": "S2 batch size",
                            "ja": "S2 バッチサイズ",
                        },
                    },
                    {
                        "key": "preprocess.resume",
                        "kind": "boolean",
                        "default": True,
                        "constraints": {},
                        "labels": {
                            "zh_CN": "复用预处理缓存",
                            "en": "Reuse preprocessing cache",
                            "ja": "前処理キャッシュを再利用",
                        },
                    },
                ],
            }
        ],
    }


def test_parse_descriptor_returns_frozen_protocol_models():
    descriptor = parse_descriptor(descriptor_payload())

    assert descriptor.module_id == "gpt-sovits-v2proplus"
    assert descriptor.frameworks[0].capabilities == (
        "preprocess",
        "train",
        "evaluate",
        "listen",
        "promote",
        "infer",
    )
    assert descriptor.frameworks[0].training_data.kind == "file"
    assert descriptor.frameworks[0].training_data.extensions == (".list",)
    assert descriptor.frameworks[0].fields[0].default == 2
    with pytest.raises(FrozenInstanceError):
        descriptor.module_id = "changed"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.update({"unknown": True}), "unknown descriptor field"),
        (
            lambda value: value["frameworks"].append(deepcopy(value["frameworks"][0])),
            "duplicate framework id",
        ),
        (
            lambda value: value["frameworks"][0]["fields"][0]["labels"].pop("ja"),
            "labels",
        ),
        (
            lambda value: value["frameworks"][0]["fields"][0].update(default=True),
            "integer",
        ),
        (
            lambda value: value["frameworks"][0]["capabilities"].append("execute_code"),
            "capability",
        ),
        (
            lambda value: value["frameworks"][0]["fields"][0].update(kind="widget"),
            "field kind",
        ),
        (
            lambda value: value["frameworks"][0]["fields"][0].update(kind=[]),
            "field kind",
        ),
        (
            lambda value: value["frameworks"][0]["capabilities"].append({}),
            "capability",
        ),
        (
            lambda value: value["frameworks"][0]["fields"][0].update(
                kind="number", default=10**1000
            ),
            "number",
        ),
        (
            lambda value: value["frameworks"][0]["fields"][0][
                "constraints"
            ].update(minimum=3),
            "constraints",
        ),
    ],
)
def test_parse_descriptor_rejects_invalid_or_extensible_payloads(mutate, message):
    payload = descriptor_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        parse_descriptor(payload)


@pytest.mark.parametrize(
    "training_data",
    [
        {"kind": "file", "extensions": []},
        {"kind": "directory"},
    ],
)
def test_parse_descriptor_accepts_file_and_directory_training_data(training_data):
    payload = descriptor_payload()
    payload["frameworks"][0]["training_data"] = training_data

    descriptor = parse_descriptor(payload)

    parsed = descriptor.frameworks[0].training_data
    assert parsed.kind == training_data["kind"]
    assert parsed.extensions == tuple(training_data.get("extensions", []))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(protocol_version=1), "unsupported module protocol version 1; expected 2"),
        (lambda value: value["frameworks"][0]["training_data"].update(extra=True), "training_data"),
        (lambda value: value["frameworks"][0].update(training_data={"kind": "archive"}), "training_data"),
        (
            lambda value: value["frameworks"][0].update(
                training_data={"kind": "directory", "extensions": [".wav"]}
            ),
            "extensions",
        ),
        (
            lambda value: value["frameworks"][0]["training_data"].update(
                extensions=["list"]
            ),
            "extension",
        ),
        (
            lambda value: value["frameworks"][0]["training_data"].update(
                extensions=[".*"]
            ),
            "extension",
        ),
    ],
)
def test_parse_descriptor_rejects_invalid_training_data(mutate, message):
    payload = descriptor_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        parse_descriptor(payload)
