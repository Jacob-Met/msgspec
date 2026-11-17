from __future__ import annotations

from typing import Annotated, Union

import pytest

import msgspec

RGB = tuple[float, float, float]
BORDER = RGB | tuple[RGB, RGB, RGB, RGB]
THREE = tuple[int, int, int]
FOUR = tuple[int, int, int, int]
TRIPLE_SHAPE = tuple[()] | tuple[int] | tuple[int, int, int]


class Wrapped(msgspec.Struct):
    values: list[BORDER]
    optional: BORDER | None = None


@pytest.mark.parametrize(
    "typehint,examples",
    [
        (
            RGB | tuple[RGB, RGB, RGB, RGB],
            [
                (1.0, 2.0, 3.0),
                ((1.0, 2.0, 3.0),) * 4,
            ],
        ),
        (THREE | FOUR, [(1, 2, 3), (1, 2, 3, 4)]),
        (TRIPLE_SHAPE, [(), (7,), (1, 2, 3)]),
        (Union[THREE, FOUR], [(1, 2, 3), (1, 2, 3, 4)]),
        (THREE | FOUR | None, [None, (1, 2, 3), (1, 2, 3, 4)]),
        (tuple[int, int] | tuple[str, str], [(1, 2), ("a", "b")]),
        (tuple[bool, int] | tuple[str, int], [(True, 2), ("yes", 2)]),
        (tuple[None] | tuple[dict[str, int]], [(None,), ({"hi": 2},)]),
        (tuple[bytes] | tuple[int], [(b"binary",), (42,)]),
        (tuple[list[int], int] | tuple[str, int], [([1, 2], 3), ("a", 4)]),
        (tuple[None | int] | tuple[str], [(None,), (1,), ("hello",)]),
    ],
)
def test_tuple_union_all_decode_paths(typehint, examples):
    for obj in examples:
        for protocol in (msgspec.json, msgspec.msgpack):
            encoded = protocol.encode(obj)
            decoder = protocol.Decoder(typehint)
            assert decoder.decode(encoded) == obj
            assert protocol.decode(encoded, type=typehint) == obj
        assert msgspec.convert(obj, type=typehint) == obj


@pytest.mark.parametrize(
    "typehint",
    [
        THREE | FOUR,
        Union[THREE, FOUR],
        BORDER,
        list[BORDER],
        dict[str, list[BORDER]],
        Wrapped,
    ],
)
def test_tuple_union_can_be_compiled_multiple_times_without_shared_state(typehint):
    for _ in range(30):
        msgspec.json.Decoder(typehint)
        msgspec.msgpack.Decoder(typehint)
        msgspec.convert([], type=list[BORDER])


def test_nested_tuple_union_values():
    obj = Wrapped(
        values=[
            (1.0, 2.0, 3.0),
            ((1.0, 2.0, 3.0),) * 4,
        ],
        optional=((2.0, 3.0, 4.0),) * 4,
    )
    for protocol in (msgspec.json, msgspec.msgpack):
        encoded = protocol.encode(obj)
        assert protocol.decode(encoded, type=Wrapped) == obj
        assert protocol.Decoder(Wrapped).decode(encoded) == obj
    data = msgspec.to_builtins(obj)
    assert msgspec.convert(data, type=Wrapped) == obj


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [1, 2],
        [1, 2, 3, 4, 5],
        [1, "bad", 3],
        [1, 2, 3, "bad"],
    ],
)
@pytest.mark.parametrize("protocol", [msgspec.json, msgspec.msgpack])
def test_tuple_union_wrong_shape_or_nested_type(protocol, bad):
    with pytest.raises(msgspec.ValidationError):
        protocol.decode(protocol.encode(bad), type=THREE | FOUR)
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(bad, type=THREE | FOUR)


@pytest.mark.parametrize(
    "bad",
    [
        b"[1,2,3,]",
        b"[1,2,3",
        b"[1,2,3]]",
        b"[[1,2],[3,4]",
        b"[[1,2], [3,4],",
        b"[[1,2], ]",
        b"[1, 2, not-json]",
    ],
)
def test_json_tuple_union_malformed_payload(bad):
    with pytest.raises(msgspec.DecodeError):
        msgspec.json.decode(bad, type=THREE | FOUR)


def test_tuple_union_member_constraints():
    smaller = tuple[Annotated[int, msgspec.Meta(ge=1)], int, int]
    larger = tuple[int, int, int, Annotated[int, msgspec.Meta(le=4)]]
    typ = smaller | larger
    for val in ((1, 2, 3), (1, 2, 3, 4)):
        for protocol in (msgspec.json, msgspec.msgpack):
            assert protocol.decode(protocol.encode(val), type=typ) == val
    for val in ((0, 2, 3), (1, 2, 3, 5)):
        for protocol in (msgspec.json, msgspec.msgpack):
            with pytest.raises(msgspec.ValidationError):
                protocol.decode(protocol.encode(val), type=typ)
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert([1, 2], type=typ)


@pytest.mark.parametrize("proto", [msgspec.json, msgspec.msgpack])
def test_malformed_element_is_reported_at_original_path(proto):
    typ = dict[str, list[BORDER]]
    data = {"outer": [[1, 2, "invalid"]]}
    with pytest.raises(msgspec.ValidationError) as e:
        proto.decode(proto.encode(data), type=typ)
    assert "[...][0][2]" in str(e.value)


@pytest.mark.parametrize(
    "typehint",
    [
        tuple[int] | tuple[float],
        tuple[int, int] | tuple[float, float],
        tuple[bytes] | tuple[str],
        tuple[list[int]] | tuple[tuple[int, ...]],
        tuple[int, ...] | tuple[int, int],
    ],
)
def test_indistinguishable_tuple_union_still_requires_disambiguation(typehint):
    with pytest.raises(TypeError, match="more than one array-like"):
        msgspec.json.Decoder(typehint)
    with pytest.raises(TypeError, match="more than one array-like"):
        msgspec.msgpack.Decoder(typehint)


def test_tuple_union_strict_conversion_and_extra_scalar_branch():
    typ = THREE | FOUR
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert([1, "2", 3], type=typ)
    assert msgspec.convert([1, "2", 3], type=typ, strict=False) == (1, 2, 3)
    with pytest.raises(TypeError, match="more than one array-like"):
        msgspec.msgpack.Decoder(THREE | FOUR | str)


@pytest.mark.parametrize("proto", [msgspec.json, msgspec.msgpack])
def test_equal_length_union_wrong_first_kind(proto):
    typ = tuple[int, int] | tuple[str, str]
    for obj in ((True, 1), ([], []), (None, None)):
        with pytest.raises(msgspec.ValidationError):
            proto.decode(proto.encode(obj), type=typ)
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(obj, type=typ)


def test_json_tuple_unions_do_not_skip_input_validation():
    typ = tuple[str, str] | tuple[list[int], int]
    for blob in (b'["a",NaN]', b'["a","b",]', b"[[1,2],1,]", b"[[1,2],1] garbage"):
        with pytest.raises(msgspec.DecodeError):
            msgspec.json.decode(blob, type=typ)
