from __future__ import annotations

import datetime
import enum
import uuid
from dataclasses import dataclass
from typing import Annotated, Literal, NamedTuple, TypedDict

import pytest

import msgspec


class Choice(enum.Enum):
    ITEM = "item"


class Payload(msgspec.Struct):
    content: str | bytes


class Nested(msgspec.Struct):
    payload: list[Payload]


class Recursive(msgspec.Struct):
    content: str | bytes
    child: Recursive | None = None


class TaggedOne(msgspec.Struct, tag=True):
    content: str | bytes


class TaggedTwo(msgspec.Struct, tag=True):
    content: int


class IntegerTaggedOne(msgspec.Struct, tag=11):
    content: str | bytes


class IntegerTaggedTwo(msgspec.Struct, tag=13):
    content: int


class TypedPayload(TypedDict):
    content: str | bytes


@dataclass
class DataPayload:
    content: str | bytes


class NamedPayload(NamedTuple):
    content: str | bytes


@pytest.mark.parametrize(
    ("typ", "str_value", "bin_value"),
    [
        (str | bytes, "abc", b"abc"),
        (bytes | str, "abc", b"abc"),
        (str | bytearray, "abc", bytearray(b"abc")),
        (str | memoryview, "abc", memoryview(b"abc")),
        (Literal["item"] | bytes, "item", b"abc"),
        (Choice | bytes, Choice.ITEM, b"abc"),
        (datetime.date | bytes, datetime.date(2026, 11, 17), b"abc"),
    ],
)
def test_msgpack_disjoint_string_binary_unions(typ, str_value, bin_value):
    decoder = msgspec.msgpack.Decoder(typ)
    for value in (str_value, bin_value):
        result = decoder.decode(msgspec.msgpack.encode(value))
        if isinstance(value, (bytes, bytearray, memoryview)):
            assert bytes(result) == bytes(value)
            assert type(result) is type(value)
        else:
            assert result == value
            assert type(result) is type(value)

    assert msgspec.convert(str_value, typ) == str_value
    assert msgspec.convert(bin_value, typ) == bin_value


@pytest.mark.parametrize("typ", [str | bytes, str | bytearray, str | memoryview])
def test_msgpack_disjoint_union_struct_and_containers(typ):
    Item = msgspec.defstruct("Item", [("data", typ)])

    value = [Item("hello"), Item(b"\x00\xff")]
    result = msgspec.msgpack.decode(msgspec.msgpack.encode(value), type=list[Item])
    assert result[0].data == "hello"
    assert bytes(result[1].data) == b"\x00\xff"


def test_msgpack_nested_recursive_and_tagged_structs():
    root = Recursive("root", Recursive(b"\x00\xff"))
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(root), type=Recursive) == root
    items = [TaggedOne(b"data"), TaggedTwo(3), TaggedOne("hi")]
    assert (
        msgspec.msgpack.decode(
            msgspec.msgpack.encode(items), type=list[TaggedOne | TaggedTwo]
        )
        == items
    )
    assert msgspec.msgpack.decode(
        msgspec.msgpack.encode({"content": b"\x00\xff"}), type=TypedPayload
    ) == {"content": b"\x00\xff"}
    assert msgspec.msgpack.decode(
        msgspec.msgpack.encode(DataPayload(b"data")), type=DataPayload
    ) == DataPayload(b"data")
    assert msgspec.msgpack.decode(
        msgspec.msgpack.encode(NamedPayload(b"data")), type=NamedPayload
    ) == NamedPayload(b"data")


@pytest.mark.parametrize(
    "typ",
    [
        str | bytes,
        bytes | str,
        str | bytearray,
        Literal["abc"] | bytes,
        Choice | bytes,
        datetime.date | bytes,
        list[str | bytes],
        dict[str, list[str | bytes]],
        Payload,
        Nested,
        Recursive,
        TaggedOne | TaggedTwo,
        list[TaggedOne | TaggedTwo],
        IntegerTaggedOne | IntegerTaggedTwo,
        TypedPayload,
        DataPayload,
        NamedPayload,
    ],
)
def test_json_rejects_ambiguous_string_binary_unions(typ):
    with pytest.raises(TypeError, match="ambiguous in JSON"):
        msgspec.json.Decoder(typ)
    with pytest.raises(TypeError, match="ambiguous in JSON"):
        msgspec.json.decode(b"null", type=typ)


@pytest.mark.parametrize(
    "typ", [str | uuid.UUID, bytes | uuid.UUID, bytes | bytearray, bytes | memoryview]
)
def test_overlapping_msgpack_wire_types_still_rejected(typ):
    with pytest.raises(TypeError, match="more than one"):
        msgspec.msgpack.Decoder(typ)


def test_unambiguous_json_decoders_unchanged():
    for typ, value in [(str | int, "abc"), (uuid.UUID, uuid.uuid4()), (bytes, b"foo")]:
        encoded = msgspec.json.encode(value)
        assert msgspec.json.Decoder(typ).decode(encoded) == value
        assert msgspec.json.decode(encoded, type=typ) == value


def test_binary_union_constraints():
    typ = str | Annotated[bytes, msgspec.Meta(min_length=3)]
    decoder = msgspec.msgpack.Decoder(typ)
    assert decoder.decode(msgspec.msgpack.encode("a")) == "a"
    assert decoder.decode(msgspec.msgpack.encode(b"abc")) == b"abc"
    with pytest.raises(msgspec.ValidationError, match="length"):
        decoder.decode(msgspec.msgpack.encode(b"ab"))
