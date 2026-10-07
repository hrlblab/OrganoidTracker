"""Organoid and cyst annotations in source-pixel coordinates on the annotation frame.

An organoid is the marker point the user clicked; its cysts are the boxes drawn around cysts on
the annotation frame (the last chronological frame in reverse mode). Cyst ids are the tracker's
global object ids (the GUI numbers them 1, 2, 3, ... across organoids). An organoid may have no
cysts: it still counts in the population statistics (percentage of organoids with cysts), so it
is kept here and in every export.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class AnnotationError(ValueError):
    """Annotations that cannot be tracked or analyzed as given."""


ORGANOID_KEYS = ("organoid_id", "point", "cysts")
CYST_KEYS = ("cyst_id", "bbox")


def _check_keys(mapping: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise AnnotationError(f"{where}: unknown key(s) {unknown}; valid keys are {list(allowed)}")
    missing = [key for key in allowed if key not in mapping]
    if missing:
        raise AnnotationError(f"{where}: missing key(s) {missing}")


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationError(f"{where}: expected a number, got {value!r}")
    if not math.isfinite(value):
        raise AnnotationError(f"{where}: expected a finite number, got {value!r}")
    return value


def _identifier(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AnnotationError(f"{where}: expected a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class CystAnnotation:
    """A cyst: the tracker's object id and its box on the annotation frame."""

    cyst_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels

    def __post_init__(self) -> None:
        _identifier(self.cyst_id, "cyst_id")
        where = f"cyst {self.cyst_id}"
        box = tuple(self.bbox)
        if len(box) != 4:
            raise AnnotationError(f"{where}: bbox must have 4 values [x1, y1, x2, y2], got {len(box)}")
        x1, y1, x2, y2 = (_number(value, f"{where}: bbox") for value in box)
        if min(x1, y1, x2, y2) < 0:
            raise AnnotationError(f"{where}: bbox coordinates must not be negative: {list(box)}")
        if not (x1 < x2 and y1 < y2):
            raise AnnotationError(f"{where}: bbox must satisfy x1 < x2 and y1 < y2: {list(box)}")
        object.__setattr__(self, "bbox", box)

    def check_inside(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.bbox
        if x2 > width or y2 > height:
            raise AnnotationError(
                f"cyst {self.cyst_id}: bbox {list(self.bbox)} exceeds the frame size {width}x{height}"
            )

    def to_document(self) -> dict[str, Any]:
        return {"cyst_id": self.cyst_id, "bbox": list(self.bbox)}


@dataclass(frozen=True)
class OrganoidAnnotation:
    """An organoid marker point and the cysts assigned to it (possibly none)."""

    organoid_id: int
    point: tuple[float, float]
    cysts: tuple[CystAnnotation, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.organoid_id, "organoid_id")
        where = f"organoid {self.organoid_id}"
        point = tuple(self.point)
        if len(point) != 2:
            raise AnnotationError(f"{where}: point must have 2 values [x, y], got {len(point)}")
        for value in point:
            _number(value, f"{where}: point")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "cysts", tuple(self.cysts))

    def to_document(self) -> dict[str, Any]:
        return {
            "organoid_id": self.organoid_id,
            "point": list(self.point),
            "cysts": [cyst.to_document() for cyst in self.cysts],
        }


@dataclass(frozen=True)
class AnnotationSet:
    """All organoids of a run, with unique organoid ids and globally unique cyst ids."""

    organoids: tuple[OrganoidAnnotation, ...]

    def __post_init__(self) -> None:
        organoids = tuple(self.organoids)
        object.__setattr__(self, "organoids", organoids)
        seen_organoids: set[int] = set()
        owners: dict[int, int] = {}
        for organoid in organoids:
            if organoid.organoid_id in seen_organoids:
                raise AnnotationError(f"organoid id {organoid.organoid_id} is used twice")
            seen_organoids.add(organoid.organoid_id)
            for cyst in organoid.cysts:
                if cyst.cyst_id in owners:
                    raise AnnotationError(
                        f"cyst id {cyst.cyst_id} is used twice (organoids {owners[cyst.cyst_id]} and "
                        f"{organoid.organoid_id}); cyst ids are global object ids"
                    )
                owners[cyst.cyst_id] = organoid.organoid_id

    @property
    def cysts(self) -> list[CystAnnotation]:
        return [cyst for organoid in self.organoids for cyst in organoid.cysts]

    def cyst_ids(self) -> list[int]:
        return [cyst.cyst_id for cyst in self.cysts]

    def organoids_without_cysts(self) -> list[int]:
        return [organoid.organoid_id for organoid in self.organoids if not organoid.cysts]

    def check_inside(self, width: int, height: int) -> None:
        """Raise when a box lies outside a ``width`` x ``height`` frame."""
        for cyst in self.cysts:
            cyst.check_inside(width, height)

    def organoid_data(self) -> dict[int, dict[str, Any]]:
        """The ``{organoid_id: {"point": (x, y), "cysts": [{"cyst_id", "bbox"}]}}`` shape the analysis consumes."""
        return {
            organoid.organoid_id: {
                "point": organoid.point,
                "cysts": [{"cyst_id": cyst.cyst_id, "bbox": cyst.bbox} for cyst in organoid.cysts],
            }
            for organoid in self.organoids
        }

    @classmethod
    def from_organoid_data(cls, data: Mapping[int, Mapping[str, Any]]) -> AnnotationSet:
        """From the GUI's ``organoid_data`` mapping (see :meth:`organoid_data`)."""
        return cls.from_documents(
            [
                {"organoid_id": organoid_id, "point": list(info["point"]), "cysts": list(info.get("cysts", []))}
                for organoid_id, info in data.items()
            ]
        )

    @classmethod
    def from_documents(cls, items: Iterable[Any], where: str = "organoids") -> AnnotationSet:
        """From JSON entries ``[{"organoid_id", "point", "cysts": [{"cyst_id", "bbox"}]}]``.

        Keys and container types are checked at every level: an unknown or missing key is named
        (a misspelled ``cysts`` is an error, not an organoid without cysts), and ``cysts`` must be
        a list (``[]`` for an organoid without cysts).
        """
        organoids = []
        for index, item in enumerate(items):
            here = f"{where}[{index}]"
            if not isinstance(item, Mapping):
                raise AnnotationError(f"{here}: expected an object with the keys {list(ORGANOID_KEYS)}")
            _check_keys(item, ORGANOID_KEYS, here)
            raw_cysts = item["cysts"]
            if not isinstance(raw_cysts, (list, tuple)):
                raise AnnotationError(
                    f"{here}.cysts: expected a list of cysts ([] for an organoid without cysts), "
                    f"got {type(raw_cysts).__name__}"
                )
            cysts = []
            for cyst_index, cyst in enumerate(raw_cysts):
                there = f"{here}.cysts[{cyst_index}]"
                if not isinstance(cyst, Mapping):
                    raise AnnotationError(f"{there}: expected an object with the keys {list(CYST_KEYS)}")
                _check_keys(cyst, CYST_KEYS, there)
                bbox = cyst["bbox"]
                if not isinstance(bbox, (list, tuple)):
                    raise AnnotationError(f"{there}.bbox: expected a list [x1, y1, x2, y2]")
                cysts.append(CystAnnotation(cyst_id=cyst["cyst_id"], bbox=tuple(bbox)))
            point = item["point"]
            if not isinstance(point, (list, tuple)):
                raise AnnotationError(f"{here}.point: expected a list [x, y]")
            organoids.append(
                OrganoidAnnotation(organoid_id=item["organoid_id"], point=tuple(point), cysts=tuple(cysts))
            )
        return cls(tuple(organoids))

    def to_documents(self) -> list[dict[str, Any]]:
        return [organoid.to_document() for organoid in self.organoids]
