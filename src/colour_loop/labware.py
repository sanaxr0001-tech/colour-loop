"""OT-2 deck layout built from real Opentrons labware definitions.

The three definitions in ``labware/`` are copied unchanged from the Opentrons repository
(``shared-data/labware/definitions/2``, commit 5b51a98, the same one PyLabRobot pins):

* ``opentrons_96_tiprack_300ul``                - 300 uL tips
* ``opentrons_15_tuberack_falcon_15ml_conical`` - holds the three dye stocks
* ``nest_96_wellplate_200ul_flat``              - the 96-well plate the colours are mixed in

PyLabRobot can turn tip-rack and tube-rack definitions into resources itself, but only by
downloading them at run time; we feed it the vendored copies instead so a run needs no network.
PyLabRobot has no loader for plates, so ``_plate_from_definition`` does the equivalent here.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

from pylabrobot.resources import Coordinate, Plate, TipRack, Tube, TubeRack, Well
from pylabrobot.resources.opentrons import OTDeck
from pylabrobot.resources.opentrons import load as ot_load
from pylabrobot.resources.well import CrossSectionType, WellBottomType

from .physics import DYES

LABWARE_DIR = Path(__file__).parent / "labware"

TIP_RACK = "opentrons_96_tiprack_300ul"
TUBE_RACK = "opentrons_15_tuberack_falcon_15ml_conical"
PLATE = "nest_96_wellplate_200ul_flat"

# OT-2 deck slots (1 is front-left, 12 is the trash)
TIP_SLOT, DYE_SLOT, PLATE_SLOT = 1, 4, 5

DYE_TUBE_VOLUME_UL = 12_000.0  # each 15 mL tube starts with 12 mL of dye stock


def load_definition(name: str) -> dict:
    return json.loads((LABWARE_DIR / f"{name}.json").read_text())


@contextlib.contextmanager
def _vendored_definitions() -> Iterator[None]:
    """Make PyLabRobot's Opentrons loaders read ``labware/`` instead of downloading."""
    original = ot_load._download_ot_resource_file

    def local(ot_name: str, force_download: bool = False) -> dict:
        return load_definition(ot_name)

    ot_load._download_ot_resource_file = local
    try:
        yield
    finally:
        ot_load._download_ot_resource_file = original


def _plate_from_definition(name: str, ot_name: str) -> Plate:
    data = load_definition(ot_name)
    wells: Dict[str, Well] = {}
    for column in data["ordering"]:
        for item in column:
            spec = data["wells"][item]
            diameter = spec["diameter"]
            well = Well(
                name=f"{name}_{item}",
                size_x=diameter,
                size_y=diameter,
                size_z=spec["depth"],
                bottom_type=WellBottomType.FLAT,
                cross_section_type=CrossSectionType.CIRCLE,
                max_volume=spec["totalLiquidVolume"],
            )
            well.location = Coordinate(
                x=spec["x"] - diameter / 2, y=spec["y"] - diameter / 2, z=spec["z"]
            )
            wells[item] = well
    dims = data["dimensions"]
    return Plate(
        name=name,
        size_x=dims["xDimension"],
        size_y=dims["yDimension"],
        size_z=dims["zDimension"],
        ordered_items=wells,
        model=data["metadata"]["displayName"],
    )


@dataclass
class Layout:
    deck: OTDeck
    tips: TipRack
    dye_rack: TubeRack
    dye_tubes: Dict[str, Tube]
    plate: Plate


def build_layout() -> Layout:
    """Create the deck with tips, dye tubes (red, yellow, blue in A1, B1, C1) and the plate."""
    with _vendored_definitions():
        tips = ot_load.load_ot_tip_rack(TIP_RACK, plr_resource_name="tips")
        dye_rack = ot_load.load_ot_tube_rack(TUBE_RACK, plr_resource_name="dyes")
    plate = _plate_from_definition("plate", PLATE)

    dye_tubes: Dict[str, Tube] = {}
    for dye, position in zip(DYES, ("A1", "B1", "C1")):
        holder = dye_rack.get_item(position)
        tube = Tube(
            name=f"{dye}_dye",
            size_x=holder.get_size_x(),
            size_y=holder.get_size_y(),
            size_z=holder.get_size_z(),
            max_volume=15_000.0,
        )
        dye_rack[position] = tube
        dye_tubes[dye] = tube

    deck = OTDeck()
    deck.assign_child_at_slot(tips, TIP_SLOT)
    deck.assign_child_at_slot(dye_rack, DYE_SLOT)
    deck.assign_child_at_slot(plate, PLATE_SLOT)
    return Layout(deck=deck, tips=tips, dye_rack=dye_rack, dye_tubes=dye_tubes, plate=plate)
