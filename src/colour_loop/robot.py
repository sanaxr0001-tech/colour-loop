"""The simulated OT-2: PyLabRobot's ``LiquidHandler`` on an OT-2 deck with the simulator backend.

Nothing here talks to hardware. ``OpentronsOT2Simulator`` is PyLabRobot's own simulation backend
for the OT-2; it accepts every command and does the tip and volume bookkeeping, so wells fill and
tips disappear exactly as they would on a real run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Sequence

from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import OpentronsOT2Simulator
from pylabrobot.resources.tip_tracker import set_tip_tracking
from pylabrobot.resources.volume_tracker import set_volume_tracking

from .labware import DYE_TUBE_VOLUME_UL, Layout, build_layout
from .physics import DYES

WELL_TOTAL_UL = 180.0  # every well is made up to this volume (the plate holds 200 uL)
VOLUME_STEP_UL = 0.1  # the pipette's volume resolution we round to

Event = Callable[[dict], Awaitable[None] | None]


@dataclass
class WellFill:
    """What was dispensed into one well."""

    well: str
    volumes_ul: Dict[str, float]

    @property
    def fractions(self) -> Dict[str, float]:
        total = sum(self.volumes_ul.values())
        return {d: v / total for d, v in self.volumes_ul.items()}


def fractions_to_volumes(fractions: Sequence[float], total_ul: float = WELL_TOTAL_UL) -> List[float]:
    """Turn dye fractions into pipettable volumes that sum to exactly ``total_ul``.

    Each volume is rounded to the pipette resolution; the largest absorbs the rounding remainder.
    """
    volumes = [round(f * total_ul / VOLUME_STEP_UL) * VOLUME_STEP_UL for f in fractions]
    biggest = max(range(len(volumes)), key=volumes.__getitem__)
    volumes[biggest] = round(volumes[biggest] + (total_ul - sum(volumes)), 1)
    return [round(v, 1) for v in volumes]


@dataclass
class ColourRobot:
    """A simulated OT-2 that mixes dyes into wells of a 96-well plate."""

    step_delay: float = 0.0  # seconds to pause after each pipetting step, so a viewer can follow
    on_event: Event | None = None
    layout: Layout = field(init=False)
    lh: LiquidHandler = field(init=False)
    _next_well: int = field(init=False, default=0)
    _next_tip: int = field(init=False, default=0)
    ledger: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)

    async def setup(self) -> None:
        set_tip_tracking(True)
        set_volume_tracking(True)
        self.layout = build_layout()
        for tube in self.layout.dye_tubes.values():
            tube.set_volume(DYE_TUBE_VOLUME_UL)
        self.lh = LiquidHandler(
            backend=OpentronsOT2Simulator(left_pipette_name="p300_single_gen2", right_pipette_name=None),
            deck=self.layout.deck,
        )
        await self.lh.setup()

    async def stop(self) -> None:
        await self.lh.stop()

    async def _emit(self, **event) -> None:
        if self.on_event is not None:
            result = self.on_event(event)
            if asyncio.iscoroutine(result):
                await result
        if self.step_delay:
            await asyncio.sleep(self.step_delay)

    async def mix_wells(self, mixes: Sequence[Sequence[float]]) -> List[WellFill]:
        """Dispense one mix (dye fractions in ``DYES`` order) into each of the next free wells.

        One fresh tip per dye per call: the tip fetches that dye for every well, then is discarded.
        """
        plate_wells = self.layout.plate.get_all_items()
        # column-major fill (A1, B1, ... H1, A2, ...), which is the order PyLabRobot lists wells in
        targets = plate_wells[self._next_well : self._next_well + len(mixes)]
        if len(targets) < len(mixes):
            raise RuntimeError("the plate is full")
        self._next_well += len(mixes)

        plan = [fractions_to_volumes(m) for m in mixes]
        fills = []
        for well, vols in zip(targets, plan):
            key = self.layout.plate.get_child_identifier(well)
            self.ledger[key] = dict(zip(DYES, vols))
            fills.append(WellFill(well=key, volumes_ul=self.ledger[key]))

        tip_spots = self.layout.tips.get_all_items()
        for d_index, dye in enumerate(DYES):
            tip = tip_spots[self._next_tip]
            self._next_tip += 1
            await self.lh.pick_up_tips([tip])
            await self._emit(kind="tip", action="pick_up", dye=dye, tip=tip.name)
            for well, vols in zip(targets, plan):
                volume = vols[d_index]
                if volume <= 0:
                    continue
                key = self.layout.plate.get_child_identifier(well)
                await self.lh.aspirate([self.layout.dye_tubes[dye]], vols=[volume])
                await self.lh.dispense([well], vols=[volume])
                await self._emit(kind="dispense", dye=dye, well=key, volume_ul=volume)
            await self.lh.discard_tips()
            await self._emit(kind="tip", action="discard", dye=dye)
        return fills

    def well_volume_ul(self, well: str) -> float:
        """Total liquid PyLabRobot's volume tracker has in a well (cross-checks the ledger)."""
        return self.layout.plate.get_well(well).tracker.get_used_volume()
