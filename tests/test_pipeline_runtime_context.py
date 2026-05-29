from __future__ import annotations

import sys
import types

fake = types.ModuleType("iduconfig")


class _Config:
    def get(self, *_args, **_kwargs):
        return "1"


fake.Config = _Config
sys.modules.setdefault("iduconfig", fake)

from pipeline_modules.business import common, matching_layer
from pipeline_modules.business.pipeline_impl import _init_runtime_context


def test_runtime_context_isolated_between_runs() -> None:
    common.ENABLE_EMBED_FAST_MATCH = False
    zone_templates_1 = [{"zone_code": "A1", "base_zone_code": "A", "zone_name": "Zone A", "zone_summary": "summary a", "retrieval_text_short": "short a", "main": [{"vri_code": "1.1", "vri_name": "Жилье"}], "conditional": [], "auxiliary": []}]
    zone_templates_2 = [{"zone_code": "B1", "base_zone_code": "B", "zone_name": "Zone B", "zone_summary": "summary b", "retrieval_text_short": "short b", "main": [{"vri_code": "2.1", "vri_name": "Магазины"}], "conditional": [], "auxiliary": []}]
    classifier = {"1.1": {"name": "Жилье"}, "2.1": {"name": "Магазины"}}

    ctx1 = _init_runtime_context(zone_templates_1, classifier)
    ctx2 = _init_runtime_context(zone_templates_2, classifier)

    assert "A1" in ctx1.zone_lookup and "B1" not in ctx1.zone_lookup
    assert "B1" in ctx2.zone_lookup and "A1" not in ctx2.zone_lookup
    assert getattr(matching_layer, "zone_lookup", {}) == {}
