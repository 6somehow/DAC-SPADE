import os
import sys
import types

import pytest

from tests.helpers.layouts import LAYOUTS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
UTILS_PATH = os.path.join(ROOT, "csrc", "spade_utils")

for path in (UTILS_PATH,):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import spade_utils  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("spade_utils")

    def _missing_spade_utils(*_args, **_kwargs):
        pytest.skip("spade_utils not built", allow_module_level=True)

    stub.scatter_mask = _missing_spade_utils
    stub.static_sink_diag_set = _missing_spade_utils
    stub.mask_to_bsr = _missing_spade_utils
    stub.cossim = _missing_spade_utils
    sys.modules["spade_utils"] = stub

try:
    import spade_hy  # noqa: F401
except ModuleNotFoundError:
    hy_stub = types.ModuleType("spade_hy")

    def _missing_spade_hy(*_args, **_kwargs):
        pytest.skip("spade_hy not built", allow_module_level=True)

    hy_stub.summarize_forward = _missing_spade_hy
    sys.modules["spade_hy"] = hy_stub


@pytest.fixture(params=LAYOUTS)
def layout(request):
    return request.param
