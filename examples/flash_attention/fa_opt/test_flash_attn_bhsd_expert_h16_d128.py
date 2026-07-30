import runpy
import sys
from pathlib import Path


# The example keeps both its driver and ref_flash_attn inside the main guard,
# reachable only by running the module, so this executes it under its own name
# and lets the AssertionError torch.testing raises fail the test. argv carries
# no arguments, which leaves argparse on the defaults the example documents and
# the reference check enabled.
def test_flash_attn_bhsd_expert_h16_d128() -> None:
    source = Path(__file__).with_name("flash_attn_bhsd_expert_h16_d128.py")
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
