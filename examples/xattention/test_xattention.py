import runpy
import sys
from pathlib import Path


# The example's guard is 162 lines that allocate two dozen scratch buffers, run
# the kernel over them and assert the merged output against
# ref_x_attention_decode. Restating that here would be copying the allocation,
# not testing it, so the module is executed under its own name and the
# AssertionError torch.testing raises is what fails the test.
def test_xattention() -> None:
    source = Path(__file__).with_name("xattention.py")
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
