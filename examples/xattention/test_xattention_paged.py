import runpy
import sys
from pathlib import Path


# As with the unpaged variant next to it, the guard allocates the scratch the
# kernel needs and asserts the merged output against ref_x_attention_decode, so
# the module is executed under its own name rather than having its allocation
# copied here.
def test_xattention_paged() -> None:
    source = Path(__file__).with_name("xattention_paged.py")
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
