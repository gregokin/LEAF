"""Diagnostic only: test missing CUDA header hypothesis in memory; NOT candidate qualification."""
import pytest
from maple_syrup import subsurface_transpiration as st
assert '#include <math.h>' not in st._CUDA_SRC
st._CUDA_SRC = '#include <math.h>\n' + st._CUDA_SRC
print('DIAGNOSTIC: CUDA source prefixed in memory with math.h; files remain unchanged',flush=True)
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null','tests','native/test_transpiration.py','-k','gpu or cuda']))
