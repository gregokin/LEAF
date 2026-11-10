"""Diagnostic only: test NVRTC built-in infinity hypothesis in memory; NOT qualification."""
import pytest
from maple_syrup import subsurface_transpiration as st
assert 'INFINITY' in st._CUDA_SRC
st._CUDA_SRC=st._CUDA_SRC.replace('INFINITY','__longlong_as_double(0x7ff0000000000000LL)')
print('DIAGNOSTIC: CUDA infinity replaced in memory with IEEE double intrinsic; files unchanged',flush=True)
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null','tests','native/test_transpiration.py','-k','gpu or cuda']))
