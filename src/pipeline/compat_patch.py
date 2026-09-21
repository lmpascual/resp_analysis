"""
Windows OpenBLAS / SciPy / Matplotlib Compatibility Patch Module

Ensures safe execution on Windows systems by patching operations that trigger
OpenBLAS crashes (sgemm, solve, lstsq) using robust PyTorch CPU kernels.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch

try:
    import threadpoolctl
    for cls in [threadpoolctl.OpenBLASController, threadpoolctl.MKLController]:
        if hasattr(cls, 'get_num_threads'): cls.get_num_threads = lambda self: 1
        if hasattr(cls, 'set_num_threads'): cls.set_num_threads = lambda self, n: None
except Exception:
    pass

orig_np_solve = np.linalg.solve
def safe_solve(a, b):
    try:
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        return torch.linalg.solve(t_a, t_b).numpy()
    except Exception:
        return orig_np_solve(a, b)
np.linalg.solve = safe_solve

orig_np_norm = np.linalg.norm
def safe_norm(x, ord=None, axis=None, keepdims=False):
    try:
        t = torch.as_tensor(x)
        if axis is None and not keepdims:
            return float(torch.linalg.norm(t, ord=ord))
        return torch.linalg.norm(t, ord=ord, dim=axis, keepdim=keepdims).numpy()
    except Exception:
        return orig_np_norm(x, ord=ord, axis=axis, keepdims=keepdims)
np.linalg.norm = safe_norm

orig_np_lstsq = np.linalg.lstsq
def safe_lstsq(a, b, rcond=None):
    try:
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        res = torch.linalg.lstsq(t_a, t_b)
        x = res.solution.numpy()
        residuals = res.residuals.numpy() if res.residuals.numel() > 0 else np.array([])
        rank = int(res.rank)
        s = res.singular_values.numpy()
        return x, residuals, rank, s
    except Exception:
        return orig_np_lstsq(a, b, rcond=rcond)

np.linalg.lstsq = safe_lstsq
try:
    import numpy.linalg as nla
    nla.lstsq = safe_lstsq
    import numpy.lib._polynomial_impl as npi
    npi.lstsq = safe_lstsq
except Exception:
    pass

orig_np_inv = np.linalg.inv
def safe_inv(a):
    try:
        t_a = torch.as_tensor(a, dtype=torch.float64)
        return torch.linalg.inv(t_a).numpy()
    except Exception:
        return orig_np_inv(a)

np.linalg.inv = safe_inv
try:
    import numpy.linalg as nla
    nla.inv = safe_inv
    import numpy.linalg._linalg as nla_mod
    nla_mod.inv = safe_inv
except Exception:
    pass

orig_dot = np.dot
def safe_dot(a, b, out=None):
    try:
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        res = (t_a @ t_b).numpy()
        if out is not None:
            out[...] = res
            return out
        return res
    except Exception:
        return orig_dot(a, b, out=out)

np.dot = safe_dot

try:
    import matplotlib.transforms as mtransforms
    mtransforms.np.dot = safe_dot
    mtransforms.inv = safe_inv
except Exception:
    pass

try:
    import matplotlib.bezier as mb
    def safe_bezier_call(self, t):
        t = np.asarray(t)
        M = (np.power.outer(1 - t, self._orders[::-1]) * np.power.outer(t, self._orders))
        t_M = torch.as_tensor(M, dtype=torch.float64)
        t_px = torch.as_tensor(self._px, dtype=torch.float64)
        return (t_M @ t_px).numpy()
    mb.BezierSegment.__call__ = safe_bezier_call
except Exception:
    pass


