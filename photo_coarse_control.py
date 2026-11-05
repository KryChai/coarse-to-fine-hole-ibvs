"""Pure exponential gain for the photo-height nearest-hole stage; no mm cap."""
import math
import numpy as np


def exponential_gain(distance, tolerance=5., minimum=.03, maximum=.15, scale=200.):
    if not all(math.isfinite(v) for v in (distance,tolerance,minimum,maximum,scale)):
        raise ValueError("non-finite photo coarse settings")
    if distance < 0 or tolerance <= 0 or not 0 < minimum <= maximum < 1 or scale <= 0:
        raise ValueError("invalid photo coarse settings")
    return minimum+(maximum-minimum)*(-math.expm1(-max(0.,distance-tolerance)/scale))


def photo_step(error, jacobian, **settings):
    error=np.asarray(error,dtype=float)
    matrix=np.asarray(jacobian,dtype=float)
    if error.shape!=(2,) or not np.isfinite(error).all():
        raise ValueError("invalid image error")
    if matrix.shape!=(2,2) or not np.isfinite(matrix).all() or np.linalg.cond(matrix)>1e6:
        raise ValueError("invalid photo-height Jacobian")
    gain=exponential_gain(float(np.linalg.norm(error)),**settings)
    delta=-gain*np.linalg.solve(matrix,error)
    if not np.isfinite(delta).all():
        raise ValueError("invalid photo displacement")
    return gain,delta
