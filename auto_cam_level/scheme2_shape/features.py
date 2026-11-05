import numpy as np

from auto_cam_level.common.geometry import complete_array, finite, MeasurementError


class ShapeFeatures:
    def __init__(self, camera, count=8, enhanced=True, gamma=1., max_size_residual=.05):
        if type(count) is not int or count < 6:
            raise ValueError("shape scheme requires >=6 holes")
        if type(enhanced) is not bool:
            raise ValueError("enhanced must be boolean")
        if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(max_size_residual) or max_size_residual <= 0:
            raise ValueError("invalid size-gradient gates")
        self.camera, self.count, self.enhanced = camera, count, enhanced
        self.gamma, self.max_size_residual = gamma, max_size_residual

    def __call__(self, points, diameters=None):
        _, xy = complete_array(points, self.count, self.camera)
        centered = xy-xy.mean(axis=0)
        covariance = centered.T@centered/self.count
        trace = float(np.trace(covariance))
        s = np.array([(covariance[0, 0]-covariance[1, 1])/trace, 2*covariance[0, 1]/trace])
        values = dict(ellipse_features=s.tolist(), anisotropy=float(np.linalg.norm(s)),
                      coordinate_frame="undistorted_normalized", enhanced=self.enhanced)
        if self.enhanced:
            # Diameters are already in normalized image units (area-equivalent
            # ellipse diameters). BBox width is never used as a substitute.
            d = finite(diameters, (self.count,))
            if np.any(d <= 0):
                raise MeasurementError("enhanced shape requires all positive fitted diameters")
            radius = np.sqrt(trace)
            design = np.column_stack((np.ones(self.count), centered/radius))
            coeff = np.linalg.lstsq(design, np.log(d), rcond=None)[0]
            residual = np.log(d)-design@coeff
            if np.max(np.abs(residual)) > self.max_size_residual:
                raise MeasurementError("inconsistent hole sizes/gradient; check equal physical hole rims")
            s = np.r_[s, self.gamma*coeff[1:]]
            values.update(size_gradient=coeff[1:].tolist(), size_log_rms=float(np.sqrt(np.mean(residual**2))))
        return s, values
