"""Pose of a COMPLETE regular ring; phase and yaw are intentionally unobservable."""
import cv2
import numpy as np

from auto_cam_level.common.geometry import complete_array, ring_points, normal_error, MeasurementError


class PoseFeatures:
    def __init__(self, camera, count=8, radius_mm=40., method="pnp",
                 max_reprojection_px=2., ambiguity_px=.1, ambiguity_deg=2.):
        if method not in ("pnp", "homography"):
            raise ValueError("pose method must be pnp or homography")
        self.camera, self.count, self.method = camera, count, method
        self.model = ring_points(count, radius_mm)
        self.max_reprojection_px = float(max_reprojection_px)
        self.ambiguity_px, self.ambiguity_deg = float(ambiguity_px), float(ambiguity_deg)
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.max_reprojection_px, self.ambiguity_px, self.ambiguity_deg)):
            raise ValueError("pose gates must be finite and positive")

    def __call__(self, points, diameters=None):
        pixels, xy = complete_array(points, self.count, self.camera)
        # Convex cyclic order is invariant to perspective. A cyclic shift only
        # changes the arbitrary in-plane phase, never the estimated normal.
        centered = xy-xy.mean(axis=0)
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        order = np.argsort(angles)
        # Keep the arbitrary model phase near image +X. IPPE's conversion to
        # Rodrigues vectors is unstable at an exact pi in-plane rotation in
        # some OpenCV versions, including a perfectly frontoparallel ring.
        order = np.roll(order, -int(np.argmin(np.abs(angles[order]))))
        pixels, xy = pixels[order], xy[order]
        candidates = []
        if self.method == "pnp":
            result = cv2.solvePnPGeneric(self.model, pixels, self.camera.K, self.camera.D,
                                        flags=cv2.SOLVEPNP_IPPE)
            if not result[0]:
                raise MeasurementError("IPPE returned no pose")
            for rv, tv in zip(result[1], result[2]):
                # IPPE supplies both planar candidates; refine EACH against all
                # points before comparing them. This matters near a frontal view.
                rv, tv = cv2.solvePnPRefineLM(self.model, pixels, self.camera.K,
                                            self.camera.D, rv.copy(), tv.copy())
                candidates.append((cv2.Rodrigues(rv)[0], tv.reshape(3)))
        else:
            # H is fitted on UNDISTORTED normalized coordinates, so no K^-1
            # or lens-distortion correction is applied to it a second time.
            H, mask = cv2.findHomography(self.model[:, :2], xy, cv2.RANSAC,
                self.max_reprojection_px/max(self.camera.K[0, 0], self.camera.K[1, 1]))
            if H is None or mask is None or not mask.all():
                raise MeasurementError("homography rejected a ring correspondence")
            scale = 2/(np.linalg.norm(H[:, 0])+np.linalg.norm(H[:, 1]))
            if H[2, 2] < 0:
                scale = -scale
            a, b, t = scale*H[:, 0], scale*H[:, 1], scale*H[:, 2]
            U, _, Vt = np.linalg.svd(np.column_stack((a, b, np.cross(a, b))))
            R = U@np.diag([1., 1., np.linalg.det(U@Vt)])@Vt
            candidates.append((R, t))
        accepted = []
        for R, t in candidates:
            if not np.isfinite(np.r_[R.ravel(), t]).all():
                continue
            depth = (self.model@R.T+t)[:, 2]
            if np.min(depth) <= 0:
                continue
            projected = cv2.projectPoints(self.model, cv2.Rodrigues(R)[0], t,
                                         self.camera.K, self.camera.D)[0].reshape(-1, 2)
            residuals = np.linalg.norm(projected-pixels, axis=1)
            error, normal, angle = normal_error(R[:, 2])
            if residuals.max() <= self.max_reprojection_px and angle < np.deg2rad(60):
                accepted.append((float(np.sqrt(np.mean(residuals**2))), error, normal, angle, R, t))
        accepted.sort(key=lambda c: c[0])
        if not accepted:
            raise MeasurementError("pose failed depth/reprojection/tilt gates")
        best = accepted[0]
        for other in accepted[1:]:
            separation = np.rad2deg(np.arccos(np.clip(best[2]@other[2], -1, 1)))
            if other[0]-best[0] < self.ambiguity_px and separation > self.ambiguity_deg:
                raise MeasurementError("ambiguous planar PnP normals; improve view or use homography")
        return best[1], dict(normal=best[2].tolist(), tilt_deg=float(np.rad2deg(best[3])),
            reprojection_rms_px=best[0], R_CP=best[4].tolist(), t_CP_mm=best[5].tolist(),
            method=self.method, phase_is_arbitrary=True)
