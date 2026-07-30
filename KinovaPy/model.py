import os

import numpy as np
import pinocchio as pin

from KinovaPy import MESHES_PATH, URDF_PATH

DEFAULT_URDF_NAME = "GEN3-7DOF-VISION_ARM_URDF_V12.urdf"
DEFAULT_FRAME_NAMES = ["end_effector_link"]


class PinocchioModel:
    """Kinematics/dynamics model of the Kinova arm, backed by Pinocchio.

    Callers push a joint state in via `update(q, dq)` and read every derived
    quantity back out via `get_info()` as a dict of plain numpy arrays, so
    the rest of the codebase never has to import or hold onto pinocchio
    types directly.
    """

    def __init__(self, urdf_path=None, meshes_dir=None, frame_names=None, root_joint=None):
        urdf_path = urdf_path or os.path.join(URDF_PATH, DEFAULT_URDF_NAME)
        meshes_dir = meshes_dir or MESHES_PATH

        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, meshes_dir, root_joint=root_joint)
        self.rmodel = robot.model
        self.rdata = self.rmodel.createData()

        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nu = self.rmodel.nv

        self.frame_names = list(frame_names) if frame_names else list(DEFAULT_FRAME_NAMES)
        self.frame_ids = {name: self.rmodel.getFrameId(name) for name in self.frame_names}
        # First configured frame is treated as the primary end-effector.
        self.ee_frame_name = self.frame_names[0]
        self.ee_frame_id = self.frame_ids[self.ee_frame_name]

        self.q = np.zeros(self.nq)
        self.dq = np.zeros(self.nv)
        self._updated = False

    def update(self, q, dq):
        """Push a joint state through kinematics and dynamics.

        q: (nq,) joint positions [rad]
        dq: (nv,) joint velocities [rad/s]
        """
        q = np.asarray(q, dtype=float).reshape(self.nq)
        dq = np.asarray(dq, dtype=float).reshape(self.nv)

        pin.forwardKinematics(self.rmodel, self.rdata, q, dq)
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        pin.crba(self.rmodel, self.rdata, q)
        # crba only fills the upper triangle of the mass matrix.
        self.rdata.M = np.triu(self.rdata.M) + np.triu(self.rdata.M, 1).T
        pin.computeGeneralizedGravity(self.rmodel, self.rdata, q)
        pin.nonLinearEffects(self.rmodel, self.rdata, q, dq)

        self.q = q
        self.dq = dq
        self._updated = True

    def _require_updated(self):
        if not self._updated:
            raise RuntimeError("PinocchioModel.update(q, dq) must be called before querying the model.")

    def _frame_pose(self, frame_id):
        oMf = self.rdata.oMf[frame_id]
        rotation = np.array(oMf.rotation)
        return {
            "position": np.array(oMf.translation),
            "rotation": rotation,
            "rpy": np.array(pin.rpy.matrixToRpy(rotation)),
        }

    def _frame_jacobian(self, frame_id):
        return np.array(
            pin.getFrameJacobian(self.rmodel, self.rdata, frame_id, pin.LOCAL_WORLD_ALIGNED)
        )

    def get_info(self):
        """Return every derived quantity from the last `update()` as numpy arrays."""
        self._require_updated()

        frame_poses = {name: self._frame_pose(fid) for name, fid in self.frame_ids.items()}
        frame_jacobians = {name: self._frame_jacobian(fid) for name, fid in self.frame_ids.items()}

        return {
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "frame_poses": frame_poses,
            "frame_jacobians": frame_jacobians,
            "mass_matrix": np.array(self.rdata.M),
            "gravity": np.array(self.rdata.g),
            "nonlinear_effects": np.array(self.rdata.nle),
            "position_lower_limit": np.array(self.rmodel.lowerPositionLimit),
            "position_upper_limit": np.array(self.rmodel.upperPositionLimit),
            "velocity_limit": np.array(self.rmodel.velocityLimit),
            "effort_limit": np.array(self.rmodel.effortLimit),
        }
