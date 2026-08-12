"""The brain: a picture (and optionally how the robot feels) goes in, the next
second of robot movement comes out.

It is deliberately small enough to read in one sitting:

    picture  ->  ResNet-18  ->  grid of features  ->  flatten  ->  512 numbers
                                                                      |
    state    ->  one linear layer  ->  64 numbers  --------------- joined
    (joints + gripper)                                                |
                                                              chunk x 7 numbers

ResNet-18 is not ours - it was trained by someone else on a million photographs
of cats, boats and mushrooms. It already knows about edges, corners, textures and
shapes, and we are only teaching it what those things mean for OUR table. That is
called transfer learning, and it is why 132 demonstrations is enough. From
scratch you would need thousands.

One detail worth arguing about: we FLATTEN the last grid instead of averaging it.
Averaging would tell us what is in the picture; flattening keeps *where* it is.
For "reach toward the cube", where is the whole question.

WHY THE STATE GOES IN THROUGH ITS OWN LITTLE LAYER
The picture turns into 512 numbers. If we simply glued 8 raw state numbers onto
the end of those, they would be 8 voices in a crowd of 512, and the network would
mostly ignore them. Giving the state its own layer, widening it to 64, means both
streams arrive at the decision with a comparable say. It also lets the network
learn what a joint angle *means* before having to combine it with anything.

Set `state_size=0` and the whole state branch disappears; the policy is then
exactly the picture-only one.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

from . import vision


class Policy(nn.Module):
    """Picture in, chunk of actions out."""

    def __init__(
        self,
        action_size=7,
        chunk=10,
        action_mean=None,
        action_std=None,
        state_size=0,
        state_mean=None,
        state_std=None,
        image_size=(vision.IMAGE_H, vision.IMAGE_W),
        pretrained=True,
    ):
        super().__init__()
        self.action_size = int(action_size)
        self.chunk = int(chunk)
        self.state_size = int(state_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))

        # --- eyes: ResNet-18 with its final classifier chopped off ------------
        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # no internet on first run, usually
            print(f"[policy] could not fetch pretrained weights ({exc}); starting from scratch.")
            print("[policy] training will be noticeably worse - ask for help getting the weights.")
            backbone = torchvision.models.resnet18(weights=None)
        # children()[:-2] drops the global average pool and the classifier, so we
        # keep the grid of features instead of one summary vector.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # How big is that grid? Easier (and safer) to ask than to work it out.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
            n_features = self.backbone(dummy).numel()

        # --- what the picture is worth ----------------------------------------
        self.image_trunk = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(),
        )

        # --- what the robot feels about itself (optional) ---------------------
        joined = 512
        if self.state_size:
            self.state_trunk = nn.Sequential(nn.Linear(self.state_size, 64), nn.ReLU())
            joined += 64
        else:
            self.state_trunk = None

        # --- brain: turn all of it into a plan --------------------------------
        self.head = nn.Linear(joined, self.chunk * self.action_size)

        # --- units: how to translate between "real" and "network" numbers -----
        # Stored INSIDE the model (as buffers) so that saving the policy saves
        # the units too. A policy that forgets its units is a policy that drives
        # the robot into the table.
        if action_mean is None:
            action_mean = np.zeros(self.action_size, dtype=np.float32)
        if action_std is None:
            action_std = np.ones(self.action_size, dtype=np.float32)
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32))

        # A policy that does not use the state must not carry units for it, or
        # saving and loading disagree about how big those units are.
        if state_mean is None or not self.state_size:
            state_mean = np.zeros(self.state_size, dtype=np.float32)
        if state_std is None or not self.state_size:
            state_std = np.ones(self.state_size, dtype=np.float32)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))

    # ------------------------------------------------------------------ units

    def normalize_actions(self, actions):
        """Real units -> network units. Use this on your training targets."""
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions):
        """Network units -> real units. Use this before touching the robot."""
        return actions * self.action_std + self.action_mean

    # --------------------------------------------------------------- thinking

    def normalize_state(self, state):
        """Real units -> network units, for what the robot feels."""
        return (state - self.state_mean) / self.state_std

    def forward(self, images, state=None):
        """images: (B, 3, H, W), state: (B, state_size) or None
        -> (B, chunk, action_size), in NETWORK units."""
        features = self.image_trunk(self.backbone(images))
        if self.state_size:
            if state is None:
                raise ValueError(
                    f"this policy also feels {self.state_size} numbers about the robot; "
                    f"pass them as the second argument"
                )
            features = torch.cat([features, self.state_trunk(self.normalize_state(state))], dim=1)
        flat = self.head(features)
        return flat.view(-1, self.chunk, self.action_size)

    @property
    def device(self):
        return self.action_mean.device

    @torch.no_grad()
    def predict(self, image, state=None):
        """One camera picture (H, W, 3 uint8), plus how the robot feels if this
        policy uses that -> (chunk, action_size) in REAL units.

        This is the only method the robot script needs.
        """
        was_training = self.training
        self.eval()
        batch = vision.to_batch(image, device=self.device)
        state_batch = None
        if self.state_size:
            if state is None:
                raise ValueError(
                    f"this policy also feels {self.state_size} numbers about the robot "
                    f"(joint angles and the gripper); pass them as the second argument"
                )
            state_batch = torch.as_tensor(
                np.asarray(state, dtype=np.float32), device=self.device
            ).reshape(1, -1)
            if state_batch.shape[1] != self.state_size:
                raise ValueError(
                    f"this policy feels {self.state_size} numbers, got {state_batch.shape[1]}"
                )
        out = self.unnormalize_actions(self(batch, state_batch)[0])
        if was_training:
            self.train()
        return out.cpu().numpy()

    # ------------------------------------------------------------------- disk

    def save(self, path):
        """Write the whole policy - weights, shape and units - to one file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "action_size": self.action_size,
                "chunk": self.chunk,
                "state_size": self.state_size,
                "image_size": self.image_size,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        """Read a policy back. No configuration needed - it is all in the file."""
        blob = torch.load(path, map_location=device, weights_only=False)
        policy = cls(
            action_size=blob["action_size"],
            chunk=blob["chunk"],
            state_size=blob.get("state_size", 0),  # older files had no state branch
            image_size=blob["image_size"],
            pretrained=False,  # about to be overwritten anyway
        )
        policy.load_state_dict(blob["state_dict"])
        policy.to(device)
        policy.eval()
        return policy


def pick_device():
    """The GPU if there is one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"
