import torch
import torch.nn as nn
import torch.nn.functional as F

from addict import Dict

# Import from Panda_repo
import sys
import os

# Add Panda_repo to sys.path
# Assuming the script is located at src/collider_fm/model.py
# and Panda_repo is at the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, "Panda_repo"))

# Add src to sys.path to allow imports from collider_fm
sys.path.append(os.path.join(project_root, "src"))
from panda.model_base import PointTransformerV3
from panda.structure import Point
from panda.module import PointModule, PointSequential


class PandaSelfDistillation(nn.Module):
    """
    Panda Model for Self-distillation on ColliderML.
    Uses PointTransformerV3 as the backbone.
    """

    def __init__(
        self,
        in_channels=6,
        embed_channels=64,
        num_prototypes=1024,
        projection_dim=256,
        prediction_dim=256,
        temp_student=0.1,
        temp_teacher=0.04,
        center_momentum=0.9,
        **backbone_kwargs,
    ):
        super().__init__()

        # Backbone (Student and Teacher)
        # Standard Panda config for PTv3
        default_backbone_kwargs = dict(
            in_channels=in_channels,
            order=("z", "z-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(embed_channels, embed_channels * 2, embed_channels * 4, embed_channels * 8, embed_channels * 16),
            enc_num_head=(embed_channels // 16, embed_channels // 8, embed_channels // 4, embed_channels // 2, embed_channels),
            enc_patch_size=(48, 48, 48, 48, 48),
            enc_mode=True,
            enable_flash=False,  # Disable flash attention if not available
        )
        default_backbone_kwargs.update(backbone_kwargs)

        self.student_backbone = PointTransformerV3(**default_backbone_kwargs)
        self.teacher_backbone = PointTransformerV3(**default_backbone_kwargs)

        # Output dimension of backbone is enc_channels[-1]
        backbone_dim = default_backbone_kwargs["enc_channels"][-1]

        # Projection Head (shared between student and teacher)
        self.student_head = self._build_head(backbone_dim, projection_dim, prediction_dim)
        self.teacher_head = self._build_head(backbone_dim, projection_dim, prediction_dim, use_prediction=False)

        # Prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, projection_dim))

        # Initialize teacher with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False

        # Self-distillation parameters
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, projection_dim))

    def _build_head(self, in_dim, proj_dim, pred_dim, use_prediction=True):
        layers = [
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        ]
        if use_prediction:
            layers.extend(
                [
                    nn.BatchNorm1d(proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, pred_dim),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, views):
        """
        views: List of data_dicts, each representing a different view of the same batch of events.
        """
        # In multi-view distillation, we usually have multiple views (global, local, masked)
        # Student sees all views, Teacher sees only global/unmasked views

        student_outputs = []
        for view in views:
            # Student backbone
            out = self.student_backbone(view)
            # Global pooling or use per-point features for local consistency
            # Panda paper emphasizes sensor-level (per-point) features
            feat = out.feat
            # Apply student head
            proj = self.student_head(feat)
            # Similarity with prototypes
            proj = F.normalize(proj, dim=-1)
            logits = torch.mm(proj, self.prototypes.t())
            student_outputs.append(logits)

        # Teacher outputs (only for global views, usually the first two)
        with torch.no_grad():
            teacher_outputs = []
            for view in views[:2]:  # Assuming first two are global views
                out = self.teacher_backbone(view)
                feat = out.feat
                proj = self.teacher_head(feat)
                proj = F.normalize(proj, dim=-1)
                logits = torch.mm(proj, self.prototypes.t())
                teacher_outputs.append(logits)

        return student_outputs, teacher_outputs

    @torch.no_grad()
    def update_teacher(self, m):
        """
        Momentum update of the teacher backbone and head.
        m: momentum coefficient
        """
        for param_s, param_t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            # Note: teacher_head might be shorter if it doesn't have prediction layers
            if param_t.shape == param_s.shape:
                param_t.data.mul_(m).add_((1 - m) * param_s.data)


def panda_loss(student_logits, teacher_logits, center, temp_s, temp_t):
    """
    Cross-entropy loss between student and teacher distributions.
    """
    # Teacher distributions (centered and sharpened)
    teacher_probs = F.softmax((teacher_logits - center) / temp_t, dim=-1)

    # Student distributions (log-softmax)
    student_log_probs = F.log_softmax(student_logits / temp_s, dim=-1)

    # Loss: cross-entropy
    loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
    return loss


if __name__ == "__main__":
    # Mock test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing PandaSelfDistillation model...")
    model = PandaSelfDistillation(in_channels=8).to(device)
    print("Model initialized successfully.")

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    # Mock forward pass if GPU is available
    if torch.cuda.is_available():
        print("Running mock forward pass on GPU...")
        N = 1000  # Number of points
        data_dict = Dict(
            feat=torch.randn(N, 8).cuda(),
            coord=torch.rand(N, 3).cuda() * 100,
            grid_size=torch.tensor([0.1]).cuda(),
            offset=torch.tensor([N]).cuda(),
            batch=torch.zeros(N, dtype=torch.long).cuda()
        )
        # PointTransformerV3 expects some specific structure
        # We wrap it in a list as forward expects views
        try:
            student_logits, teacher_logits = model([data_dict, data_dict])
            print(f"Forward pass successful. Student logits shape: {student_logits[0].shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

