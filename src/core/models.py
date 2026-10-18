import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

"""
Contains neural network backbones, which are all PyTorch modules.
"""

def build_extractor(in_dim: int, hidden_dim: int, nonlinear: bool = False) -> nn.Module:
    """Per-view projection into hidden_dim."""
    if nonlinear:
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
    return nn.Linear(in_dim, hidden_dim)

def build_backbone(hidden_dim: int, depth: int = 0) -> nn.Module:
    """
    Post-fusion (or per-view, for late fusion) block.
    depth=0: No additional layers (Identity).
    depth>0: Stacks of (Linear -> ReLU). depth=1 produces the W_g -> ReLU block for Mid fusion.
    """
    if depth == 0:
        return nn.Identity()
    
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    return nn.Sequential(*layers)

class SplitNetwork(nn.Module):
    """
    Single head baseline, architecturally aligned with MultiTaskModel.
    Supported fusion_types: "early", "mid", "late"
    """
    def __init__(self, sensor_dims: Dict[int, int], hidden_dim: int,
                 num_classes: int, fusion_type: str = "mid",
                 extractor_nonlinear: bool = False, backbone_depth: int = None):
        super().__init__()
        self.sensor_dims = sensor_dims
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        if self.fusion_type not in ["early", "mid", "late"]:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}. 'mid_late' has been retired.")

        # Default depths to match the schematics exactly
        if backbone_depth is None:
            backbone_depth = 1 if fusion_type == "mid" else 0

        # 1) Extraction layer
        if self.fusion_type == "early":
            total_dim = sum(sensor_dims.values())
            self.extractors = build_extractor(total_dim, hidden_dim, extractor_nonlinear)
        else:
            self.extractors = nn.ModuleDict({
                f"view_{v}": build_extractor(dim, hidden_dim, extractor_nonlinear)
                for v, dim in sensor_dims.items()
            })

        # 2) Shared backbone
        self.backbone = build_backbone(hidden_dim, backbone_depth)

        # 3) Output heads
        if self.fusion_type == "late":
            self.classifier = nn.ModuleDict({
                f"view_{v}": nn.Linear(hidden_dim, num_classes)
                for v in sensor_dims.keys()
            })
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs_dict: Dict[str, Tensor]) -> Tensor:
        if self.fusion_type == "early":
            # Concat -> Extract (Wg) -> ReLU -> Backbone -> Classify
            concat = torch.cat(
                [inputs_dict[f"view_{v}"] for v in sorted(self.sensor_dims.keys())],
                dim=1
            )
            x = self.extractors(concat)
            x = torch.relu(x)
            x = self.backbone(x)
            return self.classifier(x)

        elif self.fusion_type == "late":
            # Extract (Wv) -> ReLU -> Backbone -> Classify -> Mean
            all_logits = []
            for v in sorted(self.sensor_dims.keys()):
                view_key = f"view_{v}"
                x = self.extractors[view_key](inputs_dict[view_key])
                x = torch.relu(x) 
                x = self.backbone(x)
                logits = self.classifier[view_key](x)
                all_logits.append(logits)
            return torch.stack(all_logits).mean(dim=0)

        elif self.fusion_type == "mid":
            # Extract (Wv) -> ReLU -> Sum -> Backbone (Wg -> ReLU) -> Classify
            embs = []
            for v in sorted(self.sensor_dims.keys()):
                x_v = self.extractors[f"view_{v}"](inputs_dict[f"view_{v}"])
                # Explicit ReLU *before* the sum to match the new schematic
                embs.append(torch.relu(x_v))
            
            combined = torch.stack(embs).sum(dim=0)
            x = self.backbone(combined)
            return self.classifier(x)


class MultiTaskModel(nn.Module):
    """
    Adapted implementation from 'A Good Data Augmentation Policy Is Not All You Need'.
    Multi-task model with two heads (primary and auxiliary).
    """
    def __init__(self, sensor_dims: Dict[int, int], hidden_dim: int,
                 num_classes: int, fusion_type: str = "mid",
                 extractor_nonlinear: bool = False, backbone_depth: int = None):
        super().__init__()
        self.sensor_dims = sensor_dims
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        if self.fusion_type not in ["early", "mid", "late"]:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}. 'mid_late' has been retired.")

        if backbone_depth is None:
            backbone_depth = 1 if fusion_type == "mid" else 0

        # 1) Extraction layer
        if self.fusion_type == "early":
            total_dim = sum(sensor_dims.values())
            self.extractors = build_extractor(total_dim, hidden_dim, extractor_nonlinear)
        else:
            self.extractors = nn.ModuleDict({
                f"view_{v}": build_extractor(dim, hidden_dim, extractor_nonlinear)
                for v, dim in sensor_dims.items()
            })

        # 2) Shared backbone
        self.backbone = build_backbone(hidden_dim, backbone_depth)

        # 3) Output heads
        if self.fusion_type == "late":
            self.classifier = nn.ModuleDict({
                f"view_{v}": nn.Linear(hidden_dim, num_classes)
                for v in sensor_dims.keys()
            })
            self.aux_classifier = nn.ModuleDict({
                f"view_{v}": nn.Linear(hidden_dim, num_classes)
                for v in sensor_dims.keys()
            })
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.aux_classifier = nn.Linear(hidden_dim, num_classes)
            
    def forward(self, inputs_dict: Dict[str, Tensor], use_auxiliary: bool = False) -> Tensor:
        if self.fusion_type == "early":
            # Concat -> Extract (Wg) -> ReLU -> Backbone -> Route
            concat = torch.cat(
                [inputs_dict[f"view_{v}"] for v in sorted(self.sensor_dims.keys())],
                dim=1
            )
            x = self.extractors(concat)
            x = torch.relu(x)
            x = self.backbone(x)
            return self.aux_classifier(x) if use_auxiliary else self.classifier(x)

        elif self.fusion_type == "late":
            # Extract (Wv) -> ReLU -> Backbone -> Route -> Mean
            all_logits = []
            for v in sorted(self.sensor_dims.keys()):
                view_key = f"view_{v}"
                x = self.extractors[view_key](inputs_dict[view_key])
                x = torch.relu(x)
                x = self.backbone(x)
                
                if use_auxiliary:
                    logits = self.aux_classifier[view_key](x)
                else:
                    logits = self.classifier[view_key](x)
                all_logits.append(logits)
            return torch.stack(all_logits).mean(dim=0)

        elif self.fusion_type == "mid":
            # Extract (Wv) -> ReLU -> Sum -> Backbone (Wg -> ReLU) -> Route
            embs = []
            for v in sorted(self.sensor_dims.keys()):
                x_v = self.extractors[f"view_{v}"](inputs_dict[f"view_{v}"])
                # Explicit ReLU *before* the sum to match the new schematic
                embs.append(torch.relu(x_v))
            
            combined = torch.stack(embs).sum(dim=0)
            x = self.backbone(combined)
            return self.aux_classifier(x) if use_auxiliary else self.classifier(x)
