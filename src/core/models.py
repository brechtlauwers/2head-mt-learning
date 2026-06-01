import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

"""
Contains neural network backbones, which are all PyTorch modules.
"""


class SplitNetwork(nn.Module):
    """
    Single head baseline, architecturally aligned with MultiTaskModel.
    Supported fusion_types: "early", "mid", "mid_late", "late"
    """
    def __init__(self, sensor_dims: Dict[int, int], hidden_dim: int,
                 num_classes: int, fusion_type: str = "mid_late"):
        super().__init__()
        self.sensor_dims = sensor_dims
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        # 1) Extraction layer
        if self.fusion_type == "early":
            # Sum of all sensor dims
            total_dim = sum(sensor_dims.values())
            self.extractors = nn.Linear(total_dim, hidden_dim)
        else:
            # View specific extractors
            self.extractors = nn.ModuleDict({
                f"view_{v}": nn.Linear(dim, hidden_dim)
                for v, dim in sensor_dims.items()
            })

        # 2) Shared backbone
        if self.fusion_type == "mid":
            self.backbone = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            # The other fusion methods just have parameter-free relu
            self.backbone = nn.ReLU()

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
            # Concat -> Extract -> Backbone -> Classify
            concat = torch.cat(
                [inputs_dict[f"view_{v}"] for v in sorted(self.sensor_dims.keys())],
                dim=1
            )
            x = self.extractors(concat)
            x = self.backbone(x)
            return self.classifier(x)

        elif self.fusion_type == "late":
            # Extract -> Backbone -> Classify -> Mean
            all_logits = []
            for v in sorted(self.sensor_dims.keys()):
                view_key = f"view_{v}"
                x = self.extractors[view_key](inputs_dict[view_key])
                x = self.backbone(x)
                logits = self.classifier[view_key](x)
                all_logits.append(logits)
            return torch.stack(all_logits).mean(dim=0)

        elif self.fusion_type in ["mid", "mid_late"]:
            # Extract -> Sum -> Backbone -> Classify
            combined = torch.stack([
                self.extractors[f"view_{v}"](inputs_dict[f"view_{v}"])
                for v in sorted(self.sensor_dims.keys())
            ]).sum(dim=0)
            x = self.backbone(combined)
            return self.classifier(x)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        


class MultiTaskModel(nn.Module):
    """
    Adapted implementation from 'A Good Data Augmentation Policy Is Not All You Need'.
    Multi-task model with two heads (primary and auxiliary).
    """
    def __init__(self, sensor_dims: Dict[int, int], hidden_dim: int,
                 num_classes: int, fusion_type: str = "mid_late"):
        super().__init__()
        self.sensor_dims = sensor_dims
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        # 1) Extraction layer
        if self.fusion_type == "early":
            # Sum of all sensor dims
            total_dim = sum(sensor_dims.values())
            self.extractors = nn.Linear(total_dim, hidden_dim)
        else:
            # View specific extractors
            self.extractors = nn.ModuleDict({
                f"view_{v}": nn.Linear(dim, hidden_dim)
                for v, dim in sensor_dims.items()
            })

        # 2) Shared backbone
        if self.fusion_type == "mid":
            self.backbone = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            # The other fusion methods just have parameter-free relu
            self.backbone = nn.ReLU()

        # 3) Output heads (primary and auxiliary)
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
            # Concat -> Extract -> Backbone -> Route
            concat = torch.cat(
                [inputs_dict[f"view_{v}"] for v in sorted(self.sensor_dims.keys())],
                dim=1
            )
            x = self.extractors(concat)
            x = self.backbone(x)
            return self.aux_classifier(x) if use_auxiliary else self.classifier(x)

        elif self.fusion_type == "late":
            # Extract -> Backbone -> Route -> Mean
            all_logits = []
            for v in sorted(self.sensor_dims.keys()):
                view_key = f"view_{v}"
                x = self.extractors[view_key](inputs_dict[view_key])
                x = self.backbone(x)
                
                if use_auxiliary:
                    logits = self.aux_classifier[view_key](x)
                else:
                    logits = self.classifier[view_key](x)
                all_logits.append(logits)
            return torch.stack(all_logits).mean(dim=0)

        elif self.fusion_type in ["mid", "mid_late"]:
            # Extract -> Sum -> Backbone -> Route
            combined = torch.stack([
                self.extractors[f"view_{v}"](inputs_dict[f"view_{v}"])
                for v in sorted(self.sensor_dims.keys())
            ]).sum(dim=0)
            x = self.backbone(combined)
            return self.aux_classifier(x) if use_auxiliary else self.classifier(x)
            
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
