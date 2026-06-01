import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""
    def __init__(self, encoder_dim: List[int]):
        """encoder_dim: list of hidden sizes, ending with the size of the latent representation."""
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1

        # Encoder logic
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        # Decoder logic
        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if i < self._dim - 1:
                decoder_layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sample features.
        x: float tensor [num, feat_dim]
        latent: float tensor, representation Z, [n_nodes, latent_dim]
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode sample features.
        latent: float tensor, representation Z, [n_nodes, latent_dim]
        x: float tensor, reconstruction x, [num, feat_dim]
        """
        x_hat = self._decoder(latent)
        x_hat = F.normalize(x_hat, p=2, dim=1)  # l2 norm
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through autoencoder"""
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self, prediction_dim: List[int]):
        """prediction_dim: list of hidden sizes, ending with the size of the lat. repr. of autoencoder."""
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._prediction_dim = prediction_dim

        # Encoder logic
        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            encoder_layers.append(nn.ReLU())
        self._encoder = nn.Sequential(*encoder_layers)

        # Decoder logic
        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Data recovery by prediction.
        x: float tensor [num, feat_dim]
        latent: float tensor, representation Z, [n_nodes, latent_dim]
        output: float tensor, recovered data, [n_nodes, feat_dim]
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class DCP_Net(nn.Module):
    """
    Main Dual Contrastive Prediction model.
    Dynamically builds Autoencoders and Predictors for N views.
    """
    def __init__(self, view_dims: Dict[int, int], latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        
        # Build an Autoencoder for every view
        for v, dim in view_dims.items():
            ae_dims = [dim, hidden_dim, latent_dim]
            self.encoders[str(v)] = Autoencoder(encoder_dim=ae_dims)
            
        # Build cross-view predictors (translates Z_i -> Z_j)
        # Translator between different latent spaces.
        for i in view_dims.keys():
            for j in view_dims.keys():
                if i != j:
                    pred_dims = [latent_dim, latent_dim * 2] 
                    self.predictors[f"{i}_to_{j}"] = Prediction(prediction_dim=pred_dims)


def dcp_loss(
        model: DCP_Net, batch_dict: Dict[str, torch.Tensor],
        sn_mask: Dict[str, torch.Tensor], temperature: float = 0.5
) -> torch.Tensor:
    """
    Calculates the unsupervised loss for pre-training the DCP model.
    Original code just throws incomplete samples away, but I don't.
    sn_mask: The sample/view availability matrix. 1 is available, 0 not available.
    """
    latents= {}
    loss_recon, loss_pred, loss_contra = 0.0, 0.0, 0.0
    views = list(batch_dict.keys())
    
    # Reconstruction loss (masked like in CPM nets)
    for v_key, x in batch_dict.items():
        v_idx = v_key.split('_')[1]  # Only get view number
        
        x_hat, z = model.encoders[v_idx](x)
        latents[v_idx] = z
        
        # Calculate raw MSE per sample
        # Reduction 'none' makes that for a missing view, reconstruction error becomes 0.0
        raw_mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
        # Multiply by the availability mask, then average
        loss_recon += (raw_mse * sn_mask[v_idx]).mean()

    views = list(latents.keys())
        
    # Cross-prediction and contrastive loss (masked like in CPM nets)
    for i in views:
        for j in views:
            if i != j:
                # We can only predict/contrast if both views are present!
                # If either is missing, the pair is invalid.
                valid_pair_mask = sn_mask[i] * sn_mask[j]
                
                # Predict Z_j from Z_i
                z_j_pred, _ = model.predictors[f"{i}_to_{j}"](latents[i])
                raw_pred_mse = F.mse_loss(z_j_pred, latents[j], reduction='none').mean(dim=1)
                loss_pred += (raw_pred_mse * valid_pair_mask).mean()
                
                # Instance contrastive loss (InfoNCE)
                z_i_norm = F.normalize(latents[i], dim=1)
                z_j_norm = F.normalize(latents[j], dim=1)
                
                logits = torch.matmul(z_i_norm, z_j_norm.T) / temperature
                labels = torch.arange(z_i_norm.size(0), device=z_i_norm.device)
                
                # Calculate cross entropy per sample
                loss_i2j = F.cross_entropy(logits, labels, reduction='none')
                loss_j2i = F.cross_entropy(logits.T, labels, reduction='none')
                
                # Mask the contrastive loss
                valid_contra = ((loss_i2j + loss_j2i) / 2.0) * valid_pair_mask
                loss_contra += valid_contra.mean()

    num_pairs = len(views) * (len(views) - 1)
    
    loss_recon /= len(views)
    loss_pred /= num_pairs
    loss_contra /= num_pairs

    # Return the total loss
    return loss_recon + loss_pred + loss_contra
