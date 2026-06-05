#!/usr/bin/env python3

import torch
import torch.nn as nn

from models.gonet import Generator, InvG, Discriminator


class GONetTemporalFeatureReducer(nn.Module):
    """
    Reduces GONet per-frame features into compact temporal features.

    Inputs per frame:
        img_error: [N, 3, 128, 128]
        dis_error: [N, 512, 8, 8]
        dis_real:  [N, 512, 8, 8]

    Output:
        features: [N, 30]

    This follows the paper's GONet+T idea:
        phi_R -> FC -> 10D
        phi_D -> FC -> 10D
        phi_F -> FC -> 10D
        concat -> 30D
    """

    def __init__(self, reduced_dim: int = 10):
        super().__init__()

        self.reduced_dim = reduced_dim

        self.img_fc = nn.Sequential(
            nn.Linear(3 * 128 * 128, reduced_dim),
            nn.ReLU(inplace=True),
        )

        self.dis_error_fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, reduced_dim),
            nn.ReLU(inplace=True),
        )

        self.dis_real_fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, reduced_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        img_error: torch.Tensor,
        dis_error: torch.Tensor,
        dis_real: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = torch.abs(img_error).reshape(img_error.size(0), -1)
        img_feat = self.img_fc(img_feat)

        dis_err_feat = torch.abs(dis_error).reshape(dis_error.size(0), -1)
        dis_err_feat = self.dis_error_fc(dis_err_feat)

        dis_real_feat = dis_real.reshape(dis_real.size(0), -1)
        dis_real_feat = self.dis_real_fc(dis_real_feat)

        features = torch.cat(
            [img_feat, dis_err_feat, dis_real_feat],
            dim=1,
        )

        return features


class GONetTemporalClassifier(nn.Module):
    """
    LSTM classifier for GONet+T.

    Input:
        temporal_features: [B, T, input_dim]
        lengths: [B]

    Output:
        probs: [B, T, 1]
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.output = nn.Linear(out_dim, 1)

    def forward(
        self,
        temporal_features: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        # Simple masked-training version.
        # We use padded sequences directly. The loss mask will ignore padding.
        # This is easier to debug than pack_padded_sequence.
        h, _ = self.lstm(temporal_features)
        logits = self.output(h)
        probs = torch.sigmoid(logits)

        return probs


class GONetTemporalFull(nn.Module):
    """
    Full GONet+T module.

    Contains:
        frozen generator
        frozen invg
        frozen discriminator
        trainable feature reducer
        trainable LSTM classifier

    Input:
        images: [B, T, 3, 128, 128]

    Output:
        probs: [B, T, 1]
    """

    def __init__(
        self,
        generator: Generator,
        invg: InvG,
        discriminator: Discriminator,
        feature_reducer: GONetTemporalFeatureReducer,
        temporal_classifier: GONetTemporalClassifier,
    ):
        super().__init__()

        self.generator = generator
        self.invg = invg
        self.discriminator = discriminator
        self.feature_reducer = feature_reducer
        self.temporal_classifier = temporal_classifier

    def freeze_gonet_backbone(self):
        self.generator.eval()
        self.invg.eval()
        self.discriminator.eval()

        for module in [self.generator, self.invg, self.discriminator]:
            for p in module.parameters():
                p.requires_grad = False

    def extract_temporal_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input:
            images: [B, T, 3, 128, 128]

        Output:
            temporal_features: [B, T, 30]
        """

        b, t, c, h, w = images.shape

        flat_images = images.reshape(b * t, c, h, w)

        with torch.no_grad():
            z_hat = self.invg(flat_images)
            img_gen = self.generator(z_hat)

            dis_real = self.discriminator(flat_images)
            dis_gen = self.discriminator(img_gen)

            img_error = flat_images - img_gen
            dis_error = dis_real - dis_gen

        features = self.feature_reducer(
            img_error=img_error,
            dis_error=dis_error,
            dis_real=dis_real,
        )

        temporal_features = features.reshape(b, t, -1)

        return temporal_features

    def forward(
        self,
        images: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        temporal_features = self.extract_temporal_features(images)
        probs = self.temporal_classifier(
            temporal_features=temporal_features,
            lengths=lengths,
        )
        return probs


def masked_prediction_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Prediction loss over valid frames only.

    preds:   [B, T, 1]
    targets: [B, T, 1]
    mask:    [B, T]
    """

    mask_f = mask.float().unsqueeze(-1)

    if loss_type == "mse":
        per_elem = (preds - targets) ** 2
    elif loss_type == "l1":
        per_elem = torch.abs(preds - targets)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = (per_elem * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    return loss


def masked_smoothness_loss(
    preds: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    Smoothness loss between consecutive valid predictions.

    preds: [B, T, 1]
    mask:  [B, T]

    Computes loss on pairs:
        t and t+1
    only when both are valid.
    """

    if preds.size(1) < 2:
        return preds.sum() * 0.0

    pred_prev = preds[:, :-1, :]
    pred_next = preds[:, 1:, :]

    pair_mask = mask[:, :-1] & mask[:, 1:]
    pair_mask_f = pair_mask.float().unsqueeze(-1)

    if pair_mask_f.sum() < 1:
        return preds.sum() * 0.0

    if loss_type == "l1":
        per_pair = torch.abs(pred_next - pred_prev)
    elif loss_type == "mse":
        per_pair = (pred_next - pred_prev) ** 2
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = (per_pair * pair_mask_f).sum() / pair_mask_f.sum().clamp_min(1.0)

    return loss


def init_temporal_weights(module: nn.Module, std: float = 0.02):
    classname = module.__class__.__name__

    if classname.find("Linear") != -1:
        nn.init.normal_(module.weight.data, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname.find("LSTM") != -1:
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from models.gonet import build_gonet_modules

    gen, invg, dis, _ = build_gonet_modules(nz=100)

    gen = gen.to(device)
    invg = invg.to(device)
    dis = dis.to(device)

    reducer = GONetTemporalFeatureReducer(reduced_dim=10).to(device)
    temporal = GONetTemporalClassifier(
        input_dim=30,
        hidden_dim=64,
        num_layers=1,
    ).to(device)

    model = GONetTemporalFull(
        generator=gen,
        invg=invg,
        discriminator=dis,
        feature_reducer=reducer,
        temporal_classifier=temporal,
    ).to(device)

    model.freeze_gonet_backbone()

    x = torch.randn(2, 11, 3, 128, 128).to(device)
    mask = torch.ones(2, 11, dtype=torch.bool).to(device)
    y = torch.rand(2, 11, 1).to(device)

    out = model(x)

    loss_pred = masked_prediction_loss(out, y, mask)
    loss_smooth = masked_smoothness_loss(out, mask)

    print("Device:", device)
    print("Input:", x.shape)
    print("Output:", out.shape)
    print("Prediction loss:", float(loss_pred.item()))
    print("Smoothness loss:", float(loss_smooth.item()))