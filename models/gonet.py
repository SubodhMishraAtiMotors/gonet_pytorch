#!/usr/bin/env python3

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    GONet/DCGAN-style generator.

    Input:
        z: [B, nz], B: Batch size, nz: latent dimension

    Output:
        image: [B, 3, 128, 128]: 128x128 RGB image
    """

    def __init__(self, nz: int = 100, use_tanh: bool = False):
        super().__init__()

        self.nz = nz
        self.use_tanh = use_tanh

        # Fully connected layer to project from latent space of 100 dimensions to a 8x8x512 dimensioned vector, 
        # which can then be reshaped into a 512-channel 8x8 feature map for the transposed convolutional layers.
        self.fc = nn.Sequential(
            nn.Linear(nz, 8 * 8 * 512),
            nn.BatchNorm1d(num_features = 8 * 8 * 512),
            nn.ReLU(inplace=True),
        )

        # Transposed convolutional layers to upsample from 8x8 to 128x128
        # out_size = (in_size - 1) * stride - 2 * padding + kernel_size (+ output_padding, optionally) 
        # Inplace ReLU is memory efficient, but should be avoided with skip connections. 
        # Here we don't have skip connections, so Inplace ReLU is fine.
        layers = [
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size=4, stride=2, padding=1),
        ]

        if use_tanh:
            layers.append(nn.Tanh())

        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.reshape(z.size(0), 512, 8, 8)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator from GONet.

    Input:
        image: [B, 3, 128, 128], B: Batch size, Input is a 128x128 RGB Image

    Output by default:
        features: [B, 512, 8, 8], B: Batch size, Output is a 512 channeled 8x8 feature map, but can return logits too.

    If return_logits=True:
        features, logits
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 128),
            nn.ELU(inplace=True),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 256),
            nn.ELU(inplace=True),

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 512),
            nn.ELU(inplace=True),
        )

        self.classifier = nn.Linear(in_features = 512 * 8 * 8, out_features = 2)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        h = self.features(x)

        if return_logits:
            logits = self.classifier(h.reshape(x.size(0), -1))
            return h, logits

        return h

class InvG(nn.Module):
    """
    Inverse generator.

    Maps an image back into the generator latent space.

    Input:
        image: [B, 3, 128, 128], B: Batch size, Input is a 128x128 RGB Image

    Output:
        z: [B, nz], B: Batch size, nz: latent dimension, Output is the latent vector corresponding to the input image
    """

    def __init__(self, nz: int = 100):
        super().__init__()

        self.nz = nz

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(in_features = 512 * 8 * 8, out_features = self.nz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.reshape(x.size(0), -1)
        z = self.fc(h)
        return z

class GONetClassifier(nn.Module):
    """
    Final GONet classification layer, equivalent to FL in the Chainer code.

    Inputs:
        img_error: [B, 3, 128, 128], B: Batch size, Input is the pixel-wise error between the real and generated image, calculated as
            img_real - img_gen

        dis_error: [B, 512, 8, 8], B: Batch size, Input is the feature-wise error between the real and generated image, calculated as
            dis_real - dis_gen

        dis_real: [B, 512, 8, 8], B: Batch size, Input is the discriminator features of the real image

    Output:
        prob: [B, 1], B: Batch size, Output is the traversability probability
    """

    def __init__(self):
        super().__init__()

        self.l_img = nn.Linear(in_features = 3 * 128 * 128, out_features = 1)
        self.l_dis = nn.Linear(in_features = 512 * 8 * 8, out_features = 1)
        self.l_fdis = nn.Linear(in_features = 512 * 8 * 8, out_features = 1)
        self.l_final = nn.Linear(in_features = 3, out_features = 1)

    def forward(
        self,
        img_error: torch.Tensor,
        dis_error: torch.Tensor,
        dis_real: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.abs(img_error).reshape(img_error.size(0), -1)
        h = self.l_img(h)

        g = torch.abs(dis_error).reshape(dis_error.size(0), -1)
        g = self.l_dis(g)

        f = dis_real.reshape(dis_real.size(0), -1)
        f = self.l_fdis(f)

        x = torch.cat([h, g, f], dim=1)
        prob = torch.sigmoid(self.l_final(x))

        return prob

class GONetFull(nn.Module):
    """
    Convenience wrapper for inference after all submodules are trained.

    It performs:

        z = invg(img_real)
        img_gen = gen(z)
        dis_real = dis(img_real)
        dis_gen = dis(img_gen)
        prob = fl(img_real - img_gen, dis_real - dis_gen, dis_real)
    """

    def __init__(
        self,
        generator: Generator,
        invg: InvG,
        discriminator: Discriminator,
        classifier: GONetClassifier,
    ):
        super().__init__()

        self.generator = generator
        self.invg = invg
        self.discriminator = discriminator
        self.classifier = classifier

    def forward(self, img_real: torch.Tensor):
        z = self.invg(img_real)
        img_gen = self.generator(z)

        dis_real = self.discriminator(img_real)
        dis_gen = self.discriminator(img_gen)

        prob = self.classifier(
            img_error=img_real - img_gen,
            dis_error=dis_real - dis_gen,
            dis_real=dis_real,
        )

        return {
            "prob": prob,
            "z": z,
            "img_gen": img_gen,
            "dis_real": dis_real,
            "dis_gen": dis_gen,
        }

def init_weights_normal(module: nn.Module, std: float = 0.02):
    """
    DCGAN-style initialization
    """

    classname = module.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname.find("Linear") != -1:
        nn.init.normal_(module.weight.data, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname.find("BatchNorm") != -1:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

def build_gonet_modules(
    nz: int = 100,
    use_tanh: bool = False,
    initialize: bool = True,
):
    """
    Helper function to construct all GONet modules.

    Returns:
        generator, invg, discriminator, classifier
    """

    generator = Generator(nz=nz, use_tanh=use_tanh)
    invg = InvG(nz=nz)
    discriminator = Discriminator()
    classifier = GONetClassifier()

    if initialize:
        generator.apply(init_weights_normal)
        invg.apply(init_weights_normal)
        discriminator.apply(init_weights_normal)
        classifier.apply(init_weights_normal)

    return generator, invg, discriminator, classifier

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    nz = 100

    gen, invg, dis, fl = build_gonet_modules(nz=nz)
    gen = gen.to(device)
    invg = invg.to(device)
    dis = dis.to(device)
    fl = fl.to(device)

    x = torch.randn(batch_size, 3, 128, 128).to(device)
    z = torch.randn(batch_size, nz).to(device)

    x_gen_from_z = gen(z)
    z_from_x = invg(x)

    dis_real = dis(x)
    dis_gen = dis(x_gen_from_z)

    prob = fl(
        img_error=x - x_gen_from_z,
        dis_error=dis_real - dis_gen,
        dis_real=dis_real,
    )

    print("Device:", device)
    print("Input image:", x.shape)
    print("Latent z:", z.shape)
    print("Generated image:", x_gen_from_z.shape)
    print("Encoded z:", z_from_x.shape)
    print("Discriminator features:", dis_real.shape)
    print("GONet probability:", prob.shape)