#!/usr/bin/env python3

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    GONet/DCGAN-style generator.

    Input:
        z: [B, nz]

    Output:
        image: [B, 3, 128, 128]
    """

    def __init__(self, nz: int = 100, use_tanh: bool = False):
        super().__init__()

        self.nz = nz
        self.use_tanh = use_tanh

        self.fc = nn.Sequential(
            nn.Linear(nz, 8 * 8 * 512),
            nn.BatchNorm1d(8 * 8 * 512),
            nn.ReLU(inplace=True),
        )

        layers = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        ]

        if use_tanh:
            layers.append(nn.Tanh())

        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.reshape(z.size(0), 512, 8, 8)
        x = self.deconv(x)
        return x


class InvG(nn.Module):
    """
    Inverse generator.

    Maps an image back into the generator latent space.

    Input:
        image: [B, 3, 128, 128]

    Output:
        z: [B, nz]
    """

    def __init__(self, nz: int = 100):
        super().__init__()

        self.nz = nz

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512 * 8 * 8, nz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.reshape(x.size(0), -1)
        z = self.fc(h)
        return z


class Discriminator(nn.Module):
    """
    Discriminator from GONet.

    In the original Chainer inference code, the discriminator computes logits
    internally but returns the intermediate feature map h.

    We expose both options here.

    Input:
        image: [B, 3, 128, 128]

    Output by default:
        features: [B, 512, 8, 8]

    If return_logits=True:
        features, logits
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
        )

        self.classifier = nn.Linear(512 * 8 * 8, 2)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        h = self.features(x)

        if return_logits:
            logits = self.classifier(h.reshape(x.size(0), -1))
            return h, logits

        return h


class GONetClassifier(nn.Module):
    """
    Final GONet classification layer, equivalent to FL in the Chainer code.

    Inputs:
        img_error: [B, 3, 128, 128]
            img_real - img_gen

        dis_error: [B, 512, 8, 8]
            dis_real - dis_gen

        dis_real: [B, 512, 8, 8]
            discriminator features of the real image

    Output:
        prob: [B, 1]
            traversability probability
    """

    def __init__(self):
        super().__init__()

        self.l_img = nn.Linear(3 * 128 * 128, 1)
        self.l_dis = nn.Linear(512 * 8 * 8, 1)
        self.l_fdis = nn.Linear(512 * 8 * 8, 1)
        self.l_final = nn.Linear(3, 1)

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
    DCGAN-style initialization, matching the original Chainer code's
    Normal(wscale=0.02) initialization approximately.
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