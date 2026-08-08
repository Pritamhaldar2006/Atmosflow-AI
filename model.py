"""
LiteRIFE: a compact, RIFE-inspired frame interpolation network,
sized to fine-tune on a single Colab T4 (16 GB) at 256x256 patches.

Architecture (see README for the full rationale):
  1. IFBlock x3 (coarse -> fine) -- predicts bidirectional optical flow
     and a fusion/occlusion mask directly, in a supervised end-to-end
     fashion (no separate classical optical-flow step).
  2. Backward warping of I0 and I1 toward the target timestep using the
     predicted flow.
  3. A lightweight U-Net "refine" head that takes the warped frames,
     the flow, the fusion mask, and the raw context features, and
     predicts a residual correction. This is what fixes the blur /
     ghosting that plain flow-warping produces on deforming cloud fields.

All channels are treated as single-channel (grayscale IR/VIS brightness
temperature or reflectance) by default; set `in_channels` to 3 if you
concatenate multiple spectral bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, flow):
    """Backward-warp `img` using optical flow `flow` (B,2,H,W)."""
    B, C, H, W = img.shape
    xx = torch.arange(0, W, device=img.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=img.device).view(-1, 1).repeat(1, W)
    grid = torch.stack((xx, yy), 0).float().unsqueeze(0).repeat(B, 1, 1, 1)
    vgrid = grid + flow
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(img, vgrid, mode="bilinear",
                          padding_mode="border", align_corners=True)


def conv(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=True)]
    if act:
        layers.append(nn.PReLU(out_ch))
    return nn.Sequential(*layers)


class IFBlock(nn.Module):
    """Single coarse-to-fine flow-estimation block.

    Input: concat(I0, I1, warped_flow_from_coarser_scale, t_embedding)
    Output: delta_flow (4 ch: flow0 + flow1), fusion_mask (1 ch)
    """

    def __init__(self, in_ch, hidden=96):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_ch, hidden, 3, 2, 1),
            conv(hidden, hidden, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(hidden, hidden), conv(hidden, hidden),
            conv(hidden, hidden), conv(hidden, hidden),
        )
        self.lastconv = nn.ConvTranspose2d(hidden, 5, 4, 2, 1)

    def forward(self, x, scale=1.0):
        H, W = x.shape[-2:]
        if scale != 1.0:
            # analyze at a coarser resolution for large/global motion,
            # then resize back to the original H, W exactly (robust to
            # non-power-of-2 sizes, unlike chaining two scale_factor calls)
            small_h, small_w = max(1, int(H / scale)), max(1, int(W / scale))
            x_in = F.interpolate(x, size=(small_h, small_w), mode="bilinear",
                                  align_corners=False)
        else:
            x_in = x
        feat = self.conv0(x_in)          # internal stride-4 downsample
        feat = self.convblock(feat) + feat
        out = self.lastconv(feat)        # stride-2 upsample (half of stride-4)
        out = F.interpolate(out, size=(H, W), mode="bilinear",
                             align_corners=False)
        flow = out[:, :4] * scale
        mask = out[:, 4:5]
        return flow, mask


class ContextNet(nn.Module):
    """Shallow feature extractor used to give the refine head texture
    context beyond the raw warped pixels (helps hallucinate plausible
    cloud texture in occluded / newly-forming regions)."""

    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.pyramid1 = conv(in_ch, hidden, 3, 2, 1)
        self.pyramid2 = conv(hidden, hidden, 3, 1, 1)

    def forward(self, x, flow):
        f1 = self.pyramid1(x)
        f1_warped = warp(f1, F.interpolate(flow, scale_factor=0.5,
                                            mode="bilinear",
                                            align_corners=False) * 0.5)
        f2 = self.pyramid2(f1_warped)
        return f2


class RefineNet(nn.Module):
    """Small U-Net-style residual refinement head."""

    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.down1 = conv(in_ch, hidden, 3, 2, 1)
        self.down2 = conv(hidden, hidden * 2, 3, 2, 1)
        self.mid = nn.Sequential(conv(hidden * 2, hidden * 2),
                                  conv(hidden * 2, hidden * 2))
        self.up2 = nn.ConvTranspose2d(hidden * 2, hidden, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1)
        self.out = nn.Conv2d(hidden // 2, 1, 3, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m = self.mid(d2) + d2
        u2 = self.up2(m) + d1
        u1 = self.up1(u2)
        return torch.tanh(self.out(u1))  # residual in [-1, 1]


class LiteRIFE(nn.Module):
    """Full pipeline: 3-scale coarse-to-fine flow + fusion + refinement.

    forward(I0, I1, t) -> predicted frame at relative time t in (0, 1)
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        pair_ch = in_channels * 2
        self.block0 = IFBlock(pair_ch + 4 + 1, hidden=96)   # coarsest
        self.block1 = IFBlock(pair_ch + 4 + 1, hidden=64)
        self.block2 = IFBlock(pair_ch + 4 + 1, hidden=48)   # finest
        self.contextnet = ContextNet(in_channels, hidden=32)
        self.refine = RefineNet(pair_ch + 4 + 1 + 32 * 2, hidden=64)

    def forward(self, I0, I1, t):
        B, C, H, W = I0.shape
        t_map = torch.full((B, 1, H, W), float(t), device=I0.device,
                            dtype=I0.dtype)
        flow = torch.zeros(B, 4, H, W, device=I0.device, dtype=I0.dtype)
        mask = torch.zeros(B, 1, H, W, device=I0.device, dtype=I0.dtype)

        for block, scale in [(self.block0, 4.0), (self.block1, 2.0),
                              (self.block2, 1.0)]:
            x = torch.cat([I0, I1, flow, t_map], dim=1)
            d_flow, d_mask = block(x, scale=scale)
            flow = flow + d_flow
            mask = mask + d_mask

        warped0 = warp(I0, flow[:, :2])
        warped1 = warp(I1, flow[:, 2:4])
        fusion = torch.sigmoid(mask)
        blended = warped0 * fusion + warped1 * (1 - fusion)

        c0 = self.contextnet(I0, flow[:, :2])
        c1 = self.contextnet(I1, flow[:, 2:4])
        refine_in = torch.cat(
            [warped0, warped1, flow, fusion,
             F.interpolate(c0, size=(H, W), mode="bilinear", align_corners=False),
             F.interpolate(c1, size=(H, W), mode="bilinear", align_corners=False)],
            dim=1,
        )
        residual = self.refine(refine_in)
        pred = torch.clamp(blended + residual * 0.1, 0.0, 1.0)
        return pred, flow, fusion


if __name__ == "__main__":
    # quick shape sanity check (CPU, tiny tensor)
    m = LiteRIFE(in_channels=1)
    a = torch.rand(1, 1, 64, 64)
    b = torch.rand(1, 1, 64, 64)
    out, flow, mask = m(a, b, 0.5)
    print("output:", out.shape, "flow:", flow.shape, "mask:", mask.shape)
