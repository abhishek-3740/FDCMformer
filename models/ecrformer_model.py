import torch
from torch import Tensor, nn
from timm.layers import to_2tuple, trunc_normal_
from einops import rearrange
from typing import Optional, Sequence

try:
    from models.module_util import LayerNorm, split_integer
    from models.module import (
        ChannelAttention, SpatialAttention, LinearAttnBlock,
        RestormerBlock, ECRformerBlock,
        TopkAttnBlock, UNetBottleneck,
    )
except ImportError:
    from module_util import LayerNorm, split_integer
    from module import (
        ChannelAttention, SpatialAttention, LinearAttnBlock,
        RestormerBlock, ECRformerBlock,
        TopkAttnBlock, UNetBottleneck,
    )

if __name__ != "__main__":
    print = lambda *args, **kwargs: ...


def format_shape(shape):
    if isinstance(shape, (list, tuple)):
        return f"({', '.join(str(s) for s in shape)})"
    return str(shape)


class ECRformerModel(nn.Module):
    def __init__(
        self,
        in_chans: int | Sequence[int] = 3,
        out_chans: int = 3,
        num_layers: int = 4,
        num_blocks: Sequence[int] = [4, 3, 2, 1],
        features_start: int = 64,
        drop_path_rate: float = 0.,
        bilinear: bool = False,
        cbam: Optional[str] = None,
        block_type: str | Sequence[str] = 'multi_dilated',
        conv_type: str = 'conv',
        norm_type: str = 'batch',
        decoupled_input: bool = True,
        bottle_neck: Optional[str] = 'topk',
        num_refine: Optional[int] = None,
        pos_encoding: Optional[str] = None,
        gated_skip: bool = True,
        bcma: bool = False,
    ) -> None:
        if num_layers < 1:
            raise ValueError(
                f"num_layers = {num_layers}, expected: num_layers > 0")

        assert len(num_blocks) == num_layers
        if not isinstance(in_chans, list):
            in_chans = [in_chans]
        num_inputs = len(in_chans)

        dpr_cnt = 0
        dprs = [x.item() for x in torch.linspace(
            0, drop_path_rate, num_layers * 2 - 2)]

        down_block_type, up_block_type = to_2tuple(block_type)

        assert bottle_neck in ['topk', 'sa', 'tsa', None]
        assert pos_encoding in ['rand', 'sin', None]

        super().__init__()
        self.num_layers = num_layers
        self.decoupled_input = decoupled_input
        self.bottle_neck = bottle_neck
        self.gated_skip = gated_skip
        self.bcma = bcma

        num_bottle_neck = num_blocks[-1]
        self.num_bottle_neck = num_bottle_neck

        conv_type = get_conv(conv_type)
        norm_type = get_norm(norm_type)

        # Stem
        if self.bcma:
            # M1: Bi-Cross-Modal Attention stem (requires two decoupled inputs).
            assert num_inputs == 2, \
                f"bcma=True requires exactly two inputs [SAR, optical], got {in_chans}"
            self.stem = BCMAStem(
                in_ch_list=in_chans,
                out_ch_list=split_integer(features_start, num_inputs, [1, 1]),
                kernel_size=7,
                conv_type=conv_type, norm_type=norm_type)
        elif self.decoupled_input:
            self.stem = DecoupledEncoder(
                in_ch_list=in_chans,
                out_ch_list=split_integer(features_start, num_inputs, [1, 1]),
                kernel_size=7, dilation=[1, ],
                conv_type=conv_type, norm_type=norm_type)
        else:
            self.stem = nn.Conv2d(
                in_chans, features_start,
                kernel_size=7, padding=3, padding_mode='reflect')
            # raise NotImplementedError()

        encoder_feats = [features_start * (2 ** i) for i in range(num_layers)]
        decoder_feats = encoder_feats[::-1]
        decoder_feats[-1] *= 2
        print(f"encoder_feats: {encoder_feats}")
        print(f"decoder_feats: {decoder_feats}")

        # Encoder
        encoder = []
        downsampler = []
        down_proj = []
        for idx in range(num_layers - 1):
            feats = encoder_feats[idx]
            next_feats = encoder_feats[idx + 1]

            down_post = []
            if isinstance(cbam, str):
                for item in cbam.lower().split('+'):
                    if '1ca' in item:
                        down_post.append(ChannelAttention(feats))
                    if '1sa' in item:
                        down_post.append(SpatialAttention())
                    if '1la' in item:
                        down_post.append(LinearAttnBlock(feats))
                    if '1ta' in item:
                        down_post.append(RestormerBlock(feats))

            down_fn = [get_block(down_block_type)(
                feats, drop_path_rate=dprs[dpr_cnt], conv_type=conv_type, norm_type=norm_type,
            ) for _ in range(num_blocks[idx])]

            dpr_cnt += 1
            encoder.append(nn.Sequential(*down_fn, *down_post))

            down_proj.append(nn.Conv2d(feats, out_chans,
                                       kernel_size=3, padding=1, padding_mode='reflect'))

            downsampler.append(nn.Sequential(
                nn.Conv2d(feats, next_feats // 4, kernel_size=3, stride=1,
                          padding=1, padding_mode='reflect', bias=False),
                nn.PixelUnshuffle(2),
                ChannelAttention(next_feats),
                SpatialAttention(),
            ))

        self.encoder = nn.ModuleList(encoder)
        self.downsampler = nn.ModuleList(downsampler)

        # Bottleneck
        feats = encoder_feats[-1]
        self.bottle_neck_module = None
        if bottle_neck and num_bottle_neck:
            if bottle_neck == 'sa':
                self.bottle_neck_module = nn.Sequential(
                    *[UNetBottleneck(feats, feats)
                      for _ in range(num_bottle_neck)]
                )
            elif bottle_neck == 'topk':
                self.bottle_neck_module = nn.Sequential(
                    *[TopkAttnBlock(feats) for _ in range(num_bottle_neck)],
                )
            elif bottle_neck == 'tsa':
                self.bottle_neck_module = nn.Sequential(
                    *[RestormerBlock(feats)
                      for _ in range(num_bottle_neck)],
                )

        neck_size = 256 // 2 ** (num_layers - 1)
        if pos_encoding == 'rand':
            self.learned_pos_embed = nn.Parameter(
                torch.zeros(1, feats, *to_2tuple(neck_size)))
            trunc_normal_(self.learned_pos_embed, std=.02)
        elif pos_encoding == 'sin':
            self.learned_pos_embed = nn.Parameter(posemb_sincos_2d(
                *to_2tuple(neck_size), feats) / 50)
        elif not pos_encoding:
            self.learned_pos_embed = 0.

        # Decoder
        decoder = []
        upsampler = []
        up_proj = []
        skip_gates = []
        for idx in range(1, num_layers):
            last_feats = decoder_feats[idx - 1]
            feats = decoder_feats[idx]

            upsampler.append(
                nn.Sequential(
                    nn.Conv2d(last_feats, 4 * feats, kernel_size=3, stride=1,
                              padding=1, padding_mode='reflect', bias=False),
                    nn.PixelShuffle(2),
                    ChannelAttention(feats),
                    SpatialAttention())
                if last_feats != feats
                else nn.Sequential(
                    nn.Conv2d(last_feats, 2 * last_feats, kernel_size=3, stride=1,
                              padding=1, padding_mode='reflect', bias=False),
                    nn.PixelShuffle(2),
                    ChannelAttention(last_feats // 2),
                    SpatialAttention())
            )

            up_post = []
            if isinstance(cbam, str):
                for item in cbam.lower().split('+'):
                    if 'ca2' in item:
                        up_post.append(ChannelAttention(feats))
                    if 'sa2' in item:
                        up_post.append(SpatialAttention())
                    if 'la2' in item:
                        up_post.append(LinearAttnBlock(feats))
                    if 'ta2' in item:
                        up_post.append(RestormerBlock(feats))

            up_fn = [get_block(up_block_type)(
                feats, drop_path_rate=dprs[dpr_cnt], conv_type=conv_type, norm_type=norm_type)
                for _ in range(num_blocks[num_layers - idx - 1])]
            dpr_cnt += 1

            skip_feats = encoder_feats[-1 - idx]
            fuser = nn.Conv2d(feats + skip_feats, feats, kernel_size=3, padding=1,
                              padding_mode='reflect') if last_feats != feats else nn.Identity()
            decoder.append(nn.Sequential(fuser, *up_fn, *up_post))
            up_proj.append(nn.Conv2d(feats, out_chans,
                                     kernel_size=3, padding=1, padding_mode='reflect'))

            # M7: gated cross-scale skip. The upsampled decoder feature has
            # `x_up_ch` channels (== feats, or last_feats//2 at the last level
            # where last_feats == feats); the encoder skip has `skip_feats`.
            x_up_ch = feats if last_feats != feats else last_feats // 2
            skip_gates.append(GatedSkipFusion(dec_ch=x_up_ch, enc_ch=skip_feats)
                              if gated_skip else None)

        self.upsampler = nn.ModuleList(upsampler)
        self.decoder = nn.ModuleList(decoder)
        self.skip_gates = nn.ModuleList(skip_gates) if gated_skip else None

        # Refine
        feats = decoder_feats[-1]
        self.refine = None
        if num_refine:
            refine = [RestormerBlock(feats, kernel_size=3, dilation=[1, 2, 4],
                                     conv_type=conv_type, norm_type=norm_type)
                      for _ in range(num_refine)]
            self.refine = nn.Sequential(*refine)

        # Final Convolution
        self.final_conv = nn.Conv2d(feats, out_chans,
                                    kernel_size=3, padding=1, padding_mode='reflect')

        self.down_proj = nn.ModuleList(down_proj)
        self.up_proj = nn.ModuleList(up_proj)

    def forward(self, x: Tensor, return_map=False) -> Tensor:
        # Stem
        x_temp = self.stem(x)
        x_enc = [x_temp]
        print(f"Stem: {format_shape(x.shape)} -> {format_shape(x_temp.shape)}")

        # Encoder path
        print("\nEncoder path:")
        down_projs = []
        for i in range(len(self.encoder)):
            down = self.downsampler[i]
            layer = self.encoder[i]
            proj = self.down_proj[i]

            x_out = layer(x_temp)
            x_enc.append(x_out)
            x_down = down(x_out)

            print(f"Input: {format_shape(x_temp.shape)}, "
                  f"Encoder_{i} out: {format_shape(x_out.shape)}, "
                  f"Down: {format_shape(x_down.shape)}")

            mid = proj(x_out)
            print(f"Down proj_{i}: {format_shape(x_enc[i+1].shape)}"
                  f" -> {format_shape(mid.shape)}")
            down_projs.append(mid)

            x_temp = x_down

        # Bottleneck
        print("\nBottleneck:")
        neck = x_temp
        if self.bottle_neck_module and self.num_bottle_neck:
            neck = self.bottle_neck_module(neck + self.learned_pos_embed)
            print(f"Bottleneck: {neck.shape}")

        # Decoder path
        print("\nDecoder path:")
        x_temp = neck
        up_projs = []
        x_dec = []

        for i in range(len(self.decoder)):
            up = self.upsampler[i]
            layer = self.decoder[i]
            proj = self.up_proj[i]

            skip = x_enc[-1 - i]
            x_up = up(x_temp)

            print(f"Up: {format_shape(x_temp.shape)} -> "
                  f"{format_shape(x_up.shape)}, "
                  f"skip: {format_shape(skip.shape)}")

            # M7: gated cross-scale skip (falls back to plain concat if disabled)
            if self.skip_gates is not None:
                x_fuse = self.skip_gates[i](x_up, skip)
            else:
                x_fuse = torch.cat([x_up, skip], dim=1)
            x_out = layer(x_fuse)
            x_dec.append(x_out)

            print(f"Fuse (x_up+skip): {format_shape(x_fuse.shape)}, "
                  f"Decoder_{i} out: {format_shape(x_out.shape)}")

            mid = proj(x_out)
            print(f"Up proj_{i}: {format_shape(x_dec[i].shape)}"
                  f" -> {format_shape(mid.shape)}")
            up_projs.append(mid)

            x_temp = x_out

        # Refine
        if self.refine:
            print("\nRefine:")
            x_temp = self.refine(x_temp)
            print(f"Refine out: {format_shape(x_temp.shape)}")

        x_dec.append(x_temp)

        # Final Convolution
        x_temp = self.final_conv(x_temp)
        print(f'Final out: {format_shape(x_temp.shape)}')

        if not return_map:
            return x_temp, (down_projs, up_projs)
        else:
            return x_temp, (down_projs, up_projs), x_enc, x_dec


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def get_conv(conv_type):
    if conv_type == 'conv':
        return nn.Conv2d
    else:
        raise ValueError(f"Unsupported conv_type: {conv_type}")


def get_norm(norm_type):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'layer':
        return LayerNorm
    elif norm_type == 'group':
        return nn.GroupNorm
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


def get_block(block_type):
    block_type = block_type.lower().replace('_', '')
    if block_type == 'restormer':
        return RestormerBlock
    elif block_type == 'ecrformer':
        return ECRformerBlock
    else:
        raise ValueError(f"Invalid block type: {block_type}")


class DecoupledEncoder(nn.Module):
    """Decoupled input encoder for separate SAR/optical branches."""

    def __init__(self, in_ch_list: list[int], out_ch_list: list[int],
                 kernel_size: int = 7, dilation: list[int] = [1, ],
                 conv_type=nn.Conv2d, norm_type=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.in_ch_list = in_ch_list
        branch = []
        for in_ch, out_ch in zip(in_ch_list, out_ch_list):
            print(f"DecoupledEncoder: {in_ch} -> {out_ch}")
            branch.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    padding=kernel_size // 2, padding_mode='reflect'))
        self.branch = nn.ModuleList(branch)

    def forward(self, x):
        inputs = torch.split(x, self.in_ch_list, dim=1)
        print(
            f"Input shape: {x.shape}, split into {[i.shape for i in inputs]}")
        outputs = [f(x) for f, x in zip(self.branch, inputs)]
        return torch.cat(outputs, dim=1)


class GatedSkipFusion(nn.Module):
    """M7: Gated cross-scale skip connection.

    The vanilla U-Net concatenates the encoder skip features directly with the
    upsampled decoder features. Under thick cloud the encoder skip still carries
    a residual cloud signature, which then leaks into the decoder and corrupts
    texture. This module learns a per-channel spatial gate that suppresses
    cloud-contaminated encoder features before fusion:

        g     = sigmoid(Conv1x1([x_dec; x_enc]))      # gate, enc_ch channels
        out   = concat([x_dec, g * x_enc])

    The output channel count is kept identical to the original
    ``concat([x_dec, x_enc])`` (``dec_ch + enc_ch``) so the downstream fuser /
    decoder blocks are unchanged -- this is a drop-in replacement for the plain
    concat. With zero-initialised bias the gate starts at ``0.5`` (half-pass),
    so training begins close to the original behaviour and learns to gate from
    there.
    """

    def __init__(self, dec_ch: int, enc_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dec_ch + enc_ch, enc_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor:
        g = self.gate(torch.cat([x_dec, x_enc], dim=1))
        return torch.cat([x_dec, g * x_enc], dim=1)


class CrossXCA(nn.Module):
    """M1: Cross-covariance cross-attention (XCA).

    Query is projected from ``x_q``, Key/Value from ``x_kv``. Like the model's
    existing ``TransposedAttention`` (MDTA), attention is computed over the
    channel dimension (a ``C x C`` map), so cost is **linear** in the number of
    spatial tokens -- the cheap, XCA-style attention referenced in the M1 spec.
    A depthwise 3x3 conv on q/k/v injects local spatial context before the
    channel attention.
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim,
                              bias=bias, padding_mode='reflect')
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1,
                               groups=dim * 2, bias=bias, padding_mode='reflect')
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        _, _, h, w = x_q.shape
        q = self.q_dw(self.q(x_q))
        k, v = self.kv_dw(self.kv(x_kv)).chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class BCMAStem(nn.Module):
    """M1: Bi-Cross-Modal Attention stem (fixes B2 -- trivial concat fusion).

    Drop-in replacement for ``DecoupledEncoder``. SAR and optical are encoded by
    separate 7x7 stems (the SAR stem includes a learnable depthwise
    speckle-reduction conv, the spec's mitigation for SAR speckle suppressing
    the gate). Each modality is then refined by cross-covariance cross-attention
    against the other, and merged with its own features via a learned modality
    gate::

        F_o' = g_o * F_o + (1 - g_o) * CrossXCA(F_o <- F_s)   # optical <- SAR
        F_s' = g_s * F_s + (1 - g_s) * CrossXCA(F_s <- F_o)   # SAR <- optical
        out  = concat([F_o', F_s'])                            # == features_start

    Output channels equal ``sum(out_ch_list)`` (== ``features_start``), so the
    whole downstream encoder/decoder is untouched. ``initialize_weights`` zeros
    every Conv2d bias, so the "start near the original decoupled stem" behaviour
    is anchored by a standalone learnable ``gate_bias`` parameter (init +3 ->
    gate ~= 0.95, i.e. mostly self-features) which Kaiming init does not touch.
    """

    def __init__(self, in_ch_list, out_ch_list, kernel_size: int = 7,
                 num_heads: int = 4, conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d, **kwargs):
        super().__init__()
        assert len(in_ch_list) == 2, \
            f"BCMA expects exactly two modalities [SAR, optical], got {in_ch_list}"
        assert out_ch_list[0] == out_ch_list[1], \
            f"BCMA requires equal branch widths, got {out_ch_list}"
        self.in_ch_list = list(in_ch_list)
        sar_in, opt_in = in_ch_list
        c = out_ch_list[0]

        # pick the largest #heads (<= num_heads) that divides c
        heads = num_heads
        while c % heads != 0 and heads > 1:
            heads -= 1
        self.num_heads = heads
        print(f"BCMAStem: SAR {sar_in}->{c}, optical {opt_in}->{c}, heads={heads}")

        pad = kernel_size // 2
        # SAR stem + learnable depthwise speckle-reduction conv
        self.sar_stem = nn.Sequential(
            nn.Conv2d(sar_in, c, kernel_size, padding=pad, padding_mode='reflect'),
            nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c,
                      padding_mode='reflect'),
        )
        self.opt_stem = nn.Conv2d(opt_in, c, kernel_size, padding=pad,
                                  padding_mode='reflect')

        self.norm_oq = LayerNorm(c)
        self.norm_ok = LayerNorm(c)
        self.norm_sq = LayerNorm(c)
        self.norm_sk = LayerNorm(c)
        self.cross_o = CrossXCA(c, heads)   # optical queries SAR
        self.cross_s = CrossXCA(c, heads)   # SAR queries optical

        self.gate_o = nn.Conv2d(2 * c, c, kernel_size=1)
        self.gate_s = nn.Conv2d(2 * c, c, kernel_size=1)
        # survives initialize_weights (not a Conv2d/Linear/BN): keeps the gate
        # near 1.0 at init so training starts close to the decoupled-stem model.
        self.gate_bias_o = nn.Parameter(torch.tensor(3.0))
        self.gate_bias_s = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        sar, opt = torch.split(x, self.in_ch_list, dim=1)
        Fs = self.sar_stem(sar)
        Fo = self.opt_stem(opt)

        Fo_c = self.cross_o(self.norm_oq(Fo), self.norm_ok(Fs))   # optical <- SAR
        Fs_c = self.cross_s(self.norm_sq(Fs), self.norm_sk(Fo))   # SAR <- optical

        go = torch.sigmoid(self.gate_o(torch.cat([Fo, Fs], dim=1)) + self.gate_bias_o)
        gs = torch.sigmoid(self.gate_s(torch.cat([Fs, Fo], dim=1)) + self.gate_bias_s)

        Fo = go * Fo + (1 - go) * Fo_c
        Fs = gs * Fs + (1 - gs) * Fs_c
        return torch.cat([Fo, Fs], dim=1)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = rearrange(pe, '(h w) c -> 1 c h w', h=h, w=w)
    return pe.type(dtype)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ECRformer
    net = ECRformerModel(
        in_chans=[2, 13], out_chans=13, cbam='1ca2+1sa2', pos_encoding=None,
        num_layers=4, num_blocks=[2, 3, 2, 2], bottle_neck='tsa', num_refine=4,
        features_start=48, drop_path_rate=0., conv_type='conv', norm_type='batch',
        block_type=['ecrformer', 'ecrformer'], decoupled_input=True)
    # ECRformer-Light
    net = ECRformerModel(
        in_chans=[2, 13], out_chans=13, cbam='1ca2+1sa2', pos_encoding=None,
        num_layers=4, num_blocks=[2, 2, 1, 1], bottle_neck='tsa', num_refine=2,
        features_start=32, drop_path_rate=0., conv_type='conv', norm_type='batch',
        block_type=['ecrformer', 'ecrformer'], decoupled_input=True)

    def count_parameters(model, mode='return'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def smart_print(num, title=''):
        units = ['', 'K', 'M', 'G']
        unit_idx = 0
        while num > 1000:
            num /= 1000
            unit_idx += 1
        prefix = f"{title}: " if title else ""
        print(f"{prefix}{num:.4f}{units[unit_idx]}")

    total = count_parameters(net)
    down_proj_params = count_parameters(net.down_proj)
    up_proj_params = count_parameters(net.up_proj)
    smart_print(total - down_proj_params - up_proj_params)
    smart_print(down_proj_params)
    smart_print(up_proj_params)
