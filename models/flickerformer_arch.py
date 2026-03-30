import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN_AutoCorr(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN_AutoCorr, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.project_in(x)

        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft
        power = Xf * torch.conj(Xf)
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        Xf_new = Xf + self.alpha * power
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        x_patch_new = x_patch_new + self.beta * R

        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class DWT_WindowAttention_SW(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size=8, shift_size=4, bias=False):
        super(DWT_WindowAttention_SW, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        if (input_resolution // 2) <= window_size:
            self.shift_size = 0
            self.window_size = input_resolution // 2
            window_size = input_resolution // 2

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')

        self.high_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True)
        )
        self.high_out = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size)))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H//ws, ws, W//ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, ws, ws)
        return x

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        C = windows.shape[1]
        ws = self.window_size
        x = windows.view(B, H//ws, W//ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x

    def shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        return x

    def reverse_shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x

    def window_attention(self, q, k, v):
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        attn = torch.matmul(q.transpose(-2, -1), k)

        N = self.window_size * self.window_size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        LL, Yh = self.dwt(x)
        Yh = Yh[0]
        LH, HL, HH = Yh[:, :, 0, :, :], Yh[:, :, 1, :, :], Yh[:, :, 2, :, :]

        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))

        qkv = self.qkv_dwconv(self.qkv(LL))
        q, k, v_inp = qkv.chunk(3, dim=1)
        v = v_inp * filter_hv + v_inp

        x_shifted = self.shift(LL, self.shift_size)
        q = self.window_partition(x_shifted)
        k = self.window_partition(x_shifted)
        v = self.window_partition(v)

        B_win, Cq, ws, _ = q.shape
        q = q.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        k = k.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        v = v.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)

        out = self.window_attention(q, k, v)
        out = out.view(B_win, Cq, ws, ws)
        out = self.window_reverse(out, H//2, W//2)
        out = self.reverse_shift(out, self.shift_size)
        out = self.project_out(out)

        Yh = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH, HL, HH = Yh.chunk(3, dim=1)
        Yh = torch.stack([LH, HL, HH], dim=2)
        x_hat = self.idwt((out, [Yh]))

        return x_hat


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, ffn_expansion_factor, bias, LayerNorm_type, use_att=True):
        super(TransformerBlock, self).__init__()
        self.use_att = use_att
        if use_att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.att = DWT_WindowAttention_SW(dim, num_heads, input_resolution, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN_AutoCorr(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.use_att:
            x = x + self.att(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class PhaseGuidedFilter(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(PhaseGuidedFilter, self).__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.net12 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.net23 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(dim*3, dim, 3, 1, 1)
        self.group_conv = nn.Conv2d(dim*3, dim*3, 3, 1, 1, groups=3)
        self.eps = float(1e-8)

    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=1)

        f1 = torch.fft.rfft2(x1)
        f2 = torch.fft.rfft2(x2)
        f3 = torch.fft.rfft2(x3)

        mag_1 = torch.abs(f1)
        mag_2 = torch.abs(f2)
        mag_3 = torch.abs(f3)

        phase1 = f1 / (mag_1 + self.eps)
        phase2 = f2 / (mag_2 + self.eps)
        phase3 = f3 / (mag_3 + self.eps)

        C12 = torch.abs(phase1 * torch.conj(phase2))
        C23 = torch.abs(phase3 * torch.conj(phase2))

        C12 = self.net12(C12)
        C23 = self.net23(C23)

        f1_filtered = C12 * f1
        f3_filtered = C23 * f3

        x1_filtered = torch.fft.irfft2(f1_filtered)
        x3_filtered = torch.fft.irfft2(f3_filtered)

        out = torch.cat([x1_filtered, x2, x3_filtered], dim=1)
        out = self.fusion(out)

        return out


class Flickerformer(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        img_size=512,
        dim=32,
        num_blocks=[2, 2, 2, 2],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ):
        super(Flickerformer, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.fusion = PhaseGuidedFilter(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**0), num_heads=heads[0], input_resolution=img_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_att=False) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], input_resolution=img_size//2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_att=False) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], input_resolution=img_size//4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_att=False) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], input_resolution=img_size//4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], input_resolution=img_size//2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], input_resolution=img_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        x1, x2, x3 = inp_img.chunk(3, dim=1)
        inp_enc_level1_x1 = self.conv1(x1)
        inp_enc_level1_x2 = self.conv2(x2)
        inp_enc_level1_x3 = self.conv3(x3)

        out_enc_level1_x2 = self.fusion(torch.cat([inp_enc_level1_x1, inp_enc_level1_x2, inp_enc_level1_x3], dim=1))

        out_enc_level1_x2 = self.encoder_level1(out_enc_level1_x2)

        inp_enc_level2 = self.down1_2(out_enc_level1_x2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1_x2], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1 + x2
