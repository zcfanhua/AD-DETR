import torch, math, tqdm
import itertools
from timm.layers import SqueezeExcite, trunc_normal_
try:
    from torch.amp import custom_fwd, custom_bwd
except:
    from torch.cuda.amp import custom_fwd, custom_bwd
from torch.autograd import Function
try:
    import triton
    import triton.language as tl
except:
    pass

__all__ = ['lsnet_t', 'lsnet_s', 'lsnet_b', 'SKA']

try:
    def _grid(numel: int, bs: int) -> tuple:
        return (triton.cdiv(numel, bs),)

    @triton.jit
    def _idx(i, n: int, c: int, h: int, w: int):
        ni = i // (c * h * w)
        ci = (i // (h * w)) % c
        hi = (i // w) % h
        wi = i % w
        m = i < (n * c * h * w)
        return ni, ci, hi, wi, m

    @triton.jit
    def ska_fwd(
        x_ptr, w_ptr, o_ptr,
        n, ic, h, w, ks, pad, wc,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                x_off = ((ni * ic + ci) * h + hin) * w + win
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

        tl.store(o_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_x(
        go_ptr, w_ptr, gi_ptr,
        n, ic, h, w, ks, pad, wc,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            ho = hi + pad - kh
            hb = (ho >= 0) & (ho < h)
            for kw in range(ks):
                wo = wi + pad - kw
                b = hb & (wo >= 0) & (wo < w)

                go_off = ((ni * ic + ci) * h + ho) * w + wo
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

        tl.store(gi_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_w(
        go_ptr, x_ptr, gw_ptr,
        n, wc, h, w, ic, ks, pad,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)
                w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                val = tl.zeros((BS,), dtype=AT)
                steps = (ic - ci + wc - 1) // wc
                for s in range(tl.max(steps, axis=0)):
                    cc = ci + s * wc
                    cm = (cc < ic) & m & b

                    x_off = ((ni * ic + cc) * h + hin) * w + win
                    go_off = ((ni * ic + cc) * h + hi) * w + wi

                    x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                    go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                    val += tl.where(cm, x_val * go_val, 0.0).to(AT)

                tl.store(gw_ptr + w_off, val.to(CT), mask=m)
    
    class SkaFn(Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ks = int(math.sqrt(w.shape[2]))
            pad = (ks - 1) // 2
            ctx.ks, ctx.pad = ks, pad
            n, ic, h, width = x.shape
            wc = w.shape[1]
            o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
            numel = o.numel()

            x = x.contiguous()
            w = w.contiguous()

            grid = lambda meta: _grid(numel, meta["BS"])

            ct = tl.float16 if x.dtype == torch.float16 else (tl.float32 if x.dtype == torch.float32 else tl.float64)
            at = tl.float32 if x.dtype == torch.float16 else ct

            ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            ctx.save_for_backward(x, w)
            ctx.ct, ctx.at = ct, at
            return o

        @staticmethod
        @custom_bwd
        def backward(ctx, go: torch.Tensor) -> tuple:
            ks, pad = ctx.ks, ctx.pad
            x, w = ctx.saved_tensors
            n, ic, h, width = x.shape
            wc = w.shape[1]

            go = go.contiguous()
            gx = gw = None
            ct, at = ctx.ct, ctx.at

            if ctx.needs_input_grad[0]:
                gx = torch.empty_like(x)
                numel = gx.numel()
                ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            if ctx.needs_input_grad[1]:
                gw = torch.empty_like(w)
                numel = gw.numel() // w.shape[2]
                ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at)

            return gx, gw, None, None
except:
    def _idx(i, n: int, c: int, h: int, w: int):
        pass

    def ska_fwd(
        x_ptr, w_ptr, o_ptr,
        n, ic, h, w, ks, pad, wc,
        BS,
        CT, AT
    ):
        pass

    def ska_bwd_x(
        go_ptr, w_ptr, gi_ptr,
        n, ic, h, w, ks, pad, wc,
        BS,
        CT, AT
    ):
        pass

    def ska_bwd_w(
        go_ptr, x_ptr, gw_ptr,
        n, wc, h, w, ic, ks, pad,
        BS,
        CT, AT
    ):
        pass

class SKA(torch.nn.Module):
        def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return SkaFn.apply(x, w) # type: ignore

class SKA(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w) # type: ignore

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

import torch.nn as nn

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

class Block(torch.nn.Module):    
    def __init__(self,
                 ed, kd=16, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1):
        super().__init__()
            
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=8,
                 in_chans=3,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4]):
        super().__init__()

        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 1, 1)
                           )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                blocks[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                resolution_ = (resolution - 1) // 2 + 1
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = resolution_

        self.cuda()
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640).cuda())]

    def forward(self, x):
        outputs = []
        x = self.patch_embed(x)
        outputs.append(x)
        x = self.blocks1(x)
        # outputs.append(x)
        x = self.blocks2(x)
        outputs.append(x)
        x = self.blocks3(x)
        outputs.append(x)
        x = self.blocks4(x)
        outputs.append(x)
        return outputs
    
def lsnet_t(**kwargs):
    model = LSNet(img_size=640,
                  patch_size=4,
                  embed_dim=[64, 128, 256, 384],
                  depth=[0, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

def lsnet_s(**kwargs):
    model = LSNet(img_size=640,
                  patch_size=4,
                  embed_dim=[96, 192, 320, 448],
                  depth=[1, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

def lsnet_b(**kwargs):
    model = LSNet(img_size=640,
                  patch_size=4,
                  embed_dim=[128, 256, 384, 512],
                  depth=[4, 6, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

if __name__ == '__main__':
    model = lsnet_t().cuda()
    inputs = torch.randn((1, 3, 640, 640)).cuda()
    for i in tqdm.tqdm(range(100)):
        outputs = model(inputs)
    for i in outputs:
        print(i.size())