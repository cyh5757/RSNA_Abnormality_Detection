"""Two-factor position ablation: categorical plane x continuous slice depth.

Kept separate so an already running spatial/follow-up queue keeps its source hash.
This head retains S11's per-series RoPE and class-query cross attention.
"""
from experiments.spatial_core import *


class PlaneDepthHead(SpatialHead):
    def __init__(self, config, mean, std):
        if (config['fusion'] != 'cross' or not config['sequence'] or
                not config['plane_embedding'] or config['position'] != 'rope_mm'):
            raise ValueError('Plane-depth ablation requires the fixed S11 architecture.')
        if config.get('plane_depth_mode') not in ('none', 'sum', 'joint'):
            raise ValueError('Unknown plane-depth encoding.')
        if not math.isfinite(config['mm_unit']) or config['mm_unit'] <= 0:
            raise ValueError('mm_unit must be positive and finite.')
        super().__init__(config, mean, std)

    def position_features(self, position):
        """(B, 6, T) -> (B, 6, T, D), with no new trainable parameters.

        방향은 0/1/2의 연속 좌표가 아닌 Sagittal/Coronal/Axial 범주이다.
        slice 위치는 series 첫 장에서의 mm 거리 / mm_unit; 없으면 정렬 index.
        sum: 방향 embedding과 깊이 sin/cos를 별도로 더한다.
        joint: 방향 embedding으로 깊이 신호를 조절하는 상호작용도 더한다.
        tanh는 조절량을 [-1, 1]로 제한한다. 실제 학습에서는 기존 plane
        embedding으로 gradient가 흐른다. 이것은 2D RoPE나 patch (x,y)가 아니다.
        """
        d = self.config['hidden_dim']
        frequencies = torch.exp(torch.arange(0, d, 2, device=position.device,
                                            dtype=torch.float32) * (-math.log(10000.) / d))
        angles = position.float()[..., None] * frequencies
        depth = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        mode = self.config['plane_depth_mode']
        if mode == 'none':
            return torch.zeros_like(depth)
        if mode == 'sum':
            return depth
        plane = self.plane(torch.arange(6, device=position.device) // 2)
        return depth + torch.tanh(plane)[None, :, None, :] * depth

    def forward(self, tokens, mask, coords):
        # S11 cross branch reproduced here to leave active historical code untouched.
        # A regression test requires exact S11 outputs/gradients in the 'none' control.
        if tokens.ndim != 4 or tokens.shape[1] != 6 or mask.shape != tokens.shape[:3] or coords.shape != (*mask.shape, 3):
            raise ValueError('Invalid token/coordinate shape.')
        if not mask.flatten(1).any(1).all():
            raise ValueError('Each study requires a valid image.')
        coords = torch.where(mask[..., None], coords, 0.)
        clean = torch.where(mask[..., None], tokens.float(), self.mean)
        if not torch.isfinite(coords).all() or not torch.isfinite(clean).all():
            raise ValueError('Non-finite valid token or coordinate.')
        x = self.project((clean - self.mean) / self.std)
        x = x + self.coord_kind((coords[..., 2] > .5).long())
        b, s, t, d = x.shape
        slot = torch.arange(6, device=x.device)
        x = x + (self.plane(slot // 2) + self.protocol(slot % 2))[None, :, None, :]
        pos = torch.where(coords[..., 2] > .5, coords[..., 1] / self.config['mm_unit'], coords[..., 0])
        # NEW: absolute depth is visible to the cross-view class queries as well.
        # Existing RoPE below still supplies relative depth inside each series.
        if self.config['plane_depth_mode'] != 'none':
            x = x + self.position_features(pos)
        x = x * mask[..., None]
        flat = x.reshape(b * s, t, d)
        flat_mask = mask.reshape(b * s, t)
        active = flat_mask.any(1)
        result = torch.zeros_like(flat)
        result[active] = self.slice_block(flat[active], flat_mask[active],
                                         pos.reshape(b * s, t)[active], True)
        x = result.reshape(b, s, t, d)
        memory = self.memory_norm(x.flatten(1, 2))
        query = self.queries[None].expand(b, -1, -1)
        read, _ = self.cross(self.query_norm(query), memory, memory,
                             key_padding_mask=~mask.flatten(1), need_weights=False)
        read = query + read
        read = read + self.read_ffn(self.read_norm(read))
        return (read * self.read_weight).sum(-1) + self.class_bias
