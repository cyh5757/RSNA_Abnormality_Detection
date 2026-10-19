"""Slice tokens with explicit coordinates; identical training and inference path."""
from experiments.core import *


def spatial_order(folder):
    import pydicom
    paths = sorted(folder.glob('*.dcm'))
    if not paths:
        raise FileNotFoundError(folder)
    headers = [pydicom.dcmread(p, stop_before_pixels=True, specific_tags=[
        'ImagePositionPatient', 'ImageOrientationPatient', 'InstanceNumber']) for p in paths]
    physical = all(hasattr(h, 'ImagePositionPatient') and hasattr(h, 'ImageOrientationPatient') for h in headers)
    if physical:
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        if any(not np.allclose(h.ImageOrientationPatient, orientation, atol=1e-4) for h in headers):
            raise ValueError('Inconsistent orientations inside one series.')
        normal = np.cross(orientation[:3], orientation[3:])
        if not .99 < np.linalg.norm(normal) < 1.01:
            raise ValueError('Invalid DICOM orientation.')
        values = np.array([np.dot(h.ImagePositionPatient, normal) for h in headers])
    elif all(hasattr(h, 'InstanceNumber') for h in headers):
        values = np.array([float(h.InstanceNumber) for h in headers])
    else:
        raise ValueError('Missing physical and instance coordinates.')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite DICOM coordinates.')
    order = np.argsort(values, kind='stable')
    coords = np.zeros((len(paths), 3), dtype=np.float32)
    coords[:, 0] = np.arange(len(paths))
    coords[:, 1] = values[order] - values[order][0] if physical else 0.
    coords[:, 2] = float(physical)
    return [paths[i] for i in order], coords


def selected_windows(length, config):
    if length < 1 or config['neighbors'] not in (3, 5, 10) or config['centers'] < 1:
        raise ValueError('Invalid slice selection.')
    start, end = config.get('center_range', [.25, .75])
    if not 0 <= start < end <= 1: raise ValueError('Invalid center range.')
    centers = np.unique(np.rint(np.linspace(start, end, config['centers']) * (length-1)).astype(int))
    if config['representation'] == 'triplet':
        if config['neighbors'] != 3:
            raise ValueError('Triplet encoder requires three channels.')
        return [tuple(np.clip([c-1,c,c+1], 0, length-1)) for c in centers]
    if config['representation'] != 'single':
        raise ValueError('Unknown representation.')
    left = (config['neighbors'] - 1) // 2
    indices = {int(np.clip(c+j, 0, length-1)) for c in centers
               for j in range(-left, config['neighbors']-left)}
    return [(i,) for i in sorted(indices)]


def spatial_views(uid, series, image_root, transform, config, audit=None):
    count = config['centers'] * (config['neighbors'] if config['representation'] == 'single' else 1)
    images = torch.zeros(6, count, 3, config['crop'], config['crop'])
    mask = torch.zeros(6, count, dtype=torch.bool)
    coords = torch.zeros(6, count, 3)
    rows = series.loc[series.StudyInstanceUID == uid]
    for slot, (plane, fs) in enumerate((p, f) for p in PLANES for f in (True, False)):
        choices = rows.loc[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
        choices = choices.sort_values(['Fluid_Sensitive', 'SeriesInstanceUID'], ascending=[False, True])
        if choices.empty:
            continue
        paths, positions = spatial_order(image_root / uid / choices.iloc[0].SeriesInstanceUID)
        decoded = {}
        windows=selected_windows(len(paths),config)
        if audit is not None:
            indices=sorted({i for window in windows for i in window})
            audit.append(dict(StudyInstanceUID=uid,slot=slot,plane=plane,fat_suppression=fs,
                original_slices=len(paths),unique_selected_slices=len(indices),tokens=len(windows),
                selected_index_min=min(indices),selected_index_max=max(indices),valid_fraction=len(windows)/count))
        for j, window in enumerate(windows):
            for index in window:
                if index not in decoded:
                    decoded[index] = image_tensor(paths[index], lambda x:x)[0]
            image = torch.stack([decoded[i] for i in window]) if len(window) == 3 else decoded[window[0]][None].repeat(3,1,1)
            images[slot, j] = transform(image)
            coords[slot, j] = torch.from_numpy(positions[window[len(window)//2]])
            mask[slot, j] = True
    if not mask.any():
        raise ValueError(f'No supported images for {uid}')
    return images, mask, coords


def spatial_signature(config):
    signature = {k: config[k] for k in ('representation', 'neighbors', 'centers', 'crop',
                                       'resize', 'interpolation', 'pipeline_sha256')}
    signature['pipeline_sha256'] = config.get('feature_pipeline_sha256', config['pipeline_sha256'])
    # Legacy configs used this exact range; keep their serialized cache signature valid.
    if list(config.get('center_range', [.25,.75])) != [.25,.75]:
        signature['center_range'] = list(config['center_range'])
    return signature


@torch.inference_mode()
def spatial_tokens(uid, series, image_root, encoder, transform, config, device, audit=None):
    images, mask, coords = spatial_views(uid, series, image_root, transform, config, audit=audit)
    output = torch.cat([encoder(x.to(device)).cpu().float() for x in images[mask].split(16)])
    if output.shape != (int(mask.sum()), config['feature_dim']) or not torch.isfinite(output).all():
        raise ValueError('Invalid spatial features.')
    tokens = torch.zeros(*mask.shape, config['feature_dim'], dtype=torch.float16)
    tokens[mask] = output.half()
    return tokens, mask, coords


def rotary(x, position):
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError('Rotary head dimension must be even.')
    frequencies = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.) / dim))
    angle = position.float()[:,None,:,None] * frequencies
    a, b = x.float()[...,0::2], x.float()[...,1::2]
    return torch.stack((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()), dim=-1).flatten(-2).to(x.dtype)


class SliceBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))

    def forward(self, x, mask, position, use_rope):
        b, t, d = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        if use_rope:
            q,k = rotary(q,position),rotary(k,position)
        scores = (q.float() @ k.float().transpose(-1,-2)) / math.sqrt(d//self.heads)
        scores = scores.masked_fill(~mask[:,None,None,:], -1e4).softmax(-1)
        context = (scores.to(v.dtype) @ v).transpose(1,2).reshape(b,t,d)
        x = x + self.out(context)
        return (x + self.ffn(self.norm2(x))) * mask[...,None]


class SpatialHead(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__()
        self.config = config
        d = config['hidden_dim']
        if d % config['heads'] or (d//config['heads']) % 2:
            raise ValueError('Invalid attention dimensions.')
        self.register_buffer('mean', mean.float().clone())
        self.register_buffer('std', std.float().clone())
        self.project = nn.Linear(config['feature_dim'], d)
        self.pool_score = nn.Sequential(nn.Linear(d,64),nn.Tanh(),nn.Linear(64,12))
        self.class_weight = nn.Parameter(torch.empty(12,6*d))
        self.class_bias = nn.Parameter(torch.zeros(12))
        nn.init.normal_(self.class_weight,std=.01)
        self.coord_kind = nn.Embedding(2,d)
        nn.init.zeros_(self.coord_kind.weight)
        if config['sequence']:
            self.slice_block = SliceBlock(d,config['heads'])
        if config['plane_embedding']:
            self.plane = nn.Embedding(3,d)
            self.protocol = nn.Embedding(2,d)
        mode = config['fusion']
        if mode == 'mlp':
            self.mlp = nn.Sequential(nn.Linear(6*d,d),nn.GELU(),nn.Linear(d,1))
        elif mode == 'gate':
            self.gate = nn.Linear(d,1)
            self.gate_weight = nn.Parameter(torch.randn(12,d)*.01)
        elif mode == 'cross':
            self.queries = nn.Parameter(torch.randn(12,d)*.02)
            self.memory_norm, self.query_norm = nn.LayerNorm(d),nn.LayerNorm(d)
            self.cross = nn.MultiheadAttention(d,config['heads'],batch_first=True,dropout=0.)
            self.read_norm = nn.LayerNorm(d)
            self.read_ffn = nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
            self.read_weight = nn.Parameter(torch.randn(12,d)*.01)
        elif mode != 'concat':
            raise ValueError('Unsupported fusion.')

    def forward(self, tokens, mask, coords):
        if tokens.ndim != 4 or tokens.shape[1] != 6 or mask.shape != tokens.shape[:3] or coords.shape != (*mask.shape,3):
            raise ValueError('Invalid token/coordinate shape.')
        if not mask.flatten(1).any(1).all():
            raise ValueError('Each study requires a valid image.')
        coords = torch.where(mask[...,None],coords,0.)
        clean = torch.where(mask[...,None],tokens.float(),self.mean)
        if not torch.isfinite(coords).all() or not torch.isfinite(clean).all():
            raise ValueError('Non-finite valid token or coordinate.')
        x = self.project((clean-self.mean)/self.std)
        x = x + self.coord_kind((coords[...,2]>.5).long())
        b,s,t,d = x.shape
        if self.config['plane_embedding']:
            slot=torch.arange(6,device=x.device)
            x=x+(self.plane(slot//2)+self.protocol(slot%2))[None,:,None,:]
        pos=coords[...,0]
        if self.config['position']=='rope_mm':
            pos=torch.where(coords[...,2]>.5,coords[...,1]/self.config['mm_unit'],pos)
        if self.config['position']=='additive':
            freq=torch.exp(torch.arange(0,d,2,device=x.device)*(-math.log(10000.)/d))
            angles=pos[...,None]*freq
            x=x+torch.stack((angles.sin(),angles.cos()),-1).flatten(-2)
        x=x*mask[...,None]
        if self.config['sequence']:
            flat=x.reshape(b*s,t,d); flat_mask=mask.reshape(b*s,t); active=flat_mask.any(1)
            result=torch.zeros_like(flat)
            result[active]=self.slice_block(flat[active],flat_mask[active],pos.reshape(b*s,t)[active],self.config['position'].startswith('rope'))
            x=result.reshape(b,s,t,d)
        if self.config['fusion']=='cross':
            memory=self.memory_norm(x.flatten(1,2)); query=self.queries[None].expand(b,-1,-1)
            read,_=self.cross(self.query_norm(query),memory,memory,key_padding_mask=~mask.flatten(1),need_weights=False)
            read=query+read; read=read+self.read_ffn(self.read_norm(read))
            return (read*self.read_weight).sum(-1)+self.class_bias
        attention=self.pool_score(x).masked_fill(~mask[...,None],-1e4).softmax(2)*mask[...,None]
        pooled=torch.einsum('bstc,bstd->bscd',attention,x).permute(0,2,1,3)
        if self.config['fusion']=='mlp':
            return self.mlp(pooled.flatten(2)).squeeze(-1)+self.class_bias
        if self.config['fusion']=='gate':
            weight=self.gate(pooled).squeeze(-1).masked_fill(~mask.any(2)[:,None,:],-1e4).softmax(-1)
            return ((pooled*weight[...,None]).sum(2)*self.gate_weight).sum(-1)+self.class_bias
        return (pooled.flatten(2)*self.class_weight).sum(-1)+self.class_bias
