"""Physical preprocessing and coarse spatial tokens shared by train and submit."""
from experiments.spatial_core import *
from experiments.backbones import swin_t_encoder


def quadrant_pool(x):
    # Swin-T at 224 px produces a 7x7 map; unequal areas must not get equal weights.
    return torch.stack((x[:,:4,:4].mean((1,2)), x[:,:4,4:].mean((1,2)),
                        x[:,4:,:4].mean((1,2)), x[:,4:,4:].mean((1,2))),1)


def quadrant_mean(tokens):
    return (tokens * tokens.new_tensor([16/49,12/49,12/49,9/49])[:,None]).sum(-2)


class QuadrantSwin(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base=swin_t_encoder(pretrained=pretrained)
        self.features, self.norm = base.features, base.norm

    def forward(self, x):
        return quadrant_pool(self.norm(self.features(x)))


def physical_crop(pixels, spacing, field_mm, size):
    spacing=np.asarray(spacing,dtype=float)
    if (spacing.shape!=(2,) or not np.isfinite(spacing).all() or (spacing<=0).any()
            or not np.isfinite(field_mm) or field_mm<=0 or size<2):
        raise ValueError('A physical crop requires valid positive PixelSpacing and field size.')
    h,w=pixels.shape
    if min(h,w)<2: raise ValueError('MRI matrix is too small.')
    distance=torch.linspace(-field_mm/2,field_mm/2,size,device=pixels.device)
    gy,gx=torch.meshgrid(distance/(spacing[0]*(h-1)/2),distance/(spacing[1]*(w-1)/2),indexing='ij')
    grid=torch.stack((gx,gy),-1)[None].float()
    return torch.nn.functional.grid_sample(pixels[None,None].float(),grid,
        align_corners=True,padding_mode='zeros',mode='bilinear')[0,0]


def canonical_orientation(pixels, orientation, plane):
    # DICOM first cosine is the direction of increasing pixel COLUMN.
    axes={'Sagittal':([0,1,0],[0,0,-1]),'Coronal':([1,0,0],[0,0,-1]),'Axial':([1,0,0],[0,1,0])}
    target_col,target_row=map(np.asarray,axes[plane])
    iop=np.asarray(orientation,dtype=float)
    if (iop.shape!=(6,) or not np.isfinite(iop).all() or
            not np.allclose([np.linalg.norm(iop[:3]),np.linalg.norm(iop[3:])],1.,atol=.01)
            or abs(np.dot(iop[:3],iop[3:]))>.01):
        raise ValueError('Invalid or missing ImageOrientationPatient.')
    col,row=iop[:3],iop[3:]
    if abs(row@target_col)+abs(col@target_row)>abs(col@target_col)+abs(row@target_row):
        pixels=pixels.T; col,row=row,col
    dots=np.array([col@target_col,row@target_row])
    if np.abs(dots).min()<.5: raise ValueError('IOP is inconsistent with the anatomical plane.')
    if dots[0]<0: pixels=pixels.flip(1)
    if dots[1]<0: pixels=pixels.flip(0)
    residual=float(np.degrees(np.arccos(np.clip(np.abs(dots),0,1))).max())
    return pixels,residual


def anatomy_image(path, plane, config, audit=None, flip=False):
    import pydicom
    # Reuse the exact legacy percentile calculation, including constant-image rejection.
    # Header-only second read avoids decoding compressed pixels twice.
    pixels=image_tensor(path,lambda image:image)[0]
    dcm=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['PixelSpacing','ImageOrientationPatient'])
    residual=0.
    if config.get('field_mm'):
        # Resampling to a square physical field before orientation handles anisotropic spacing.
        pixels=physical_crop(pixels,getattr(dcm,'PixelSpacing',[]),config['field_mm'],config['crop'])
    if config.get('canonical'):
        pixels,residual=canonical_orientation(pixels,getattr(dcm,'ImageOrientationPatient',[]),plane)
    if flip: pixels=pixels.flip(1)
    image=pixels[None].repeat(3,1,1)
    if config.get('field_mm'):
        image=(image-image.new_tensor([.485,.456,.406])[:,None,None])/image.new_tensor([.229,.224,.225])[:,None,None]
    else: image=make_transform(config)(image)
    if audit is not None: audit.append(dict(path=str(path),plane=plane,residual_oblique_degrees=residual))
    return image


def anatomy_views(uid,series,image_root,config,audit=None,flip=False,limit=None):
    count=config['centers']*config['neighbors'] if limit is None else limit
    images=torch.zeros(6,count,3,config['crop'],config['crop']); mask=torch.zeros(6,count,dtype=torch.bool)
    coords=torch.zeros(6,count,3); rows=series.loc[series.StudyInstanceUID==uid]
    for slot,(plane,fs) in enumerate((p,f) for p in PLANES for f in (True,False)):
        choices=rows.loc[(rows.Anatomical_Plane==plane)&(rows.Fat_Suppression==fs)]
        choices=choices.sort_values(['Fluid_Sensitive','SeriesInstanceUID'],ascending=[False,True])
        if choices.empty: continue
        paths,positions=spatial_order(image_root/uid/choices.iloc[0].SeriesInstanceUID)
        selected=[w[0] for w in selected_windows(len(paths),config)]
        if limit and len(selected)>limit:
            selected=[selected[i] for i in np.rint(np.linspace(0,len(selected)-1,limit)).astype(int)]
        for j,i in enumerate(selected):
            images[slot,j]=anatomy_image(paths[i],plane,config,audit,flip)
            mask[slot,j]=True;coords[slot,j]=torch.from_numpy(positions[i])
    if not mask.any(): raise ValueError(f'No supported MRI series: {uid}')
    return images,mask,coords


@torch.inference_mode()
def anatomy_tokens(uid,series,image_root,encoder,config,device,audit=None,flip=False):
    images,mask,coords=anatomy_views(uid,series,image_root,config,audit,flip)
    values=torch.cat([encoder(x.to(device)).cpu().float() for x in images[mask].split(16)])
    if values.shape!=(int(mask.sum()),4,config['feature_dim']) or not torch.isfinite(values).all():
        raise ValueError('Invalid quadrant features.')
    tokens=torch.zeros(*mask.shape,4,config['feature_dim'],dtype=torch.float16);tokens[mask]=values.half()
    return tokens,mask,coords


class AnatomyHead(SpatialHead):
    def forward(self,tokens,mask,coords):
        if tokens.ndim!=5 or tokens.shape[-2]!=4: raise ValueError('Expected four spatial regions per slice.')
        if not self.config['patch_attention']:
            return super().forward(quadrant_mean(tokens.float()),mask,coords)
        if mask.shape!=tokens.shape[:3] or coords.shape!=(*mask.shape,3) or not mask.flatten(1).any(1).all():
            raise ValueError('Invalid masked study.')
        clean=torch.where(mask[...,None,None],tokens.float(),self.mean)
        coords=torch.where(mask[...,None],coords,0.)
        if not torch.isfinite(clean).all() or not torch.isfinite(coords).all(): raise ValueError('Invalid valid feature.')
        x=self.project((clean-self.mean)/self.std)
        b,s,t,p,d=x.shape
        # Fixed x/y sin-cos coordinates; no extra learned parameters versus the mean control.
        xy=x.new_tensor([[-1,-1],[1,-1],[-1,1],[1,1]])
        freq=torch.exp(torch.arange(0,d//2,2,device=x.device)*(-math.log(10000.)/(d//2)))
        angles=xy[:,:,None]*freq
        pe=torch.stack((angles.sin(),angles.cos()),-1).flatten(1)
        if pe.shape[-1]!=d: raise ValueError('Patch position encoding requires hidden_dim divisible by four.')
        slot=torch.arange(6,device=x.device)
        x=x+pe[None,None,None]+self.coord_kind((coords[...,2]>.5).long())[...,None,:]
        x=x+(self.plane(slot//2)+self.protocol(slot%2))[None,:,None,None,:]
        pos=torch.where(coords[...,2]>.5,coords[...,1]/self.config['mm_unit'],coords[...,0])
        pos=pos.repeat_interleave(4,-1); expanded=mask.repeat_interleave(4,-1)
        x=x.flatten(2,3)*expanded[...,None]
        flat=x.reshape(b*s,t*p,d); m=expanded.reshape(b*s,t*p); active=m.any(1); result=torch.zeros_like(flat)
        result[active]=self.slice_block(flat[active],m[active],pos.reshape(b*s,t*p)[active],True)
        memory=self.memory_norm(result.reshape(b,s*t*p,d)); query=self.queries[None].expand(b,-1,-1)
        read,_=self.cross(self.query_norm(query),memory,memory,key_padding_mask=~expanded.flatten(1),need_weights=False)
        read=query+read;read=read+self.read_ffn(self.read_norm(read))
        return (read*self.read_weight).sum(-1)+self.class_bias


def reference_cdf(reference,values):
    reference,values=np.asarray(reference),np.asarray(values)
    if reference.ndim!=2 or values.ndim!=2 or values.shape[1]!=reference.shape[1] or not len(reference):
        raise ValueError('CDF requires a nonempty fixed training reference with matching classes.')
    if not np.isfinite(reference).all() or not np.isfinite(values).all(): raise ValueError('Invalid CDF values.')
    result=np.empty_like(values,dtype=float)
    for c in range(values.shape[1]):
        sorted_values=np.sort(reference[:,c])
        rank=(np.searchsorted(sorted_values,values[:,c],'left')+np.searchsorted(sorted_values,values[:,c],'right'))/2
        result[:,c]=(rank+.5)/(len(reference)+1)
    return result
