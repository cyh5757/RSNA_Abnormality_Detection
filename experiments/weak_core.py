"""Report-supervised MRI experiments; shared verbatim by train and offline submit."""
import re
from experiments.core import *


def make_targets(train, labels, config):
    train = train.sort_values('StudyInstanceUID').reset_index(drop=True).copy()
    if (train.StudyInstanceUID.isna().any() or not train.StudyInstanceUID.is_unique
            or labels.StudyInstanceUID.isna().any() or not labels.StudyInstanceUID.is_unique
            or not set(labels.StudyInstanceUID) <= set(train.StudyInstanceUID)):
        raise ValueError('Invalid, duplicate or foreign study IDs in label source.')
    required = [x + suffix for x in LABELS for suffix in ('', '__conf', '__verdict')]
    if not set(required) <= set(labels.columns):
        raise ValueError('Missing report label columns.')
    values = labels[[*LABELS, *[x+'__conf' for x in LABELS]]].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Weak scores/confidences must be finite values in [0,1].')
    if not labels[[x+'__verdict' for x in LABELS]].isin(['YES', 'NO', 'UNK']).all().all():
        raise ValueError('Unknown report verdict.')
    observed = train[LABELS].notna()
    if (observed.any(axis=1) != observed.all(axis=1)).any():
        raise ValueError('Unexpected partially gold-labelled study.')
    gold = observed.all(axis=1).to_numpy()
    gold_y = train.loc[gold, LABELS].to_numpy(dtype=np.float32)
    if not np.isin(gold_y, [0, 1]).all():
        raise ValueError('Gold targets must be binary.')
    source = labels.set_index('StudyInstanceUID').reindex(train.StudyInstanceUID)
    y = source[LABELS].fillna(.5).to_numpy(dtype=np.float32)
    conf = source[[x+'__conf' for x in LABELS]].fillna(0).to_numpy(dtype=np.float32)
    known = source[[x+'__verdict' for x in LABELS]].isin(['YES', 'NO']).to_numpy()
    w = ((.25 + .75 * conf) * known).astype(np.float32)
    y[gold], w[gold] = gold_y, 3.
    reports = train.Report.fillna('').map(lambda s: re.sub(r'\s+', ' ', str(s)).strip().casefold())
    if reports.eq('').any():
        raise ValueError('Missing report: cannot audit report-group leakage.')
    groups = reports.map(lambda s: hashlib.sha256(s.encode()).hexdigest()).to_numpy()
    if len(set(groups[gold])) != int(gold.sum()):
        raise ValueError('Duplicate gold reports require grouped gold CV.')
    overlap = ~gold & np.isin(groups, groups[gold])
    keep = gold | (config['weak_labels'] & ~overlap & (w.sum(1) > 0))
    audit = {'gold_studies': int(gold.sum()), 'weak_studies': int((keep & ~gold).sum()),
             'weak_gold_report_overlap': int(overlap.sum()),
             'weak_without_known_targets': int((~gold & (w.sum(1) == 0)).sum()),
             'missing_source_rows': int(source.ACL.isna().sum()),
             'known_weak_targets': {name: int(((w[:, i] > 0) & keep & ~gold).sum())
                                    for i, name in enumerate(LABELS)},
             'unknown_policy': 'zero loss weight; never assume a negative',
             'teacher_gold_independence': 'not established; development CV only'}
    # Filter supervision AFTER fixing the population, keeping the MRI/cache IDs identical.
    if 'weak_min_confidence' in config:
        threshold = config['weak_min_confidence']
        if not 0 <= threshold <= 1: raise ValueError('Invalid confidence threshold.')
        removed = keep[:, None] & ~gold[:, None] & known & (conf < threshold)
        w[removed] = 0.
        verdicts = source[[x+'__verdict' for x in LABELS]].to_numpy()
        audit['confidence_filter'] = dict(threshold=threshold,
            removed_yes={k:int((removed[:,i] & (verdicts[:,i]=='YES')).sum()) for i,k in enumerate(LABELS)},
            removed_no={k:int((removed[:,i] & (verdicts[:,i]=='NO')).sum()) for i,k in enumerate(LABELS)},
            remaining_weak_weight={k:float(w[keep & ~gold,i].sum()) for i,k in enumerate(LABELS)},
            zero_supervision_weak_studies=int((keep & ~gold & (w.sum(1)==0)).sum()))
    return train.loc[keep].reset_index(drop=True), y[keep], w[keep], gold[keep], groups[keep], audit


def find_report_labels(config, input_root=Path('/kaggle/input')):
    paths = [input_root / 'rsna-knee-llm-labels/report_labels_v2.csv',
             input_root / 'datasets/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv']
    found = [p for p in paths if p.is_file()]
    if len(found) != 1 or file_hash(found[0]) != config['labels_sha256']:
        raise ValueError('Attach the pinned Pilkwang report label dataset; CSV hash must match.')
    return found[0]


def find_training_run(experiment_id, input_root=Path('/kaggle/input')):
    slug = 'rsna-knee-' + experiment_id.replace('_', '-') + '-train'
    paths = [input_root / slug / 'run', input_root / 'notebooks/choiyoungwhan' / slug / 'run',
             input_root / 'kernels/choiyoungwhan' / slug / 'run']
    found = [p for p in paths if (p / 'bundle/assets.json').is_file()]
    if len(found) != 1:
        raise ValueError(f'Attach exactly the pinned training output of {experiment_id}.')
    meta = json.loads((found[0] / 'bundle/assets.json').read_text())
    if meta['experiment']['id'] != experiment_id:
        raise ValueError('Dependency experiment mismatch.')
    return found[0]


def view_signature(config):
    return {key: config[key] for key in ('slices', 'triplet', 'protocols', 'crop', 'resize',
                                        'interpolation', 'pipeline_sha256')}


def study_views(uid, series, image_root, transform, config):
    slots = [(plane, fs) for plane in PLANES for fs in (True, False)] if config['protocols'] else [
        (plane, None) for plane in PLANES]
    count, size = config['slices'], config['crop']
    images = torch.zeros(len(slots), count, 3, size, size)
    mask = torch.zeros(len(slots), count, dtype=torch.bool)
    positions = torch.zeros(len(slots), count)
    rows = series.loc[series.StudyInstanceUID == uid]
    for slot, (plane, fs) in enumerate(slots):
        choices = rows.loc[rows.Anatomical_Plane == plane]
        if fs is not None:
            choices = choices.loc[choices.Fat_Suppression == fs]
        choices = choices.sort_values(['Fluid_Sensitive', 'Fat_Suppression', 'SeriesInstanceUID'],
                                      ascending=[False, False, True])
        if choices.empty:
            continue
        paths, coords = ordered_slices(image_root / uid / choices.iloc[0].SeriesInstanceUID)
        centers = np.unique(np.rint(np.linspace(.25, .75, count) * (len(paths)-1)).astype(int))
        decoded = {}
        for j, center in enumerate(centers):
            if config['triplet']:
                indices = np.clip([center-1, center, center+1], 0, len(paths)-1)
                for index in indices:
                    if index not in decoded:
                        decoded[index] = image_tensor(paths[index], lambda x: x)[0]
                # Each channel remains inside this series; never bridge two protocol slots.
                images[slot, j] = transform(torch.stack([decoded[index] for index in indices]))
            else:
                images[slot, j] = image_tensor(paths[center], transform)
            mask[slot, j], positions[slot, j] = True, float(coords[center])
    if not mask.any():
        raise ValueError(f'No supported MRI input for {uid}')
    return images, mask, positions


@torch.inference_mode()
def weak_tokens(uid, series, image_root, encoder, transform, config, device):
    images, mask, positions = study_views(uid, series, image_root, transform, config)
    valid = images[mask]
    features = torch.cat([encoder(chunk.to(device)).cpu().float() for chunk in valid.split(16)])
    if features.shape != (int(mask.sum()), config['feature_dim']) or not torch.isfinite(features).all():
        raise ValueError('Invalid encoder output.')
    result = torch.zeros(*mask.shape, config['feature_dim'])
    result[mask] = features
    return result, mask, positions


class WeakHead(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__()
        slots, dim = (6 if config['protocols'] else 3), config['feature_dim']
        self.mode = config['aggregation']
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())
        if self.mode == 'mean':
            self.classifier = nn.Linear(slots * dim, 12)
        elif self.mode == 'class_mil':
            self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 12))
            self.class_weight = nn.Parameter(torch.empty(12, slots * dim))
            self.class_bias = nn.Parameter(torch.zeros(12))
            nn.init.normal_(self.class_weight, std=.01)
        else:
            raise ValueError('Unsupported weak experiment head.')

    def forward(self, tokens, mask):
        clean = torch.where(mask[..., None], tokens, self.mean)
        x = (clean - self.mean) / self.std
        if self.mode == 'mean':
            pooled = x.sum(2) / mask.sum(2).clamp_min(1)[..., None]
            return self.classifier(pooled.flatten(1))
        attention = self.attention(x).masked_fill(~mask[..., None], -1e4).softmax(2)
        attention = attention * mask[..., None]
        pooled = torch.einsum('bskc,bskd->bscd', attention, x)
        return (pooled.permute(0, 2, 1, 3).flatten(2) * self.class_weight).sum(-1) + self.class_bias


def weighted_bce(logits, targets, weights):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def per_label_weighted_bce(logits, targets, weights):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    denominator = weights.sum(0)
    valid = denominator > 0
    values = (loss * weights).sum(0) / denominator.clamp_min(1e-8)
    return (values * valid).sum() / valid.sum().clamp_min(1)


def fit_weak_head(tokens, mask, y, weights, train_index, seed, config):
    torch.manual_seed(seed)
    index = torch.as_tensor(train_index, dtype=torch.long, device=tokens.device)
    statistics = tokens[index][mask[index]]
    if not torch.isfinite(statistics).all() or len(statistics) < 2:
        raise ValueError('Invalid training features.')
    mean, std = statistics.mean(0), statistics.std(0).clamp_min(.01)
    del statistics
    head = WeakHead(config, mean, std).to(tokens.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    history = []
    for epoch in range(1, config['epochs'] + 1):
        order = index[torch.randperm(len(index), device=index.device)]
        total_loss = total_weight = 0.
        for batch in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_bce(head(tokens[batch], mask[batch]), y[batch], weights[batch])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            batch_weight = weights[batch].sum().item()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight
        history.append({'epoch': epoch, 'train_loss': total_loss / total_weight,
                        'optimizer_steps': math.ceil(len(index) / config['batch_size'])})
    return head.eval(), history


def validate_weak_bundle(path, config):
    path = Path(path).resolve()
    manifest = json.loads((path / 'assets.json').read_text())
    if manifest['experiment'] != config or manifest['labels'] != LABELS:
        raise ValueError('Wrong weak experiment bundle.')
    required = ({f'members/{eid}/assets.json' for eid in config['train_dependencies']}
                if config.get('ensemble') else {'encoder.ts', 'model.pt'})
    files = manifest['files']
    wheels = {p.relative_to(path).as_posix() for p in (path / 'wheels').glob('*.whl')}
    if (not required <= files.keys() or len(wheels) != 2 or not wheels <= files.keys()
            or not any(x.startswith('wheels/pydicom-3.0.2-') for x in wheels)
            or not any(x.startswith('wheels/python_gdcm-3.2.6-') for x in wheels)):
        raise ValueError('Missing weights or offline DICOM dependencies.')
    for name, digest in files.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or not target.is_file() or file_hash(target) != digest:
            raise ValueError(f'Invalid bundle artifact: {name}')
    if config.get('ensemble'):
        for eid in config['train_dependencies']:
            child = (path / 'members' / eid).resolve()
            if not child.is_relative_to(path): raise ValueError('Invalid member path.')
            child_config = json.loads((child / 'assets.json').read_text())['experiment']
            if child_config['id'] != eid: raise ValueError('Wrong ensemble member.')
            validate_weak_bundle(child, child_config)
    return manifest
