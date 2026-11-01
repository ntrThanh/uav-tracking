"""Portable DET-training -> MOT detection-cache workflow, bundled into the Colab notebook.

Data conversion is owned by data/tools, not duplicated here. COCO metrics here are
unfiltered COCO bbox AP@500 for checkpoint selection, not the team's final ignored-
region-aware benchmark. All metrics are fractions, not percentages.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw


def extract_zip(archive: Path, destination: Path) -> None:
    """Validate all members before extraction; refuse path escapes and symlinks."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace('\\', '/')
            target = (destination / name).resolve()
            if (not target.is_relative_to(destination) or ':' in name
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        required = sum(m.file_size for m in z.infolist())
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination).free < required + 2 * 1024**3:
            raise ValueError('Not enough local disk space to extract this archive (+2 GB reserve).')
        z.extractall(destination)


def ensure_dataset(archives: Path, raw: Path, name: str) -> Path:
    """Copy one archive locally, extract once; a completion marker prevents partial reuse."""
    archive = Path(archives) / f'{name}.zip'
    if not archive.is_file():
        raise FileNotFoundError(f'Upload {name}.zip into {archives}')
    folder = Path(raw) / name
    marker = folder / '.colab_extracted.json'
    signature = {'zip_bytes': archive.stat().st_size, 'zip_mtime': archive.stat().st_mtime}
    if marker.is_file() and json.loads(marker.read_text()) == signature:
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f'{folder} is incomplete or came from a different archive. '
                         'Use a fresh Colab runtime/local workspace; Drive files are unaffected.')
    local_zip = Path(raw).parent / f'{name}.zip'
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive, local_zip)
    print(f'Extracting {name} to local disk ...', flush=True)
    extract_zip(local_zip, raw)
    image_dir = folder / ('images' if '-DET-' in name else 'sequences')
    if not image_dir.is_dir() or not (folder / 'annotations').is_dir():
        raise ValueError(f'ZIP must contain {name}/images (or sequences) and annotations directly.')
    local_zip.unlink()
    atomic_json(marker, signature)
    return folder


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_coco(coco: dict, raw: Path, expected_images: int | None = None) -> None:
    images = {i['id']: i for i in coco['images']}
    if not images or len(images) != len(coco['images']):
        raise ValueError('Empty dataset or duplicate image IDs')
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(f'Expected {expected_images} images, found {len(images)}')
    if sorted(c['id'] for c in coco['categories']) != list(range(1, 11)):
        raise ValueError('Expected VisDrone category IDs 1..10')
    root = Path(raw).resolve()
    for im in images.values():
        path = (root / im['file_name']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/unsafe image path: {im["file_name"]}')
    seen = set()
    for ann in coco['annotations']:
        if ann['id'] in seen or ann['image_id'] not in images or ann['category_id'] not in range(1, 11):
            raise ValueError(f'Invalid annotation: {ann}')
        seen.add(ann['id'])
        x, y, w, h = ann['bbox']
        im = images[ann['image_id']]
        if not all(math.isfinite(v) for v in [x, y, w, h]) or not (
                x >= 0 and y >= 0 and w > 0 and h > 0
                and x + w <= im['width'] + 1e-5 and y + h <= im['height'] + 1e-5):
            raise ValueError(f'Invalid/out-of-bounds box: {ann}')


def learning_rate(base_lr: float, epoch: int, step: int) -> float:
    # epoch is zero-based: decay after completing 16 and 22 epochs.
    warmup = 0.001 + 0.999 * min(step / 500, 1.0)
    return base_lr * warmup * 0.1 ** (int(epoch >= 16) + int(epoch >= 22))


def check_resume_config(saved: dict, current: dict) -> None:
    if saved != current:
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        raise ValueError(f'Run configuration changed: {changed}. Use a NEW run name to avoid mixing experiments.')


def cache_lines(frame: int, prediction: dict) -> list[str]:
    if frame < 1 or len({len(prediction[k]) for k in ('boxes', 'scores', 'labels')}) != 1:
        raise ValueError('Invalid frame or prediction lengths')
    lines = []
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        x1, y1, x2, y2 = map(float, box)
        score, label_value = float(score), float(label)
        if (not all(math.isfinite(v) for v in [x1, y1, x2, y2, score, label_value])
                or label_value != int(label_value) or int(label_value) not in range(1, 11)
                or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or not 0 <= score <= 1):
            raise ValueError('Invalid detection coordinates, score or class')
        lines.append(f'{frame},-1,{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},{score:.6f},{int(label_value)}\n')
    return lines


def coco_metrics(precision: np.ndarray, recall: np.ndarray, category_ids: list[int]) -> dict:
    """Read evaluated tensors directly: COCOeval.summarize's AP can assume maxDets=100."""
    def mean_valid(a):
        values = a[a >= 0]
        return float(values.mean()) if values.size else None
    return {
        'mAP50_95': mean_valid(precision[:, :, :, 0, 2]),
        'mAP50': mean_valid(precision[0, :, :, 0, 2]),
        'mAP75': mean_valid(precision[5, :, :, 0, 2]),
        'AP_small': mean_valid(precision[:, :, :, 1, 2]),
        'AP_medium': mean_valid(precision[:, :, :, 2, 2]),
        'AP_large': mean_valid(precision[:, :, :, 3, 2]),
        'AR_500': mean_valid(recall[:, :, 0, 2]),
        'per_class': {str(c): mean_valid(precision[:, :, k, 0, 2]) for k, c in enumerate(category_ids)},
    }


def evaluate_predictions(annotation: Path, predictions: list[dict], image_ids=None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(annotation))
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {'images': copy.deepcopy(gt.dataset['images']),
                      'categories': copy.deepcopy(gt.dataset['categories']), 'annotations': []}
        dt.createIndex()
    ev = COCOeval(gt, dt, 'bbox')
    ev.params.maxDets = [1, 100, 500]
    if image_ids is not None:
        ev.params.imgIds = list(image_ids)
    ev.evaluate()
    ev.accumulate()
    result = coco_metrics(ev.eval['precision'], ev.eval['recall'], ev.params.catIds)
    result['protocol'] = 'COCO bbox maxDets=[1,100,500]; no ignored-region filtering; fractions 0..1'
    return result


def make_dataset(annotation: Path, raw: Path, training: bool = False):
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    class CocoImages(torch.utils.data.Dataset):
        def __init__(self):
            self.coco = json.loads(Path(annotation).read_text())
            self.images = self.coco['images']
            self.by_image = defaultdict(list)
            for ann in self.coco['annotations']:
                self.by_image[ann['image_id']].append(ann)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            record = self.images[index]
            with Image.open(Path(raw) / record['file_name']) as im:
                image = pil_to_tensor(im.convert('RGB')).float().div_(255)
            anns = self.by_image[record['id']]
            boxes = torch.tensor([a['bbox'] for a in anns], dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            if training and random.random() < 0.5:
                image = image.flip(-1)
                boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
            target = {
                'boxes': boxes,
                'labels': torch.tensor([a['category_id'] for a in anns], dtype=torch.int64),
                'image_id': torch.tensor(record['id'], dtype=torch.int64),
                'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                'iscrowd': torch.zeros(len(anns), dtype=torch.int64),
            }
            return image, target
    return CocoImages()


def collate(batch):
    return tuple(zip(*batch))


def make_model(cfg: dict, pretrained: bool = True):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None, trainable_backbone_layers=5,
        min_size=cfg['min_size'], max_size=cfg['max_size'],
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=1000,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 11)
    return model


def seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict_dataset(model, dataset, batch_size: int, workers: int, amp: bool):
    import torch
    from tqdm.auto import tqdm
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=workers, collate_fn=collate, pin_memory=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc='Validation', leave=False):
            with torch.autocast('cuda', enabled=amp):
                outputs = model([im.cuda(non_blocking=True) for im in images])
            for target, output in zip(targets, outputs):
                boxes = output['boxes'].float().cpu().numpy()
                labels = output['labels'].cpu().tolist()
                scores = output['scores'].float().cpu().tolist()
                for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                    predictions.append({'image_id': int(target['image_id']), 'category_id': label,
                                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                        'score': score})
    return predictions


def save_checkpoint(path: Path, payload: dict):
    import torch
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def write_history(run_dir: Path, history: list[dict]):
    atomic_json(run_dir / 'history.json', history)
    if history:
        tmp = run_dir / 'history.csv.tmp'
        with open(tmp, 'w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        os.replace(tmp, run_dir / 'history.csv')


def finish_best_publication(run_dir: Path, state: dict):
    """last.pt is authoritative; finish an interrupted best-artifact publication.

    The marker is written only after weights, predictions and metrics are complete.
    An improving epoch retains its predictions inside last.pt so repair needs no GPU.
    """
    marker = run_dir / 'best_epoch.json'
    artifacts = ['best.pt', 'best_det_val_predictions.json', 'best_det_val_metrics.json']
    if (marker.is_file() and json.loads(marker.read_text()) == {'epoch': state['best_epoch']}
            and all((run_dir / name).is_file() for name in artifacts)):
        return
    if state['epoch'] != state['best_epoch'] or state['best_predictions'] is None:
        raise ValueError('Best artifacts were removed after a later epoch. Restore the complete run folder.')
    save_checkpoint(run_dir / 'best.pt', {
        'model': state['model'], 'config': state['config'], 'epoch': state['epoch'],
        'metrics': state['metrics'], 'best_ap': state['best_ap'],
    })
    atomic_json(run_dir / 'best_det_val_predictions.json', state['best_predictions'])
    atomic_json(run_dir / 'best_det_val_metrics.json', state['metrics'])
    atomic_json(marker, {'epoch': state['best_epoch']})


def train(cfg: dict, raw: Path, train_ann: Path, val_ann: Path, run_dir: Path):
    """Resume at an epoch boundary. Never silently mix configuration/data/environment."""
    import torch
    import torchvision
    from tqdm.auto import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training.')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(cfg, train_sha256=sha256(train_ann), val_sha256=sha256(val_ann),
                    workflow_sha256=sha256(Path(__file__)), torch=torch.__version__, torchvision=torchvision.__version__)
    config_file = run_dir / 'run_config.json'
    if config_file.exists():
        check_resume_config(json.loads(config_file.read_text()), protocol)
    atomic_json(config_file, protocol)
    env_file = run_dir / 'environment.txt'
    if not env_file.exists():
        env_file.write_text(f'GPU: {torch.cuda.get_device_name(0)}\nPython: {sys.version}\n' +
                           subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf8')
    seed_all(cfg['seed'])
    training = make_dataset(train_ann, raw, training=True)
    validation = make_dataset(val_ann, raw)
    generator = torch.Generator()
    loader = torch.utils.data.DataLoader(training, batch_size=cfg['batch_size'], shuffle=True,
                                        num_workers=cfg['workers'], collate_fn=collate,
                                        pin_memory=True, generator=generator)
    last_path = run_dir / 'last.pt'
    best_path = run_dir / 'best.pt'
    state = torch.load(last_path, map_location='cpu', weights_only=False) if last_path.exists() else None
    if state:
        check_resume_config(state['config'], protocol)
        finish_best_publication(run_dir, state)
    elif best_path.exists():
        raise ValueError('best.pt exists without last.pt. Restore last.pt or choose a new RUN_NAME.')
    model = make_model(cfg, pretrained=state is None).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    start_epoch, global_step, best_ap, best_epoch, history = 0, 0, -1.0, -1, []
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, global_step = state['epoch'] + 1, state['global_step']
        best_ap, best_epoch, history = state['best_ap'], state['best_epoch'], state['history']
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        torch.cuda.set_rng_state_all(state['cuda_state'])
        print(f'Resuming after epoch {start_epoch}; best AP@500={best_ap:.4f}')
        del state
    write_history(run_dir, history)
    for epoch in range(start_epoch, cfg['epochs']):
        generator.manual_seed(cfg['seed'] + epoch)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        sums = defaultdict(float)
        image_count = 0
        bar = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}')
        for images, targets in bar:
            lr = learning_rate(cfg['lr'], epoch, global_step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            images = [im.cuda(non_blocking=True) for im in images]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=cfg['amp']):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch+1}, step {global_step}: {losses}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                sums[key] += float(value.detach()) * len(images)
            image_count += len(images)
            global_step += 1
            bar.set_postfix(loss=f'{float(loss.detach()):.3f}', lr=f'{lr:.6g}')
        preds = predict_dataset(model, validation, cfg['eval_batch_size'], cfg['workers'], cfg['amp'])
        metrics = evaluate_predictions(val_ann, preds)
        score = metrics['mAP50_95']
        if score is None or not math.isfinite(score):
            raise RuntimeError('Validation mAP is undefined; inspect dataset and evaluation.')
        elapsed = time.perf_counter() - started
        row = {'epoch': epoch + 1, 'train_loss': sum(sums.values()) / image_count,
               **{k: sums[k] / image_count for k in sorted(sums)},
               **{k: v for k, v in metrics.items() if k not in ('per_class', 'protocol')},
               'lr': lr, 'seconds': elapsed, 'vram_peak_gb': torch.cuda.max_memory_allocated() / 1024**3}
        history.append(row)
        improved = score > best_ap
        best_ap = max(score, best_ap)
        if improved:
            best_epoch = epoch
        # Saving on CPU avoids duplicating checkpoint tensors on GPU during resume.
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epoch_state = {
            'model': weights, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'config': protocol, 'epoch': epoch, 'global_step': global_step, 'best_ap': best_ap,
            'best_epoch': best_epoch, 'metrics': metrics, 'best_predictions': preds if improved else None,
            'history': history, 'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(), 'cuda_state': torch.cuda.get_rng_state_all(),
        }
        # Commit the resumable epoch first. Best artifacts can be repaired from this state.
        save_checkpoint(last_path, epoch_state)
        finish_best_publication(run_dir, epoch_state)
        del epoch_state
        write_history(run_dir, history)
        atomic_json(run_dir / 'status.json', {'completed_epochs': epoch + 1, 'planned_epochs': cfg['epochs'],
                                            'best_ap': best_ap, 'last_epoch_seconds': elapsed})
        print(f'Saved epoch {epoch+1}: AP@500={score:.4f}, AP50={metrics["mAP50"]:.4f}, '
              f'{elapsed/60:.1f} min. Best={best_ap:.4f}', flush=True)
    return model, history


def load_best(run_dir: Path):
    import torch
    latest = torch.load(Path(run_dir) / 'last.pt', map_location='cpu', weights_only=False)
    finish_best_publication(Path(run_dir), latest)
    del latest
    state = torch.load(Path(run_dir) / 'best.pt', map_location='cpu', weights_only=False)
    model = make_model(state['config'], pretrained=False).cuda().eval()
    model.load_state_dict(state['model'])
    return model, state


def plot_results(run_dir: Path, annotation: Path, raw: Path, class_names: dict):
    """Save train curve, per-class AP and six fixed DET-val GT/prediction comparisons."""
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    out = run_dir / 'figures'
    out.mkdir(exist_ok=True)
    history = json.loads((run_dir / 'history.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r['train_loss'] for r in history])
    axes[0].set(title='Faster R-CNN: train loss', xlabel='Epoch', ylabel='Loss')
    for key in ('mAP50_95', 'mAP50', 'AP_small'):
        axes[1].plot(epochs, [r[key] for r in history], label=key)
    axes[1].set(title='DET val: COCO AP@500 (unfiltered)', xlabel='Epoch', ylabel='AP (0–1)')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'train_curve_frcnn_r50.png', dpi=160)
    plt.show()
    metrics = json.loads((run_dir / 'best_det_val_metrics.json').read_text())
    fig, ax = plt.subplots(figsize=(11, 4))
    values = [metrics['per_class'][str(k)] for k in class_names]
    ax.bar(list(class_names.values()), [v if v is not None else 0 for v in values])
    ax.set(title='DET val AP per class (absent classes shown as 0)', ylabel='AP@500 (0–1)')
    ax.tick_params(axis='x', rotation=40)
    fig.tight_layout()
    fig.savefig(out / 'ap_per_class_frcnn_r50.png', dpi=160)
    plt.show()
    coco = json.loads(Path(annotation).read_text())
    gt, pred = defaultdict(list), defaultdict(list)
    for ann in coco['annotations']:
        gt[ann['image_id']].append(ann)
    for ann in json.loads((run_dir / 'best_det_val_predictions.json').read_text()):
        if ann['score'] >= 0.4:
            pred[ann['image_id']].append(ann)
    picks = random.Random(0).sample(coco['images'], min(6, len(coco['images'])))
    atomic_json(out / 'det_val_preview_frames.json', picks)
    for index, record in enumerate(picks, 1):
        with Image.open(Path(raw) / record['file_name']) as image:
            image = image.convert('RGB')
        panels = []
        for title, annotations in [('GT', gt[record['id']]), ('Prediction conf >= 0.4', pred[record['id']])]:
            panel = image.copy()
            draw = ImageDraw.Draw(panel)
            for ann in annotations:
                x, y, w, h = ann['bbox']
                color = tuple(int(v * 255) for v in plt.get_cmap('tab10')(ann['category_id'] - 1)[:3])
                draw.rectangle((x, y, x+w, y+h), outline=color, width=3)
                draw.text((x, max(0, y-12)), class_names[ann['category_id']], fill=color)
            panel.thumbnail((1000, 700))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 24), fill='black')
            draw.text((5, 5), title, fill='white')
            panels.append(panel)
        sheet = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), 'white')
        sheet.paste(panels[0], (0, 0))
        sheet.paste(panels[1], (panels[0].width, 0))
        sheet.save(out / f'det_val_{index:02d}.jpg')
    return out


def export_mot(model, raw: Path, split: str, run_dir: Path, amp: bool = True):
    """All frames, all 10 classes; no ignored-region filtering or tracking here."""
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    from tqdm.auto import tqdm
    if split not in ('val', 'test-dev'):
        raise ValueError('Only MOT val or test-dev can be exported')
    sequences = Path(raw) / f'VisDrone2019-MOT-{split}' / 'sequences'
    seq_dirs = sorted(p for p in sequences.iterdir() if p.is_dir())
    expected = {'val': (7, 2846), 'test-dev': (17, 6635)}[split]
    frame_lists = {s.name: sorted(s.glob('*.jpg')) for s in seq_dirs}
    if (len(seq_dirs), sum(map(len, frame_lists.values()))) != expected:
        raise ValueError(f'Incomplete MOT {split}; expected {expected[0]} sequences, {expected[1]} frames')
    for name, frames in frame_lists.items():
        if [int(p.stem) for p in frames] != list(range(1, len(frames)+1)):
            raise ValueError(f'Non-contiguous frame numbers in {name}')
    out = Path(run_dir) / 'results/dets/frcnn_r50' / split
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(Path(run_dir) / 'best.pt')
    manifest_path = out / 'manifest.json'
    protocol = {'checkpoint_sha256': fingerprint, 'split': split, 'score': 0.05, 'nms': 0.5,
                'max_det': 1000, 'amp': amp, 'class_ids': 'VisDrone 1..10', 'batch_size': 1,
                'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        check_resume_config(manifest['protocol'], protocol)
    else:
        manifest = {'protocol': protocol, 'sequences': {}}
        atomic_json(manifest_path, manifest)
    model.eval()
    # Warm-up is excluded from timings; read/convert/H2D is outside detector timing.
    first = frame_lists[seq_dirs[0].name][0]
    with Image.open(first) as im:
        tensor = pil_to_tensor(im.convert('RGB')).float().div_(255).cuda()
    with torch.inference_mode(), torch.autocast('cuda', enabled=amp):
        for _ in range(50):
            model([tensor])
    torch.cuda.synchronize()
    for seq in seq_dirs:
        final = out / f'{seq.name}.txt'
        old = manifest['sequences'].get(seq.name)
        if old and final.is_file() and sha256(final) == old['sha256']:
            print('Already exported:', seq.name)
            continue
        elapsed_det, elapsed_pipeline = [], []
        frame_counts = []
        temporary = final.with_suffix('.txt.tmp')
        with open(temporary, 'w', encoding='utf8') as f, torch.inference_mode():
            for path in tqdm(frame_lists[seq.name], desc=f'MOT {split}/{seq.name}'):
                pipeline_start = time.perf_counter()
                with Image.open(path) as im:
                    tensor = pil_to_tensor(im.convert('RGB')).float().div_(255).cuda()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast('cuda', enabled=amp):
                    result = model([tensor])[0]
                torch.cuda.synchronize()
                elapsed_det.append(time.perf_counter() - started)
                result = {k: v.float().cpu().tolist() if k != 'labels' else v.cpu().tolist()
                          for k, v in result.items()}
                lines = cache_lines(int(path.stem), result)
                f.writelines(lines)
                frame_counts.append([int(path.stem), len(lines)])
                elapsed_pipeline.append(time.perf_counter() - pipeline_start)
        os.replace(temporary, final)
        record = {'sha256': sha256(final), 'n_frames': len(frame_counts),
                  'n_detections': sum(n for _, n in frame_counts), 'frame_detection_counts': frame_counts,
                  'det_seconds': sum(elapsed_det), 'pipeline_seconds': sum(elapsed_pipeline),
                  'det_ms_mean': 1000 * float(np.mean(elapsed_det)),
                  'det_ms_p50': 1000 * float(np.percentile(elapsed_det, 50)),
                  'det_ms_p95': 1000 * float(np.percentile(elapsed_det, 95))}
        manifest['sequences'][seq.name] = record
        atomic_json(manifest_path, manifest)
    records = list(manifest['sequences'].values())
    n = sum(r['n_frames'] for r in records)
    det_seconds = sum(r['det_seconds'] for r in records)
    timing = dict(protocol, n_frames=n, fps_detector=n/det_seconds,
                  detector_ms_per_frame=1000*det_seconds/n,
                  fps_read_detect_write=n/sum(r['pipeline_seconds'] for r in records),
                  note='One export pass; not the final 3-repeat benchmark. Excludes tracker/ReID/render. '
                       'Detector time includes internal resize, forward and NMS; excludes read and H2D.')
    atomic_json(out / 'timing.json', timing)
    archive = shutil.make_archive(str(out.parent / f'{split}_cache'), 'zip', root_dir=out)
    print(f'Exported {n} frames. Cache ZIP: {archive}')
    return out
