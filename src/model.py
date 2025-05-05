import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseSequential


# -------------------------------------------------------------------------
#  Open3D visualisation helper (prediction + optional ground-truth overlay)
# -------------------------------------------------------------------------
def show_sparse_tensor_as_pointcloud(
    st: spconv.SparseConvTensor,
    vbatch_index: int = 0,
    batch = None,
    prob_channel: int = 0,
    thresh: float = 0.0,
):
    """
    Visualise one virtual-batch item inside a SparseConvTensor.

    • Without *batch*:  green if pred_logit ≥ thresh else red.
    • With    *batch*:  colour by ground truth (green=propeller, red=background)
                       and modulate brightness by sigmoid(pred_logit).

    Args
    ----
    st            : SparseConvTensor (before or after the network)
    vbatch_index  : which virtual-batch item to show
    batch         : the list you passed to forward_batch; optional
    prob_channel  : feature channel holding the network score
    thresh        : decision threshold in logit space
    """
    import open3d as o3d
    import numpy as np
    import torch

    mask = st.indices[:, 0] == vbatch_index
    if not mask.any():
        print(f"No voxels for vbatch {vbatch_index}")
        return

    coords = st.indices[mask][:, 1:].cpu().numpy().astype(np.float32)
    logits = st.features[mask][:, prob_channel].cpu()

    # Default colour by prediction
    pred_is_pos = logits >= thresh
    reds   = torch.tensor([1.0, 0.0, 0.0])
    greens = torch.tensor([0.0, 1.0, 0.0])
    colors = torch.where(pred_is_pos[:, None], greens, reds).clone()

    # ─── overlay ground-truth if provided ─────────────────────────
    if batch is not None:
        # recreate the vbatch_size logic from combine_voxel_grids_to_sparse_tensor
        vbatch_size = max(len(b["voxel_grids"]) for b in batch)
        b_idx = vbatch_index // vbatch_size
        g_idx = vbatch_index %  vbatch_size
        if b_idx < len(batch) and g_idx < len(batch[b_idx]["target"]):
            tgt = batch[b_idx]["target"][g_idx].cpu().numpy().astype(bool)
            # tgt has same length/order as coords because loader keeps them aligned
            colors = np.where(
                tgt[:, None],
                np.array([0.0, 1.0, 0.0]),   # green for propeller voxels
                np.array([1.0, 0.0, 0.0])    # red  for background
            )
            # desaturate by prediction confidence for visual insight
            probs = torch.sigmoid(logits).numpy()[:, None]
            colors = 0.3 * colors + 0.7 * probs  # brighter if network confident

    # ─── Open3D point cloud ───────────────────────────────────────
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(coords)
    pcd.colors  = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])



def block(in_ch, out_ch, indice):
    return SparseSequential(
        spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class SpMiniUNet(nn.Module):
    def __init__(
        self,
        in_channels=2,          # ← default to 2 now
        base_channels=16,
        out_channels=8,
    ):
        super().__init__()
        B = base_channels

        # encoder
        self.enc1 = block(in_channels, B, "enc1")
        self.down1 = spconv.SparseConv3d(B, 2*B, kernel_size=2, stride=2, bias=False, indice_key="down1")
        self.enc2 = block(2*B, 2*B, "enc2")

        # decoder
        self.up1   = spconv.SparseInverseConv3d(2*B, B, kernel_size=2, indice_key="down1")
        self.dec1  = block(2*B, B, "dec1")  # concat skip

        self.out   = spconv.SubMConv3d(B, out_channels, 1, bias=True, indice_key="out")

    def forward(self, x):
        skip1 = self.enc1(x)            # (N, B)
        x     = self.down1(skip1)       # stride-2
        x     = self.enc2(x)            # (N/8, 2B)

        x     = self.up1(x)             # back to skip res
        # concatenate features along channel axis
        # (coords are equal because we used the same indice_key)
        assert torch.equal(x.indices, skip1.indices)
        x = x.replace_feature(torch.cat([x.features, skip1.features], dim=1))
        x     = self.dec1(x)
        x     = self.out(x)

        return x


# convenient wrapper matching the “forward_batch” pattern in the big repo
def combine_voxel_grids_to_sparse_tensor(batch):
    """
    Very small helper: concatenates all voxel grids from `batch` into
    a single SparseConvTensor with virtual-batch indices.
    """
    coords, feats = [], []
    vbatch_size = max(len(b["voxel_grids"]) for b in batch)
    for b_idx, item in enumerate(batch):
        for g_idx, grid in enumerate(item["voxel_grids"]):
            vb = b_idx * vbatch_size + g_idx
            vb_col = torch.full((len(grid), 1), vb, dtype=torch.int32)
            coords.append(torch.cat([vb_col, torch.from_numpy(grid).int()], dim=1))
            feats.append(torch.from_numpy(item["voxel_feats"][g_idx]).float())
    coords = torch.cat(coords, 0).cuda()
    feats  = torch.cat(feats, 0).cuda()

    shape = coords[:, 1:].max(0).values + 1  # quick bound
    return spconv.SparseConvTensor(feats.cuda(), coords, shape.tolist(), coords[:, 0].max().item() + 1)


class SpMiniUNetWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = SpMiniUNet(
            in_channels=2,
            base_channels=cfg.model.base_channels,
            out_channels=cfg.model.out_channels,
        )

    def forward_batch(self, batch):
        st = combine_voxel_grids_to_sparse_tensor(batch)
        out = self.net(st)

        feats_out_batch = []
        vbatch = max(len(b["voxel_grids"]) for b in batch)
        for b_idx, item in enumerate(batch):
            feats_in_this_item = []
            for g_idx in range(len(item["voxel_grids"])):
                vb = b_idx * vbatch + g_idx
                mask = out.indices[:, 0] == vb
                feats_in_this_item.append(out.features[mask])
            feats_out_batch.append(feats_in_this_item)
        return feats_out_batch
