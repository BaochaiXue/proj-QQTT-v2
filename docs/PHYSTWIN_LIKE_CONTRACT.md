# PhysTwin-Like Tracking Product Contract

This branch can generate a PhysTwin-compatible tracking product from the
single-camera Demo 3.2 stack without replacing the local model stack.

## Component Mapping

| PhysTwin role | Single-camera implementation |
| --- | --- |
| CoTracker3 Online | TAPNext++ |
| PhysTwin mask pipeline | EdgeTAM object/controller masks |
| RGB-D point cloud | RealSense or FFS depth lifted to world space |
| Track filtering | PhysTwin object/controller filtering rules |
| Sampling | Controller FPS 30 and object 5 mm grid sampling |

The compatibility target is the data contract and algorithm semantics, not
numerical equality with official PhysTwin model outputs.

## Demo 3.2 Strict Product Mode

Use:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --render-mode none \
  --headless-capture-dir result/single_demo_v3_2_ffs_masked_pcd/strict_capture \
  --tracking-product-backend phystwin-strict-tracking
```

P0 is a headless finite-window workstation generator. Live panel and pointcloud
rendering remain provisional realtime overlays.

## Querying

Strict product mode initializes TAPNext++ queries from the first frame only:

```text
EdgeTAM object mask | EdgeTAM controller mask
→ enumerate foreground pixels
→ random sample at most 5000
→ export [t=0, x, y]
→ track online with TAPNext++
→ export tracks [T,N,2] in y,x plus visibility [T,N]
```

Strict querying does not use residual PCD filters, table-Z filtering, neighbor
filters, FFS depth validity, or once-false marker retirement.

## Artifact Layout

The strict product output defaults to:

```text
<headless_capture_dir>/phystwin_like/
```

It contains:

```text
manifest.json
mask/processed_masks.pkl
tracking/0.npz
cotracker/0.npz
pcd/<frame_idx>.npz
track_process_data.pkl
final_data.pkl
tracking_2d.mp4
track_process_data.mp4
final_data.mp4
final_pcd.mp4
```

`cotracker/0.npz` is a compatibility path name. The manifest records
`not_actual_cotracker=true` and `tracker_backend=tapnextpp`.

## Required Fields

`processed_masks.pkl` uses:

```python
processed_masks[frame_idx][0]["object"]
processed_masks[frame_idx][0]["controller"]
```

When two-hand identity is available, `hand_a` and `hand_b` may also be present;
the compatibility controller mask is `hand_a | hand_b`.

`tracking/0.npz` and `cotracker/0.npz` contain:

```python
tracks      # [T,N,2], y,x
visibility  # [T,N]
queries_txy # [N,3], [0,x,y]
```

`pcd/<frame_idx>.npz` contains:

```python
points  # [1,H,W,3], world-space xyz
colors  # [1,H,W,3], RGB uint8
```

`track_process_data.pkl` and `final_data.pkl` contain:

```python
object_points
object_colors
object_visibilities
object_motions_valid
controller_points
controller_mask
```

`track_process_data.pkl` is after controller whole-window filtering and FPS 30.
`final_data.pkl` additionally applies 5 mm first-frame object grid sampling and,
for Demo v4 shape-prior outputs, includes `surface_points` and
`interior_points`.

Demo v4 FuturePhysTwin chunk roots add a top-level `READY` marker. Directory
watchers and batch consumers must ignore any discovered case directory until
that marker exists; producer-side temporary materialization lives under
`<base>/.publishing/` and is not a consumable case root.

Demo v4 online-primary aggregate cases under `<base>/data/<case>/` use the same
received-frame numbering as aligned cases: per-frame files and time-axis rows
are indexed `0, 1, 2, ...` in the order Demo v4 publishes frames. These indices
do not refer to the fake-live source recording frame ids.

## Manifest

Strict products must record:

```json
{
  "compatibility_target": "PhysTwin",
  "tracking_product_backend": "phystwin-strict-tracking",
  "tracker_backend": "tapnextpp",
  "mask_backend": "edgetam",
  "depth_backend": "ir-ffs",
  "depth_source_internal": "ffs",
  "execution_mode": "workstation_strict"
}
```

For native-depth Demo 3.2 runs, the corresponding fields are:

```json
{
  "depth_backend": "native-realsense",
  "depth_source_internal": "realsense"
}
```
