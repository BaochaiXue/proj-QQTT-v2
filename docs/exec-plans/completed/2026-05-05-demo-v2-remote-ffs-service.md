# Demo 2 Remote FFS Service

## Goal

Add a remote FFS depth source for Demo 2 so the local RTX 5090 laptop keeps RealSense capture, HF EdgeTAM streaming, masked PCD, and UI, while a second machine runs FFS TensorRT depth as a long-lived service.

## Scope

- Add `services/ffs_remote/` protocol, client, and server files.
- Add `--depth-source ffs_remote` to `demo_v2/realtime_masked_edgetam_pcd.py`.
- Keep formal recording/alignment code unchanged.
- Document first validation commands and expected interpretation.

## Design

- Use ZeroMQ multipart REQ/REP for the first version.
- Request: JSON metadata + IR left bytes + IR right bytes.
- Response: JSON metadata + depth bytes.
- Client runs synchronously with `max_inflight=1`; if the remote reply times out, the current frame is skipped rather than mixing old depth with newer masks.
- Server loads the FFS TensorRT runner once, warms it up, aligns FFS depth to the provided color calibration, and returns the same `frame_id`.

## Validation

- Add unit coverage for protocol roundtrip packing, Demo 2 CLI validation, and remote client depth decoding with a fake socket.
- Run focused unit tests and `python scripts/harness/check_all.py`.
