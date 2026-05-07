# Demo 2 Remote FFS Network iperf Baseline

Date: 2026-05-07

## Scope

This report is for network throughput baseline between:

```text
client: WSL Ubuntu RTX 5090 Laptop
server: Native Ubuntu RTX 4090
```

The 4090 has no camera role. RealSense capture remains on WSL-5090.

## Target

```text
single camera realtime: 45 FPS
three camera realtime: 15 FPS per camera, aggregate 45 camera-FPS
```

## Attempted Check

WSL-5090 checked whether an iperf3 server was listening on the two candidate
4090 endpoints:

```bash
nc -vz 192.168.0.162 5201 || true
nc -vz 128.59.19.35 5201 || true
```

Result:

```text
2026-05-07 earlier:
  192.168.0.162:5201 refused
  128.59.19.35:5201 refused

2026-05-07 later:
  192.168.0.162:5201 succeeded
  128.59.19.35:5201 succeeded
```

## Decision

```text
iperf3 baseline: not run yet
reason: WSL-5090 does not currently have the iperf3 binary
```

Attempted WSL install:

```text
sudo apt-get install -y iperf3:
  blocked because sudo requires a password/TTY

conda install -n demo_2_max -c conda-forge iperf3:
  failed; package not available from current conda channels
```

The Ubuntu-4090 side appears reachable on port `5201`; install `iperf3` on
WSL-5090 or provide a sudo-capable shell before rerunning throughput tests.

Expected server command on Ubuntu-4090:

```bash
iperf3 -s -p 5201
```

Then run from WSL-5090:

```bash
iperf3 -c 192.168.0.162 -p 5201 -t 20
iperf3 -c 192.168.0.162 -p 5201 -t 20 -R
iperf3 -c 192.168.0.162 -p 5201 -t 20 -P 4

iperf3 -c 128.59.19.35 -p 5201 -t 20
iperf3 -c 128.59.19.35 -p 5201 -t 20 -R
iperf3 -c 128.59.19.35 -p 5201 -t 20 -P 4
```
