# Demo 2 Remote FFS Network iperf Baseline

Date: 2026-05-07

## Scope

This report is for network throughput baseline between:

```text
client: WSL Ubuntu RTX 5090 Laptop
server: Native Ubuntu RTX 4090
```

The 4090 has no camera role. RealSense capture remains on WSL-5090.

## Attempted Check

WSL-5090 checked whether an iperf3 server was listening on the two candidate
4090 endpoints:

```bash
nc -vz 192.168.0.162 5201 || true
nc -vz 128.59.19.35 5201 || true
```

Result:

```text
192.168.0.162:5201 refused
128.59.19.35:5201 refused
```

## Decision

```text
iperf3 baseline: not run
reason: no iperf3 server is currently listening on the Ubuntu-4090 side
```

Start this on Ubuntu-4090 before rerunning WSL throughput tests:

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

