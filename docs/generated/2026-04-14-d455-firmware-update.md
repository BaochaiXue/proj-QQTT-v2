# 2026-04-14 D455 Firmware Update

## Official version references

- RealSense D400 firmware releases page lists `Version-5.17.0.10` for D455: `https://dev.realsenseai.com/docs/firmware-releases-d400/`
- Latest official `librealsense` release assets include `RealSense.FW.Update_2.57.7.exe`: `https://github.com/realsenseai/librealsense/releases/tag/v2.57.7`

## Initial device state

Checked with `pyrealsense2`:

```text
239222303506  current=5.13.0.55  recommended=5.17.0.10
239222300433  current=5.17.0.10  recommended=5.17.0.10
239222300781  current=5.13.0.55  recommended=5.17.0.10
```

## Downloaded tools and firmware

```text
Tool:
C:\Users\zhang\AppData\Local\Temp\RealSense.FW.Update_2.57.7.exe

Firmware archive:
C:\Users\zhang\AppData\Local\Temp\d400_series_production_fw_5_17_0_10-1.zip

Firmware image:
C:\Users\zhang\AppData\Local\Temp\d400_series_production_fw_5_17_0_10\Signed_Image_UVC_5_17_0_10.bin
```

The firmware archive came from the RealSense EULA-gated download route:

```text
https://dev.realsenseai.com/download/41896/?tmstv=1776222621
```

## Commands run

Listed devices with the updater:

```powershell
& C:\Users\zhang\AppData\Local\Temp\RealSense.FW.Update_2.57.7.exe -l
```

Updated the two cameras that were below the recommended version:

```powershell
& C:\Users\zhang\AppData\Local\Temp\RealSense.FW.Update_2.57.7.exe `
  -s 239222303506 `
  -f C:\Users\zhang\AppData\Local\Temp\d400_series_production_fw_5_17_0_10\Signed_Image_UVC_5_17_0_10.bin

& C:\Users\zhang\AppData\Local\Temp\RealSense.FW.Update_2.57.7.exe `
  -s 239222300781 `
  -f C:\Users\zhang\AppData\Local\Temp\d400_series_production_fw_5_17_0_10\Signed_Image_UVC_5_17_0_10.bin
```

## Notable outcome during update

- `239222300781` completed cleanly and the updater reported success immediately.
- `239222303506` reported `Firmware update done` and then failed its reconnect check with `Camera not connected!`
- A fresh re-enumeration right after the update showed that `239222303506` had in fact come back on `5.17.0.10`

## Final device state

Verified with both `pyrealsense2` and `rs-fw-update -l`:

```text
239222303506  current=5.17.0.10  recommended=5.17.0.10
239222300433  current=5.17.0.10  recommended=5.17.0.10
239222300781  current=5.17.0.10  recommended=5.17.0.10
```

All three expected D455 cameras enumerated after the update.
