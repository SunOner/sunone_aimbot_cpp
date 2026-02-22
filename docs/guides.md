# C++ Guides and FAQ

## Installation (from source)

### Requirements

- Windows 10/11 x64
- Visual Studio 2026 (Desktop C++)
- Windows SDK 10.0.26100.0+
- CMake
- OpenCV 4.13.0
- Optional for CUDA build:
  - CUDA Toolkit 13.1
  - cuDNN 9.17.1
  - TensorRT 10.14.1.48

### Build configurations

- `Release | x64 | DML` for DirectML build
- `Release | x64 | CUDA` for TensorRT/CUDA build

### Run

1. Build solution.
2. Place required DLLs next to executable.
3. Run `ai.exe`.
4. Open overlay with `Home`.

## FAQ

### Why is `config.ini` recreated?

If file is missing, app generates default config automatically.

### How to reload config without restart?

Use `button_reload_config` (default `F4`).

### Which backend to use?

- Supported NVIDIA + CUDA build: start with `TRT`.
- Universal mode: `DML`.

### UDP capture does not receive frames

Check:

1. `capture_method = udp_capture`
2. receiver `udp_port` is open in firewall
3. sender streams MJPEG to receiver IP/port
4. set `udp_ip = 0.0.0.0` for diagnostics (accept any sender)

### UDP capture over LAN (FFmpeg)

Receiver PC setup:

1. Set capture config:

```ini
capture_method = udp_capture
udp_ip = 0.0.0.0
udp_port = 1234
```

2. Open UDP port in Windows Firewall (run PowerShell as Administrator):

```powershell
New-NetFirewallRule -DisplayName "capt UDP 1234" -Direction Inbound -Protocol UDP -LocalPort 1234 -Action Allow
```

3. Find receiver local IP (`IPv4 Address`) with:

```powershell
ipconfig
```

Sender PC (stream desktop to receiver over LAN):

```bash
ffmpeg -f gdigrab -framerate 60 -i desktop -vf scale=640:640 -vcodec mjpeg -q:v 5 -f mjpeg udp://RECEIVER_IP:1234
```

Lower bandwidth / weaker PC variant:

```bash
ffmpeg -f gdigrab -framerate 30 -i desktop -vf scale=320:320 -vcodec mjpeg -q:v 7 -f mjpeg udp://RECEIVER_IP:1234
```

Optional camera source (example for OBS Virtual Camera):

```bash
ffmpeg -f dshow -i video="OBS Virtual Camera" -vf scale=640:640 -vcodec mjpeg -q:v 5 -f mjpeg udp://RECEIVER_IP:1234
```

### Virtual camera does not open

- camera is busy in another app
- stored camera name is outdated
- set `virtual_camera_name = None` and refresh list

## Performance Troubleshooting

1. Start with lower load:
   - `detection_resolution = 320` (or `160`)
   - `capture_fps = 60`
   - `show_window = false`
   - `game_overlay_enabled = false`
2. For fastest TRT capture path:
   - `capture_method = duplication_api`
   - `backend = TRT`
   - `capture_use_cuda = true`
   - `circle_mask = false`
   - `depth_mask_enabled = false`
3. Disable extra cost while testing:
   - `depth_inference_enabled = false`
   - `verbose = false`
   - `screenshot_button = None`

## Detection Troubleshooting

1. Verify class mapping:
   - `class_player`
   - `class_head`
2. Tune thresholds:
   - `confidence_threshold`
   - `nms_threshold`
   - `max_detections`
3. Validate input frame quality:
   - increase `detection_resolution` if needed
   - disable `circle_mask` during diagnostics
   - disable `depth_mask_enabled` to exclude masking side effects

## G HUB Input Method

Set in config:

```ini
input_method = GHUB
```

Current UI check expects G HUB version `13.1.4` and default install path `C:\Program Files\LGHUB`.

## Model Export and Conversion

### Export ONNX from `.pt`

```bash
pip install ultralytics -U
yolo export model=sunxds_0.5.6.pt format=onnx dynamic=true simplify=true
```

### Convert ONNX -> TensorRT engine

Use overlay export tools in CUDA build.

### Depth models

Place depth models in `models/depth`, then use Depth section buttons:

- `Load depth model`
- `Export depth engine`

## Project Structure (brief)

Main folders under `sunone_aimbot_cpp/`:

- `capture/` - capture backends
- `config/` - config load/save/schema
- `detector/` - DML/TRT inference wrappers
- `mouse/` - input methods
- `overlay/` - UI, stats, preview, game overlay
- `depth/` and `tensorrt/` - CUDA-specific pipeline
- `keyboard/` - key mapping/listeners
