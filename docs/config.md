# C++ Configuration Reference (`config.ini`)

This document describes `config.ini` for the C++ project in this repository.

- Source of truth in code: `sunone_aimbot_2/config/config.cpp`
- Config schema: `sunone_aimbot_2/config/config.h`

## 1. How Config Loading Works

1. On first run, if `config.ini` does not exist, the app creates it with defaults.
2. On startup, values are read from `config.ini`.
3. If a key is missing, a fallback value is used.
4. Some keys are clamped/validated during load (examples below).
5. `F4` reloads config at runtime (`button_reload_config` by default).

## 2. Value Formats

- `bool`: `true` or `false`
- `int`/`float`: regular numeric values
- `string`: raw text after `=`
- button lists: comma-separated key names, for example:
  - `button_targeting = RightMouseButton`
  - `button_exit = F2`
  - `screenshot_button = LeftAlt,F10`

Key names come from `sunone_aimbot_2/keyboard/keycodes.cpp` (examples: `LeftMouseButton`, `RightMouseButton`, `F1..F12`, `A..Z`, `Home`, `LeftAlt`, `RightControl`, `None`).

## 3. Quick Examples

### Example A: UDP receiver in LAN

```ini
capture_method = udp_capture
udp_ip = 0.0.0.0
udp_port = 1234
detection_resolution = 640
capture_fps = 60
backend = DML
```

### Example B: CUDA + TensorRT with GPU direct capture

```ini
capture_method = duplication_api
backend = TRT
capture_use_cuda = true
detection_resolution = 320
capture_fps = 120
circle_mask = false
depth_mask_enabled = false
```

### Example C: WinRT window capture by title

```ini
capture_method = winrt
capture_target = window
capture_window_title = Counter-Strike 2
capture_cursor = false
capture_borders = false
```

### Example D: Virtual camera capture

```ini
capture_method = virtual_camera
virtual_camera_name = None
virtual_camera_width = 1920
virtual_camera_heigth = 1080
```

### Example E: Lower latency and load for weak GPU

```ini
detection_resolution = 160
capture_fps = 60
confidence_threshold = 0.25
max_detections = 20
game_overlay_enabled = false
show_window = false
depth_inference_enabled = false
```

## 4. Full Key Reference

Defaults below are first-run defaults from `config.cpp`.

### 4.1 Capture

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `capture_method` | string | `duplication_api` | `duplication_api`, `winrt`, `virtual_camera`, `udp_capture` |
| `capture_target` | string | `monitor` | Used by WinRT: `monitor` or `window` |
| `capture_window_title` | string | empty | Used when `capture_method=winrt` and `capture_target=window` |
| `udp_ip` | string | `0.0.0.0` | For `udp_capture`; `0.0.0.0` accepts any sender |
| `udp_port` | int | `1234` | Clamped to `1..65535` |
| `detection_resolution` | int | `320` | Allowed: `160`, `320`, `640`; others become `320` |
| `capture_fps` | int | `60` | UI range `0..240`; `0` means uncapped capture limiter |
| `monitor_idx` | int | `0` | Monitor index for monitor-based capture |
| `circle_mask` | bool | `true` | Circular crop mask |
| `capture_borders` | bool | `true` | WinRT border option |
| `capture_cursor` | bool | `true` | WinRT cursor option |
| `virtual_camera_name` | string | `None` | `None` means auto-select |
| `virtual_camera_width` | int | `1920` | UI range `128..3840` |
| `virtual_camera_heigth` | int | `1080` | UI range `128..2160` |

### 4.2 Target

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `disable_headshot` | bool | `false` | If `true`, body targeting only |
| `body_y_offset` | float | `0.15` | Body target Y offset |
| `head_y_offset` | float | `0.05` | Head target Y offset |
| `auto_aim` | bool | `false` | Automatic target lock behavior |

### 4.3 Mouse (FOV, Speed, Prediction, Correction)

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `fovX` | int | `106` | UI range `10..120` |
| `fovY` | int | `74` | UI range `10..120` |
| `minSpeedMultiplier` | float | `0.1` | UI range `0.1..5.0` |
| `maxSpeedMultiplier` | float | `0.1` | UI range `0.1..5.0` |
| `predictionInterval` | float | `0.01` | UI range `0.00..0.5`; `0.00` disables prediction |
| `prediction_futurePositions` | int | `20` | UI range `1..40` |
| `draw_futurePositions` | bool | `true` | Draw predicted path |
| `snapRadius` | float | `1.5` | UI range `0.1..5.0` |
| `nearRadius` | float | `25.0` | UI range `1.0..40.0` |
| `speedCurveExponent` | float | `3.0` | UI range `0.1..10.0` |
| `snapBoostFactor` | float | `1.15` | UI range `0.01..4.0` |
| `easynorecoil` | bool | `false` | Recoil compensation master switch |
| `easynorecoilstrength` | float | `0.0` | UI range `0.1..500.0` when enabled |
| `input_method` | string | `WIN32` | `WIN32`, `GHUB`, `ARDUINO`, `KMBOX_NET`, `KMBOX_A`, `MAKCU` |

### 4.4 Wind Mouse

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `wind_mouse_enabled` | bool | `false` | Enable WindMouse behavior |
| `wind_G` | float | `18.0` | UI range `4.0..40.0` |
| `wind_W` | float | `15.0` | UI range `1.0..40.0` |
| `wind_M` | float | `10.0` | UI range `1.0..40.0` |
| `wind_D` | float | `8.0` | UI range `1.0..40.0` |

### 4.5 Arduino

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `arduino_baudrate` | int | `115200` | UI presets: `9600`, `19200`, `38400`, `57600`, `115200` |
| `arduino_port` | string | `COM0` | UI COM list `COM1..COM30` |
| `arduino_16_bit_mouse` | bool | `false` | Device-specific mode |
| `arduino_enable_keys` | bool | `false` | Device-specific mode |

### 4.6 KMBOX_NET

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `kmbox_net_ip` | string | `10.42.42.42` | Device IP |
| `kmbox_net_port` | string | `1984` | Device port as string |
| `kmbox_net_uuid` | string | `DEADC0DE` | Device UUID |

### 4.7 KMBOX_A

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `kmbox_a_pidvid` | string | empty | Format `PPPPVVVV` in one field |

### 4.8 MAKCU

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `makcu_baudrate` | int | `115200` | UI presets: `9600`, `19200`, `38400`, `57600`, `115200` |
| `makcu_port` | string | `COM0` | UI COM list `COM1..COM30` |

### 4.9 Mouse Shooting

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `auto_shoot` | bool | `false` | Auto fire logic |
| `bScope_multiplier` | float | `1.0` | UI range `0.5..2.0` |

### 4.10 AI

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `backend` | string | `TRT` on CUDA build, `DML` on DML build | CUDA build supports both `TRT` and `DML`; DML build uses `DML` |
| `dml_device_id` | int | `0` | DML adapter index |
| `ai_model` | string | `sunxds_0.5.6.engine` (CUDA) or `sunxds_0.5.6.onnx` (DML) | Model file name in `models` |
| `confidence_threshold` | float | `0.10` | UI range `0.01..1.00` |
| `nms_threshold` | float | `0.50` | UI range `0.00..1.00` |
| `max_detections` | int | `100` | UI range `1..100` |
| `export_enable_fp8` | bool | `false` | CUDA-only export option |
| `export_enable_fp16` | bool | `true` | CUDA-only export option |

### 4.11 CUDA (CUDA build only)

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `use_cuda_graph` | bool | `false` | TensorRT execution optimization |
| `use_pinned_memory` | bool | `false` | Host pinned memory mode |
| `gpuMemoryReserveMB` | int | `2048` | GPU reserve target |
| `enableGpuExclusiveMode` | bool | `true` | Exclusive behavior toggle |
| `capture_use_cuda` | bool | `true` | Direct GPU capture path for TRT + duplication API |

### 4.12 System

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `cpuCoreReserveCount` | int | `4` | Reserved CPU cores |
| `systemMemoryReserveMB` | int | `2048` | Reserved RAM amount |

### 4.13 Buttons

All button keys are comma-separated lists of key names.

| Key | Type | Default |
|---|---|---|
| `button_targeting` | list | `RightMouseButton` |
| `button_shoot` | list | `LeftMouseButton` |
| `button_zoom` | list | `RightMouseButton` |
| `button_exit` | list | `F2` |
| `button_pause` | list | `F3` |
| `button_reload_config` | list | `F4` |
| `button_open_overlay` | list | `Home` |
| `enable_arrows_settings` | bool | `false` |

### 4.14 Overlay

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `overlay_opacity` | int | `225` | UI range `220..255` |
| `overlay_ui_scale` | float | `1.0` | UI range `0.85..1.35` |
| `overlay_exclude_from_capture` | bool | `true` | Hide overlay from capture/recording |

### 4.15 Depth

Depth features require CUDA build.

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `depth_inference_enabled` | bool | `true` | Enable depth pipeline |
| `depth_model_path` | string | `depth_anything_v2.engine` | Depth model in `models/depth` |
| `depth_fps` | int | `100` | Clamped to `>= 0`; UI `0..120` |
| `depth_colormap` | int | `18` | Clamped to `0..21` |
| `depth_mask_enabled` | bool | `false` | Enable depth-based mask |
| `depth_mask_fps` | int | `5` | Clamped to `>= 0`; UI `1..30` |
| `depth_mask_near_percent` | int | `20` | Clamped to `1..100` |
| `depth_mask_alpha` | int | `90` | Clamped to `0..255` |
| `depth_mask_invert` | bool | `false` | Invert mask side |
| `depth_debug_overlay_enabled` | bool | `false` | Extra debug rendering |

### 4.16 Game Overlay

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `game_overlay_enabled` | bool | `false` | Master toggle |
| `game_overlay_max_fps` | int | `0` | UI `0..256`; `0` uncapped |
| `game_overlay_draw_boxes` | bool | `true` | Draw detection boxes |
| `game_overlay_draw_future` | bool | `true` | Draw future points |
| `game_overlay_draw_wind_tail` | bool | `true` | Draw WindMouse tail |
| `game_overlay_draw_frame` | bool | `true` | Draw capture frame |
| `game_overlay_show_target_correction` | bool | `true` | Draw correction debug |
| `game_overlay_box_a` | int | `255` | `0..255` |
| `game_overlay_box_r` | int | `0` | `0..255` |
| `game_overlay_box_g` | int | `255` | `0..255` |
| `game_overlay_box_b` | int | `0` | `0..255` |
| `game_overlay_frame_a` | int | `180` | `0..255` |
| `game_overlay_frame_r` | int | `255` | `0..255` |
| `game_overlay_frame_g` | int | `255` | `0..255` |
| `game_overlay_frame_b` | int | `255` | `0..255` |
| `game_overlay_box_thickness` | float | `2.0` | UI `0.5..10.0` |
| `game_overlay_frame_thickness` | float | `1.5` | UI `0.5..10.0` |
| `game_overlay_future_point_radius` | float | `5.0` | UI `1.0..20.0` |
| `game_overlay_future_alpha_falloff` | float | `1.0` | UI `0.1..5.0` |
| `game_overlay_icon_enabled` | bool | `false` | Enable icon overlay |
| `game_overlay_icon_path` | string | `icon.png` | Path to icon image |
| `game_overlay_icon_width` | int | `64` | UI `4..512` |
| `game_overlay_icon_height` | int | `64` | UI `4..512` |
| `game_overlay_icon_offset_x` | float | `0.0` | UI `-500..500` |
| `game_overlay_icon_offset_y` | float | `0.0` | UI `-500..500` |
| `game_overlay_icon_anchor` | string | `center` | `center`, `top`, `bottom`, `head` |
| `game_overlay_icon_class` | int | `-1` | `-1` for all classes |

### 4.17 Aim Simulation Overlay

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `aim_sim_enabled` | bool | `false` | Master toggle |
| `aim_sim_x` | int | `24` | UI `-3000..3000` |
| `aim_sim_y` | int | `24` | UI `-3000..3000` |
| `aim_sim_width` | int | `560` | Clamped `220..1920` (UI `220..1600`) |
| `aim_sim_height` | int | `360` | Clamped `180..1080` (UI `180..1000`) |
| `aim_sim_fps_min` | int | `90` | Clamped `15..360` |
| `aim_sim_fps_max` | int | `120` | Clamped `15..360` |
| `aim_sim_fps_jitter` | float | `0.15` | Clamped `0.0..0.8` |
| `aim_sim_capture_delay_ms` | float | `6.0` | Clamped `0.0..80.0` |
| `aim_sim_inference_delay_ms` | float | `12.0` | Clamped `0.0..120.0` |
| `aim_sim_use_live_inference` | bool | `true` | Use runtime inference delay |
| `aim_sim_input_delay_ms` | float | `2.0` | Clamped `0.0..60.0` |
| `aim_sim_extra_delay_ms` | float | `2.0` | Clamped `0.0..60.0` |
| `aim_sim_target_max_speed` | float | `560.0` | Clamped `20.0..2500.0` |
| `aim_sim_target_accel` | float | `1850.0` | Clamped `20.0..10000.0` |
| `aim_sim_target_stop_chance` | float | `0.25` | Clamped `0.0..0.95` |
| `aim_sim_show_observed` | bool | `true` | Show delayed target marker |
| `aim_sim_show_history` | bool | `true` | Show trajectory history |

### 4.18 Classes

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `class_player` | int | `0` | Must match your model class index |
| `class_head` | int | `1` | Must match your model class index |

### 4.19 Debug

| Key | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `show_window` | bool | `true` | Capture preview window in overlay |
| `show_fps` | bool | `false` | Legacy key; currently not used in runtime logic |
| `screenshot_button` | list | `None` | Screenshot hotkey list |
| `screenshot_delay` | int | `500` | Minimum interval (ms) between screenshots |
| `verbose` | bool | `false` | Verbose console logging |

### 4.20 Active Game Profile

| Key | Type | Default | Notes |
|---|---|---:|---|
| `active_game` | string | `UNIFIED` | Name of active profile from `[Games]` |

Profiles are stored in a `[Games]` section.

Format:

```ini
[Games]
UNIFIED = 1.0,0.022,0.022,false,0.0
MyGame = 1.2,0.022,0.022,true,103.0
```

Field order:

1. `sens`
2. `yaw`
3. `pitch` (optional; defaults to yaw if missing)
4. `fovScaled` (`true`/`false`, optional)
5. `baseFOV` (optional)

## 5. Special Notes

### 5.1 CUDA direct capture conditions

`capture_use_cuda` is effective only when all are true:

- CUDA build
- `backend = TRT`
- `capture_method = duplication_api`
- `circle_mask = false`
- depth mask is disabled

### 5.2 Runtime-only / auto-managed fields

- `fixed_input_size` exists in runtime state and is auto-detected by detector code.
- It is not a regular persisted `config.ini` key in current implementation.

### 5.3 Migration notes

If you update from older config versions:

- New/missing keys are auto-filled by fallback values.
- Invalid ranges are clamped (for keys with validation in `loadConfig`).

