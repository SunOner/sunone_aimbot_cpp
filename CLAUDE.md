# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ real-time computer vision system for ML/DL learning using video games. It combines YOLO-based object detection with multi-threaded processing pipelines. **Educational/experimental use only - not for multiplayer games.**

## Build Commands

### Visual Studio 2022 (Required)
```powershell
# Open solution
start sunone_aimbot_cpp.sln

# Build DML (DirectML - all GPUs)
msbuild sunone_aimbot_cpp.sln /p:Configuration=DML /p:Platform=x64

# Build CUDA (TensorRT - NVIDIA only)
msbuild sunone_aimbot_cpp.sln /p:Configuration=CUDA /p:Platform=x64
```

### Build Configurations
- **DML** (DirectML): Universal GPU support - works on any modern GPU
- **CUDA** (TensorRT): High-performance NVIDIA-only - requires CUDA 12.8, TensorRT 10.8

### Output
```
x64\DML\ai.exe    # DirectML build
x64\CUDA\ai.exe   # TensorRT build
```

## Architecture

### Multi-threaded Pipeline
```
captureThread → frameQueue → detectorThread → DetectionBuffer → mouseThread
                                                     ↓
                                              overlayThread (UI)
                                              gameOverlayThread (in-game visualization)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Main entry | `sunone_aimbot_cpp.cpp` | Thread orchestration, main loop |
| Screen capture | `capture/` | 3 methods: DXGI Duplication, WinRT, Virtual Camera |
| AI inference | `detector/` | Dual backend: TensorRT (trt_detector) and DirectML (dml_detector) |
| Post-processing | `detector/postProcess.cpp` | YOLO10/11 output parsing, NMS |
| Mouse control | `mouse/mouse.cpp` | Aiming calculations, prediction, wind mouse algorithm |
| Configuration | `config/config.cpp` | INI-based settings (config.ini) |
| Settings UI | `overlay/` | ImGui-based settings overlay |
| Game overlay | `overlay/Game_overlay.cpp` | DirectComposition in-game visualization |

### Detection Flow
```cpp
// Core detection structure (detector/detector.h)
struct Detection {
    cv::Rect box;        // Bounding box
    float confidence;    // Detection confidence [0-1]
    int classId;         // Target class
};
```

### Thread Synchronization
- `frameQueue`: `std::deque<cv::Mat>` - capture → detector
- `DetectionBuffer`: Thread-safe detection results (mutex + version counter)
- `std::condition_variable`: Inter-thread signaling
- `std::atomic<bool>`: Flags for aiming/shooting/zooming states

## Dependencies Structure

All third-party libraries go in `modules/`:
```
modules/
├── SimpleIni.h              # INI parsing
├── serial/                  # Arduino communication
├── TensorRT-10.8.0.43/      # CUDA build only
├── glfw-3.4.bin.WIN64/      # Window management
├── opencv/                  # Vision processing
├── cudnn/                   # CUDA build only
└── imgui/                   # UI framework (already included)
```

## Runtime Controls

- **Right Mouse Button**: Activate aiming
- **F2**: Exit
- **F3**: Pause detection
- **F4**: Reload config
- **HOME**: Toggle settings overlay

## Configuration

Main config file: `config.ini` (INI format)

Key sections: `[Capture]`, `[Detection]`, `[Mouse]`, `[Overlay]`, `[Game_overlay]`, `[Debug]`

## Model Export

Convert PyTorch to ONNX:
```bash
pip install ultralytics -U
yolo export model=model.pt format=onnx dynamic=true simplify=true
```

ONNX → TensorRT engine: Use the overlay export tab (HOME key)

## Code Patterns

### Dual Backend Pattern
Both `TrtDetector` and `DirectMLDetector` implement the same interface - code changes should maintain parity between backends.

### Input Method Abstraction
Multiple input drivers (Win32, GHub, Arduino, KMBOX) are hot-swappable via configuration.

### Capture Method Strategy
`IScreenCapture` interface with three implementations - new capture methods should implement this interface.
