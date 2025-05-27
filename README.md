<div align="center">

# Sunone Aimbot C++

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://github.com/SunOner/sunone_aimbot_cpp)
[![License MIT](https://badgen.net/github/license/SunOner/sunone_aimbot_cpp)](https://github.com/SunOner/sunone_aimbot_cpp/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SunOner/sunone_aimbot_cpp?color=ffb500)](https://github.com/SunOner/sunone_aimbot_cpp)
[![Discord server](https://badgen.net/discord/online-members/sunone)](https://discord.gg/37WVp6sNEh)

  <p>
    <a href="https://github.com/SunOner/sunone_aimbot_cpp/releases" target="_blank">
      <img width="75%" src="https://github.com/SunOner/sunone_aimbot/blob/main/media/one.gif">
    </a>
  </p>
</div>

---

# ⚡️ Ready-to-Use Builds (Recommended)

**You do NOT need to compile anything if you just want to use the aimbot!**
Precompiled `.exe` builds are provided for both CUDA (NVIDIA only) and DirectML (all GPUs).

---

### 🟢 DirectML (DML) Build — Universal (All GPUs)

* **Works on:**

  * Any modern GPU (NVIDIA, AMD, Intel, including integrated graphics)
  * Windows 10/11 (x64)
  * No need for CUDA or special drivers!
* **Recommended for:**

  * GTX 10xx/9xx/7xx series (old NVIDIA)
  * Any AMD Radeon or Intel Iris/Xe GPU
  * Laptops and office PCs with integrated graphics
* **Download DML build:**
  [DirectML Release](https://disk.yandex.ru/d/9mf8VwfN0cK96w)

### 🟡 CUDA + TensorRT Build — High Performance (NVIDIA Only)

* **Works on:**

  * NVIDIA GPUs **GTX 1660, RTX 2000/3000/4000 or newer**
  * **Requires:** CUDA 12.8, TensorRT 10.8 (included in build)
  * Windows 10/11 (x64)
* **Not supported:** GTX 10xx/Pascal and older (TensorRT 10 limitation)
* **Includes both CUDA+TensorRT and DML support (switchable in settings)**
* **Download CUDA build:**
  [CUDA + TensorRT Release](https://disk.yandex.ru/d/VjyDyWLbv7AUHQ)

**Both versions are ready-to-use: just download, unpack, run `ai.exe` and follow instructions in the overlay.**

---

## 🚀 How to Run (For Precompiled Builds)

1. **Download and unpack your chosen version (see links above).**
2. For CUDA build, install [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) if not already installed.
3. For DML build, no extra software is needed.
4. **Run `ai.exe`.**
   On first launch, the model will be exported (may take up to 5 minutes).
5. Place your `.onnx` model in the `models` folder and select it in the overlay (HOME key).
6. All settings are available in the overlay.
   Use the HOME key to open/close overlay.

### 🎮 Controls

* **Right Mouse Button:** Aim at the detected target
* **F2:** Exit
* **F3:** Pause aiming
* **F4:** Reload config
* **Home:** Open/close overlay and settings

---

# 🛠️ Build From Source (Advanced Users)

If you want to compile the project yourself or modify code, follow these instructions.

## 1. Requirements

* **Visual Studio 2022 Community** ([Download](https://visualstudio.microsoft.com/vs/community/))
* **Windows 10 or 11 (x64)**
* **Windows SDK 10.0.26100.0** or newer
* **CMake** ([Download](https://cmake.org/))
* **OpenCV 4.10.0**
* **\[For CUDA version]**

  * [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)
  * [cuDNN 9.7.1](https://developer.nvidia.com/cudnn-downloads)
  * [TensorRT 10.8.0.43](https://developer.nvidia.com/tensorrt/download/10x)
* **\[For DML version]**

  * You can use [pre-built OpenCV DLLs](https://github.com/opencv/opencv/releases/tag/4.10.0) (just copy `opencv_world4100.dll` to your exe folder)
* Other dependencies:

  * [simpleIni](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
  * [serial](https://github.com/wjwwood/serial)
  * [GLFW](https://www.glfw.org/download.html)
  * [ImGui](https://github.com/ocornut/imgui)

---

## 2. Choose Build Target in Visual Studio

* **DML (DirectML):**
  Select `Release | x64 | DML` (works on any modern GPU)
* **CUDA (TensorRT):**
  Select `Release | x64 | CUDA` (requires supported NVIDIA GPU, see above)

---

## 3. Placement of Third-Party Modules and Libraries

Before building the project, **download and place all third-party dependencies** in the following directories inside your project structure:

**Required folders inside your repository:**

```
sunone_aimbot_cpp/
└── sunone_aimbot_cpp/
    └── modules/
```

**Place each dependency as follows:**

| Library   | Path                                                              |
| --------- | ----------------------------------------------------------------- |
| SimpleIni | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/SimpleIni.h`         |
| serial    | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/serial/`             |
| TensorRT  | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/TensorRT-10.8.0.43/` |
| GLFW      | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/glfw-3.4.bin.WIN64/` |
| OpenCV    | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/`             |
| cuDNN     | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/`              |

* **SimpleIni:**
  Download [`SimpleIni.h`](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
  Place in `modules/`.

* **serial:**
  Download the [`serial`](https://github.com/wjwwood/serial) library (whole folder).
  To build, open

  ```
  sunone_aimbot_cpp/sunone_aimbot_cpp/modules/serial/visual_studio/visual_studio.sln
  ```

  * Set **C/C++ > Code Generation > Runtime Library** to **Multi-threaded (/MT)**
  * Build in **Release x64**
  * Use the built DLL/LIB with your project.

* **TensorRT:**
  Download [TensorRT 10.8.0.43](https://developer.nvidia.com/tensorrt/download/10x)
  Place the folder as shown above.

* **GLFW:**
  Download [GLFW Windows binaries](https://www.glfw.org/download.html)
  Place the folder as shown above.

* **OpenCV:**
  Use your custom build or official DLLs (see CUDA/DML notes below).
  Place DLLs either next to your exe or in `modules/opencv/`.

* **cuDNN:**
  Place cuDNN files here (for CUDA build):
  `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/`

**Example structure after setup:**

```
sunone_aimbot_cpp/
└── sunone_aimbot_cpp/
    └── modules/
        ├── SimpleIni.h
        ├── serial/
        ├── TensorRT-10.8.0.43/
        ├── glfw-3.4.bin.WIN64/
        ├── opencv/
        └── cudnn/
```

---

## 4. How to Build OpenCV 4.10.0 with CUDA Support (For CUDA Version Only)

> This section is **only required** if you want to use the CUDA (TensorRT) version and need OpenCV with CUDA support.
> For DML build, skip this step — you can use the pre-built OpenCV DLL.

**Step-by-step instructions:**

1. **Download Sources**

   * [OpenCV 4.10.0](https://github.com/opencv/opencv/releases/tag/4.10.0)
   * [OpenCV Contrib 4.10.0](https://github.com/opencv/opencv_contrib/releases/tag/4.10.0)
   * [CMake](https://cmake.org/download/)
   * [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)
   * [cuDNN 9.7.1](https://developer.nvidia.com/cudnn-downloads)

2. **Prepare Directories**

   * Create:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/`
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build`
   * Extract `opencv-4.10.0` into
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv-4.10.0`
   * Extract `opencv_contrib-4.10.0` into
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0`
   * Extract cuDNN to
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn`

3. **Configure with CMake**

   * Open CMake GUI
   * Source code:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv-4.10.0`
   * Build directory:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build`
   * Click **Configure**
     (Choose "Visual Studio 17 2022", x64)

4. **Enable CUDA Options**

   * After first configure, set the following:

     * `WITH_CUDA` = ON
     * `WITH_CUBLAS` = ON
     * `ENABLE_FAST_MATH` = ON
     * `CUDA_FAST_MATH` = ON
     * `WITH_CUDNN` = ON
     * `CUDNN_LIBRARY` =
       `full_path_to/sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/lib/x64/cudnn.lib`
     * `CUDNN_INCLUDE_DIR` =
       `full_path_to/sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/include`
     * `CUDA_ARCH_BIN` =
       See [CUDA Wikipedia](https://en.wikipedia.org/wiki/CUDA) for your GPU.
       Example for RTX 3080-Ti: `8.6`
     * `OPENCV_DNN_CUDA` = ON
     * `OPENCV_EXTRA_MODULES_PATH` =
       `full_path_to/sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules`
     * `BUILD_opencv_world` = ON
   * Uncheck:

     * `WITH_NVCUVENC`
     * `WITH_NVCUVID`
   * Click **Configure** again
     (make sure nothing is reset)
   * Click **Generate**

5. **Build in Visual Studio**

   * Open `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/OpenCV.sln`
     or click "Open Project" in CMake
   * Set build config: **x64 | Release**
   * Build `ALL_BUILD` target (can take up to 2 hours)
   * Then build `INSTALL` target

6. **Copy Resulting DLLs**

   * DLLs:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc16/bin/`
   * LIBs:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc16/lib/`
   * Includes:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/include/opencv2`
   * Copy needed DLLs (`opencv_world4100.dll`, etc.) next to your project’s executable.

---

## 5. Notes on OpenCV for CUDA/DML

* **For CUDA build (TensorRT backend):**

  * You **must** build OpenCV with CUDA support (see the guide above).
  * Place all built DLLs (e.g., `opencv_world4100.dll`) next to your executable or in the `modules` folder.
* **For DML build (DirectML backend):**

  * You can use the official pre-built OpenCV DLLs if you **only** plan to use DirectML.
  * If you want to use both CUDA and DML modes in the same executable, you should always use your custom OpenCV build with CUDA enabled (it will work for both modes).
* **Note:**
  If you run the CUDA backend with non-CUDA OpenCV DLLs, the program will not work and may crash due to missing symbols.

---

## 6. Build and Run

1. Open the solution in Visual Studio 2022.
2. Choose your configuration (`Release | x64 | DML` or `Release | x64 | CUDA`).
3. Build the solution.
4. Run `ai.exe` from the output folder.

---

## 🔄 Exporting AI Models

* Convert PyTorch `.pt` models to ONNX:

  ```bash
  pip install ultralytics -U
  yolo export model=sunxds_0.5.6.pt format=onnx dynamic=true simplify=true
  ```
* To convert `.onnx` to `.engine` for TensorRT, use the overlay export tab (open overlay with HOME).

---

## 🗂️ Old Releases

* [Legacy and old versions](https://disk.yandex.ru/d/m0jbkiLEFvnZKg)

---

## 📋 Configuration

* See all configuration options and documentation here:
  [config\_cpp.md](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md)

---

## 📚 References & Useful Links

* [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
* [OpenCV Documentation](https://docs.opencv.org/4.x/d1/dfb/intro.html)
* [ImGui](https://github.com/ocornut/imgui)
* [CppWinRT](https://github.com/microsoft/cppwinrt)
* [GLFW](https://www.glfw.org/)
* [WindMouse](https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/)
* [KMBOX](https://www.kmbox.top/)
* [Python AI Version](https://github.com/SunOner/sunone_aimbot)

---

## 📄 Licenses

### OpenCV

* **License:** [Apache License 2.0](https://opencv.org/license.html)

### ImGui

* **License:** [MIT License](https://github.com/ocornut/imgui/blob/master/LICENSE)

---
## ❤️ Support the Project & Get Better AI Models

This project is actively developed thanks to the people who support it on [Boosty](https://boosty.to/sunone) and [Patreon](https://www.patreon.com/c/sunone).  
**By supporting the project, you get access to improved and better-trained AI models!**

---

**Need help or want to contribute? Join our [Discord server](https://discord.gg/37WVp6sNEh) or open an issue on GitHub!**