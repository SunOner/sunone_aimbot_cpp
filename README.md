<div align="center">

# Sunone Aimbot C++

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://github.com/SunOner/sunone_aimbot_cpp)
[![License MIT](https://badgen.net/github/license/SunOner/sunone_aimbot_cpp)](https://github.com/SunOner/sunone_aimbot_cpp/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SunOner/sunone_aimbot_cpp?color=ffb500)](https://github.com/SunOner/sunone_aimbot_cpp)
[![CUDA 13.1](https://img.shields.io/badge/CUDA-13.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![Discord server](https://badgen.net/discord/online-members/37WVp6sNEh)](https://discord.gg/37WVp6sNEh)

  <p>
    <a href="https://github.com/SunOner/sunone_aimbot_cpp/releases" target="_blank">
      <img width="75%" src="https://github.com/SunOner/sunone_aimbot/blob/main/media/one.gif">
    </a>
  </p>
</div>

---

# Ready-to-Use Builds (Recommended)

**You do NOT need to compile anything if you just want to use the aimbot!**
Precompiled `.exe` builds are provided for both CUDA (NVIDIA only) and DirectML (all GPUs).

---

### DirectML (DML) Build — Universal (All GPUs)

* **Works on:**

  * Any modern GPU (NVIDIA, AMD, Intel, including integrated graphics)
  * Windows 10/11 (x64)
  * No need for CUDA or special drivers
* **Recommended for:**

  * GTX 10xx/9xx/7xx series (old NVIDIA)
  * Any AMD Radeon or Intel Iris/Xe GPU
  * Laptops and office PCs with integrated graphics
* **Download DML build:**
  [DirectML Release](https://disk.yandex.ru/d/jAVoMDfCuloNzw)

### CUDA + TensorRT Build — High Performance (NVIDIA Only)

* **Works on:**

  * NVIDIA GPUs **GTX 1660, RTX 2000/3000/4000/5000**
  * **Requires:** CUDA 13.1, TensorRT-10.14.1.48 (included in build)
  * Windows 10/11 (x64)
* **Not supported:** GTX 10xx/Pascal and older (TensorRT 10 limitation)
* **Includes both CUDA+TensorRT and DML support (switchable in settings)**
* **Download CUDA build:**
  [DML + CUDA + TensorRT Release](https://disk.yandex.ru/d/ltSkG0DadLBqcA)

**Both versions are ready-to-use: just download, unpack, run `ai.exe` and follow instructions in the overlay.**

---

## How to Run (For Precompiled Builds)

1. **Download and unpack your chosen version (see links above).**
2. For CUDA build, install [CUDA 13.1](https://developer.nvidia.com/cuda-13-1-0-download-archive) if not already installed.
3. For DML build, no extra software is needed.
4. **Run `ai.exe`.**
   On first launch, the model will be exported (may take up to 5 minutes).
5. Place your `.onnx` model in the `models` folder and select it in the overlay (HOME key).
6. All settings are available in the overlay.
   Use the HOME key to open/close overlay.

### Controls

* **Right Mouse Button:** Aim at the detected target
* **F2:** Exit
* **F3:** Pause aiming
* **F4:** Reload config
* **Home:** Open/close overlay and settings

---

# Build From Source (Advanced Users)

If you want to compile the project yourself or modify code, follow these instructions.

## 1. Requirements

* **Visual Studio 2022 Community** ([Download](https://visualstudio.microsoft.com/vs/community/))
* **Windows 10 or 11 (x64)**
* **Windows SDK 10.0.26100.0** or newer
* **CMake** ([Download](https://cmake.org/))
* **OpenCV 4.13.0**
* **\[For CUDA version]**

  * [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-13-1-0-download-archive)
  * [cuDNN 9.17.1](https://developer.nvidia.com/cudnn-downloads)
  * [TensorRT-10.14.1.48](https://developer.nvidia.com/tensorrt/download/10x)
* **\[For DML version]**

  * You can use [pre-built OpenCV DLLs](https://github.com/opencv/opencv/releases/tag/4.13.0) (just copy `opencv_world4130.dll` to your exe folder)
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
| TensorRT  | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/TensorRT-10.14.1.48/` |
| GLFW      | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/glfw-3.4.bin.WIN64/` |
| OpenCV    | `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/`             |

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
  Download [TensorRT-10.14.1.48](https://developer.nvidia.com/tensorrt/download/10x)
  Place the folder as shown above.

* **GLFW:**
  Download [GLFW Windows binaries](https://www.glfw.org/download.html)
  Place the folder as shown above.

* **OpenCV:**
  Use your custom build or official DLLs (see CUDA/DML notes below).
  Place DLLs either next to your exe or in `modules/opencv/`.

**Example structure after setup:**

```
sunone_aimbot_cpp/
└── sunone_aimbot_cpp/
	└── modules/
		├── SimpleIni.h
        ├── serial/
        ├── TensorRT-10.14.1.48/
        ├── glfw-3.4.bin.WIN64/
        └── opencv/
```

---

## 4. How to Build OpenCV 4.13.0 with CUDA Support (For CUDA Version Only)

> This section is **only required** if you want to use the CUDA (TensorRT) version and need OpenCV with CUDA support.
> For DML build, skip this step — you can use the pre-built OpenCV DLL.

**Step-by-step instructions:**

1. **Download Sources**

	* [OpenCV 4.13.0](https://github.com/opencv/opencv/releases/tag/4.13.0)
	* [OpenCV Contrib 4.13.0](https://github.com/opencv/opencv_contrib/releases/tag/4.13.0)
	* [CMake](https://cmake.org/download/)
	* [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-13-1-0-download-archive)
	* [cuDNN 9.17.1](https://developer.nvidia.com/cudnn-downloads)

2. **Prepare Directories**

	* Create:
		`sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/`
		`sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build`
	* Extract `opencv-4.13.0` into
		`sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv-4.13.0`
	* Extract `opencv_contrib-4.13.0` into
		`sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.13.0`
	* install cuDNN
		Default install path `C:/Program Files/NVIDIA/CUDNN/v9.17`

3. **Configure with CMake**

	* Open CMake GUI
	* Source code:
		`sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv-4.13.0`
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
			`.../sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/lib/x64/cudnn.lib`
		* `CUDNN_INCLUDE_DIR` =
			`.../sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/include`
		* `CUDA_ARCH_BIN` =
			See [CUDA Wikipedia](https://en.wikipedia.org/wiki/CUDA) for your GPU.
			Example for RTX 3080-Ti: `8.6`
		* `OPENCV_DNN_CUDA` = ON
		* `OPENCV_EXTRA_MODULES_PATH` =
			`.../sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.13.0/modules`
		* `BUILD_opencv_world` = ON
	* Uncheck:

		* `WITH_NVCUVENC`
		* `WITH_NVCUVID`
	 
	* Click **Configure** again
		(make sure nothing is reset)
	* Click **Generate**

5. **Enable GStreamer Options (Optional)**

	* If you want to use **GStreamer**, install both **Runtime** and **Development** packages from [here](https://gstreamer.freedesktop.org/download/#windows)

	* Choose:

		* **MSVC 64-bit (VS 2019, Release CRT)**
		* **1.26.10 runtime installer**
		* **1.26.10 development installer**

	* Then in **CMake GUI**:

		* Set:
		* `WITH_GSTREAMER` = **ON**

	* If `GSTREAMER_DIR` is NOT visible in CMake GUI

		* Sometimes OpenCV CMake does not show it automatically. In that case you must add it manually:

		* Click **Add Entry**
		
		* **Name:** `GSTREAMER_DIR`
		* **Type:** `PATH`
		* **Value:** `C:/Program Files/gstreamer/1.0/msvc_x86_64`

	* Then click **Configure** again (this usually auto-fills the rest).
	* If it still doesn’t, fill the variables manually like below.

	* Verify / Fill these GStreamer paths (import libs + includes)

	* Assuming:
		`GSTREAMER_DIR = C:/Program Files/gstreamer/1.0/msvc_x86_64`

	* Set:

		* `GSTREAMER_app_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstapp-1.0.lib`
		* `GSTREAMER_audio_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstaudio-1.0.lib`
		* `GSTREAMER_base_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstbase-1.0.lib`
		* `GSTREAMER_gstreamer_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstreamer-1.0.lib`
		* `GSTREAMER_pbutils_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstpbutils-1.0.lib`
		* `GSTREAMER_riff_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstriff-1.0.lib`
		* `GSTREAMER_video_LIBRARY` =
		`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gstvideo-1.0.lib`

	* And GLib deps:

		* `GSTREAMER_glib_LIBRARY` =
			`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/glib-2.0.lib`
		* `GSTREAMER_gobject_LIBRARY` =
			`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/gobject-2.0.lib`

	* Include dirs:

		* `GSTREAMER_gst_INCLUDE_DIR` =
			`C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0`
		* `GSTREAMER_glib_INCLUDE_DIR` =
			`C:/Program Files/gstreamer/1.0/msvc_x86_64/include/glib-2.0`
		* `GSTREAMER_glibconfig_INCLUDE_DIR` =
			`C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/glib-2.0/include`

	* After that:

		* Click **Configure** again (make sure nothing is reset)
		* Click **Generate**

---

### Runtime note (important)
Even if OpenCV links successfully, GStreamer must be available at runtime:
* Add to **PATH**:
  * `C:\Program Files\gstreamer\1.0\msvc_x86_64\bin`

6. **Build in Visual Studio**

   * Open `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/OpenCV.sln`
     or click "Open Project" in CMake
   * Set build config: **x64 | Release**
   * Build `ALL_BUILD` target (can take up to 2 hours)
   * Then build `INSTALL` target

7. **Copy Resulting DLLs**

   * DLLs:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc17/bin/`
   * LIBs:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc17/lib/`
   * Includes:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/include/opencv2`
   * Copy needed DLLs (`opencv_world4130.dll`, etc.) next to your project’s executable.

---

## 5. Notes on OpenCV for CUDA/DML

* **For CUDA build (TensorRT backend):**

  * You **must** build OpenCV with CUDA support (see the guide above).
  * Place all built DLLs (e.g., `opencv_world4130.dll`) next to your executable or in the `modules` folder.
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
  
  # TensorRT
  yolo export model=sunxds_0.5.6.pt format=onnx dynamic=true simplify=true
  
  # DML
  yolo export model=sunxds_0.5.6.pt format=onnx simplify=true
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
* [GStreamer](https://gstreamer.freedesktop.org/)
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