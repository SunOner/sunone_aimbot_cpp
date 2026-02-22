# Build From Source (Advanced Users)

If you want to compile the project yourself or modify code, follow these instructions.

## 1. Requirements

* **Visual Studio 2026 Community** ([Download](https://visualstudio.microsoft.com/vs/community/))
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

## 2. Choose Build Target in Visual Studio

* **DML (DirectML):**
  Select `Release | x64 | DML` (works on any modern GPU)
* **CUDA (TensorRT):**
  Select `Release | x64 | CUDA` (requires supported NVIDIA GPU, see above)

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
		(Choose "Visual Studio 18 2026", x64)

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

5. **Build in Visual Studio**

   * Open `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/OpenCV.sln`
     or click "Open Project" in CMake
   * Set build config: **x64 | Release**
   * Build `ALL_BUILD` target (can take up to 2 hours)
   * Then build `INSTALL` target

6. **Copy Resulting DLLs**

   * DLLs:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc17/bin/`
   * LIBs:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc17/lib/`
   * Includes:
     `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/include/opencv2`
   * Copy needed DLLs (`opencv_world4130.dll`, etc.) next to your project’s executable.

## 5. Notes on OpenCV for CUDA/DML

* **For CUDA build (TensorRT backend):**

  * You **must** build OpenCV with CUDA support (see the guide above).
  * Place all built DLLs (e.g., `opencv_world4130.dll`) next to your executable or in the `modules` folder.
* **For DML build (DirectML backend):**

  * You can use the official pre-built OpenCV DLLs if you **only** plan to use DirectML.
  * If you want to use both CUDA and DML modes in the same executable, you should always use your custom OpenCV build with CUDA enabled (it will work for both modes).
* **Note:**
  If you run the CUDA backend with non-CUDA OpenCV DLLs, the program will not work and may crash due to missing symbols.

## 6. Build and Run

1. Open the solution in Visual Studio 2026.
2. Choose your configuration (`Release | x64 | DML` or `Release | x64 | CUDA`).
3. Build the solution.
4. Run `ai.exe` from the output folder.

## 7. Exporting AI Models

* Convert PyTorch `.pt` models to ONNX:

  ```bash
  pip install ultralytics -U
  
  # TensorRT
  yolo export model=sunxds_0.5.6.pt format=onnx dynamic=true simplify=true
  
  # DML
  yolo export model=sunxds_0.5.6.pt format=onnx simplify=true
  ```
* To convert `.onnx` to `.engine` for TensorRT, use the overlay export tab (open overlay with HOME).
