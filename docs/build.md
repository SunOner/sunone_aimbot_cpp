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

  * Use prebuilt OpenCV package (include/lib/dll), no OpenCV build required.
  * One-time setup script:
    `powershell -ExecutionPolicy Bypass -File tools/setup_opencv_dml.ps1`
* Other dependencies:

  * [simpleIni](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
  * [serial](https://github.com/wjwwood/serial)
  * [GLFW](https://www.glfw.org/download.html)
  * [ImGui](https://github.com/ocornut/imgui)

## 2. Placement of Third-Party Modules and Libraries

Before building the project, **download and place all third-party dependencies** in the following directories inside your project structure:

**Required folders inside your repository:**

```
sunone_aimbot_2/
└── sunone_aimbot_2/
    └── modules/
```

**Place each dependency as follows:**

| Library   | Path                                                              |
| --------- | ----------------------------------------------------------------- |
| SimpleIni | `sunone_aimbot_2/sunone_aimbot_2/modules/SimpleIni.h`         |
| serial    | `sunone_aimbot_2/sunone_aimbot_2/modules/serial/`             |
| TensorRT  | `sunone_aimbot_2/sunone_aimbot_2/modules/TensorRT-10.14.1.48/` |
| GLFW      | `sunone_aimbot_2/sunone_aimbot_2/modules/glfw-3.4.bin.WIN64/` |
| OpenCV    | `sunone_aimbot_2/sunone_aimbot_2/modules/opencv/`             |

* **SimpleIni:**
  Download [`SimpleIni.h`](https://github.com/brofield/simpleini/blob/master/SimpleIni.h)
  Place in `modules/`.

* **serial:**
  Download the [`serial`](https://github.com/wjwwood/serial) library (whole folder).
  For CMake build, this library is compiled from sources automatically (no separate `serial.sln` build step).

* **TensorRT:**
  Download [TensorRT-10.14.1.48](https://developer.nvidia.com/tensorrt/download/10x)
  Place the folder as shown above.

* **GLFW:**
  Download [GLFW Windows binaries](https://www.glfw.org/download.html)
  Place the folder as shown above.

* **OpenCV:**
  For DML, run `tools/setup_opencv_dml.ps1` (downloads prebuilt OpenCV package).
  For CUDA, use your custom OpenCV build with CUDA support.

**Example structure after setup:**

```
sunone_aimbot_2/
└── sunone_aimbot_2/
	└── modules/
		├── SimpleIni.h
        ├── serial/
        ├── TensorRT-10.14.1.48/
        ├── glfw-3.4.bin.WIN64/
        └── opencv/
```

## 3. How to Build OpenCV 4.13.0 with CUDA Support (For CUDA Version Only)

> This section is **only required** if you want to use the CUDA (TensorRT) version and need OpenCV with CUDA support.
> For DML build, skip this step — you can use the pre-built OpenCV DLL.

**Fast path (recommended):**

Use helper script from repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_opencv_cuda.ps1 -AutoDetectCudaArch
```

The script requires CMake generator `Visual Studio 18 2026`.

Useful options:

* Set architecture explicitly:

  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/build_opencv_cuda.ps1 -CudaArchBin 8.6
  ```

* Build OpenCV for popular consumer NVIDIA architectures (GTX 16 / RTX 20 / 30 / 40 / 50):

  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/build_opencv_cuda.ps1 -CudaArchBin all
  ```

* Reuse already downloaded sources:

  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/build_opencv_cuda.ps1 -SkipDownload
  ```

* Configure only (without build):

  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/build_opencv_cuda.ps1 -ConfigureOnly
  ```

**Manual path:**

1. **Download Sources**

	* [OpenCV 4.13.0](https://github.com/opencv/opencv/releases/tag/4.13.0)
	* [OpenCV Contrib 4.13.0](https://github.com/opencv/opencv_contrib/releases/tag/4.13.0)
	* [CMake](https://cmake.org/download/)
	* [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-13-1-0-download-archive)
	* [cuDNN 9.17.1](https://developer.nvidia.com/cudnn-downloads)

2. **Prepare Directories**

	* Create:
		`sunone_aimbot_2/sunone_aimbot_2/modules/opencv/`
		`sunone_aimbot_2/sunone_aimbot_2/modules/opencv/build`
	* Extract `opencv-4.13.0` into
		`sunone_aimbot_2/sunone_aimbot_2/modules/opencv/opencv-4.13.0`
	* Extract `opencv_contrib-4.13.0` into
		`sunone_aimbot_2/sunone_aimbot_2/modules/opencv/opencv_contrib-4.13.0`
	* install cuDNN
		Default install path `C:/Program Files/NVIDIA/CUDNN/v9.17`

3. **Configure with CMake**

	* Open CMake GUI
	* Source code:
		`sunone_aimbot_2/sunone_aimbot_2/modules/opencv/opencv-4.13.0`
	* Build directory:
		`sunone_aimbot_2/sunone_aimbot_2/modules/opencv/build`
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
			`.../sunone_aimbot_2/sunone_aimbot_2/modules/cudnn/lib/x64/cudnn.lib`
		* `CUDNN_INCLUDE_DIR` =
			`.../sunone_aimbot_2/sunone_aimbot_2/modules/cudnn/include`
		* `CUDA_ARCH_BIN` =
			See [CUDA Wikipedia](https://en.wikipedia.org/wiki/CUDA) for your GPU.
			Example for RTX 3080-Ti: `8.6`
		* `OPENCV_DNN_CUDA` = ON
		* `OPENCV_EXTRA_MODULES_PATH` =
			`.../sunone_aimbot_2/sunone_aimbot_2/modules/opencv/opencv_contrib-4.13.0/modules`
		* `BUILD_opencv_world` = ON
	* Uncheck:

		* `WITH_NVCUVENC`
		* `WITH_NVCUVID`
	 
	* Click **Configure** again
		(make sure nothing is reset)
	* Click **Generate**

5. **Build in Visual Studio**

   * Open `sunone_aimbot_2/sunone_aimbot_2/modules/opencv/build/OpenCV.sln`
     or click "Open Project" in CMake
   * Set build config: **x64 | Release**
   * Build `ALL_BUILD` target (can take up to 2 hours)
   * Then build `INSTALL` target

6. **Copy Resulting DLLs**

   * DLLs:
     `sunone_aimbot_2/sunone_aimbot_2/modules/opencv/build/install/x64/vc*/bin/`
   * LIBs:
     `sunone_aimbot_2/sunone_aimbot_2/modules/opencv/build/install/x64/vc*/lib/`
   * Includes:
     `sunone_aimbot_2/sunone_aimbot_2/modules/opencv/build/install/include/opencv2`
   * Copy needed DLLs (`opencv_world4130.dll`, etc.) next to your project’s executable.

## 4. Notes on OpenCV for CUDA/DML

* **For CUDA build (TensorRT backend):**

  * You **must** build OpenCV with CUDA support (see the guide above).
  * Place all built DLLs (e.g., `opencv_world4130.dll`) next to your executable or in the `modules` folder.
* **For DML build (DirectML backend):**

  * Use prebuilt OpenCV package (include/lib/dll) if you **only** plan to use DirectML.
  * OpenCV runtime for DML must not import CUDA (`cudnn/cublas/npp`) or GStreamer (`gst*`) DLLs.
  * If you want to use both CUDA and DML modes in the same executable, you should always use your custom OpenCV build with CUDA enabled (it will work for both modes).
* **Note:**
  If you run the CUDA backend with non-CUDA OpenCV DLLs, the program will not work and may crash due to missing symbols.

## 5. Configure and Build Project

After sections 2-4 are complete, configure one backend with CMake.

Use separate build directories for each backend:

* **DML (DirectML):**

  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/setup_opencv_dml.ps1
  Remove-Item build/dml -Recurse -Force -ErrorAction SilentlyContinue
  cmake -S . -B build/dml -G "Visual Studio 18 2026" -A x64 -DAIMBOT_USE_CUDA=OFF
  cmake --build build/dml --config Release
  ```

  For DML build, OpenCV must be built **without CUDA** and **without GStreamer**.
  The setup script downloads prebuilt OpenCV and places it in
  `sunone_aimbot_2/modules/opencv/prebuilt/opencv/build` (auto-detected by CMake).

* **CUDA (TensorRT):**

  ```powershell
  cmake -S . -B build/cuda -G "Visual Studio 18 2026" -A x64 `
    -DAIMBOT_USE_CUDA=ON `
    -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" `
    -DCUDA_NVCC_FLAGS="--allow-unsupported-compiler"
  cmake --build build/cuda --config Release
  ```

Only `Visual Studio 18 2026` is supported for this project.

If your dependencies are stored in non-default paths, pass CMake cache variables, for example:

```powershell
cmake -S . -B build/dml -G "Visual Studio 18 2026" -A x64 `
  -DAIMBOT_OPENCV_INCLUDE_DIR="C:/opencv/include" `
  -DAIMBOT_OPENCV_LIBRARY="C:/opencv/lib/opencv_world4130.lib" `
  -DAIMBOT_ONNXRUNTIME_DIR="C:/packages/Microsoft.ML.OnnxRuntime.DirectML.1.22.0" `
  -DAIMBOT_CPPWINRT_INCLUDE_DIR="C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/cppwinrt"
```

You can open the generated solution from the build folder (`build/dml` or `build/cuda`) if you prefer building from Visual Studio UI.

Run `ai.exe` from `<build-dir>/Release/`.

## 6. Exporting AI Models

* Convert PyTorch `.pt` models to ONNX:

  ```bash
  pip install ultralytics -U
  
  # TensorRT
  yolo export model=sunxds_0.8.0.pt format=onnx dynamic=true simplify=true
  
  # DML
  yolo export model=sunxds_0.8.0.pt format=onnx simplify=true
  ```
* To convert `.onnx` to `.engine` for TensorRT, use the overlay export tab (open overlay with HOME).
