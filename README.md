<div align="center">

# Sunone Aimbot C++
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://github.com/SunOner/sunone_aimbot_cpp)
[![License MIT](https://badgen.net/github/license/SunOner/sunone_aimbot_cpp)](https://github.com/SunOner/sunone_aimbot_cpp/blob/main/LICENSE)
[![Github stars](https://img.shields.io/github/stars/SunOner/sunone_aimbot_cpp?color=ffb500)](https://github.com/SunOner/sunone_aimbot_cpp)
[![Discord server](https://badgen.net/discord/online-members/sunone)](https://discord.gg/sunone)
  <p>
    <a href="https://github.com/SunOner/sunone_aimbot_cpp/releases" target="_blank">
      <img width="75%" src="https://github.com/SunOner/sunone_aimbot/blob/main/media/one.gif"></a>
  </p>
</div>

- **This project is actively being developed thanks to the people who support on [Boosty](https://boosty.to/sunone) and [Patreon](https://www.patreon.com/sunone). By providing active support, you receive enhanced AI models.**

## How to Use

1. **Download the Latest Release**  
	Download the latest release from [here](https://github.com/SunOner/sunone_aimbot_cpp/releases).

2. **Download TensorRT**  
	Get TensorRT from [Yandex](https://disk.yandex.ru/d/S16C9oDSuF1_EQ) or [Developer Nvidia](https://developer.nvidia.com/tensorrt/download/10x).

3. **Unpack TensorRT and Aimbot**  
	Extract the contents of both TensorRT and the Aimbot.

4. **Copy DLL Files**  
	- Copy `TensorRT-10.6.0.26/lib/nvinfer_10.dll` to `sunone_aimbot_cpp/`.
	- Copy all files from `TensorRT-10.6.0.26/lib/` to `TensorRT-10.6.0.26/bin/`.

5. **Transfer the ONNX Model**  
	Copy `sunone_aimbot_cpp/models/sunxds_0_5_6.onnx` to `TensorRT-10.6.0.26/bin/`.

6. **Generate Engine model File**  
	Open Command Prompt in `TensorRT-10.6.0.26/bin/` and execute:
	```bash
	trtexec.exe --onnx=sunxds_0.5.6.onnx --saveEngine=sunxds_0.5.6.engine --fp16
	```
	Or export it .onnx model via overlay (AI).
	
7. **Finalize Setup**  
	After the export (~1-5 minutes), copy `TensorRT-10.6.0.26/bin/sunxds_0.5.6.engine` to `sunone_aimbot_cpp/models/`.

8. **Run the Application**  
	Execute `ai.exe`.

> **⚠️ WARNING:** TensorRT version 10 does not support the Pascal architecture (10 series graphics cards). Use only with GPUs of at least the 20 series.

> **ℹ️ NOTE:** This guide is intended for advanced users. If you encounter errors while building the modules, please report them on the [Discord server](https://discord.gg/sunone).

## 📺 Installation Video

[![Watch the Installation Tutorial](https://img.youtube.com/vi/EyPtfXLhiuo/0.jpg)](https://www.youtube.com/watch?v=EyPtfXLhiuo)

Click the image above to watch the installation tutorial video.

## 🛠 Build the Project from Source

1. **Install Visual Studio 2019 Community**  
   Download and install from the [official website](https://visualstudio.microsoft.com/vs/community/).

2. **Install Windows SDK**  
   Ensure you have Windows SDK version **10.0.26100.0** installed.

3. **Install CUDA and cuDNN**  
	- **CUDA 12.4**
		Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
	- **cuDNN 9.1**
		Available on the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive) website.

4. **Set Up Project Structure**  
	Create a folder named `modules` in the directory `sunone_aimbot_cpp/sunone_aimbot_cpp/modules`.
	
5. **Build OpenCV with CUDA support**
	- Download and install [cmake](https://cmake.org/).
	- Download [opencv](https://github.com/opencv/opencv).
	- Download [opencv contrib](https://github.com/opencv/opencv_contrib/tags).
	- Create new dirs, `sunone_aimbot_cpp/modules/opencv/` and `sunone_aimbot_cpp/modules/opencv/build'
	- Unpack opencv-4.10.0 to `sunone_aimbot_cpp/modules/opencv/opencv-4.10.0` and opencv_contrib-4.10.0 to `sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0`.
	- Unpack cuDNN to `sunone_aimbot_cpp/modules/cudnn`.
	- Open cmake, configure where the opencv source code is located `sunone_aimbot_cpp/modules/opencv/opencv-4.10.0`.
	- Configure where the built code will be located `sunone_aimbot_cpp/modules/opencv/build'.
	- Hit configure.
	- Uncheck:
		- `WITH_NVCUVENC`
		- `WITH_NVCUVID`.
	- Check or configure:
		- `WITH_CUDA`
		- `WITH_CUBLAS`
		- `CUDA_FAST_MATH`
		- `WITH_CUDNN`
		- `CUDNN_LIBRARY` = `enter full path/sunone_aimbot_cpp/modules/cudnn/lib/x64/cudnn.lib`
		- `CUDNN_INCLUDE_DIR` = `enter full path/sunone_aimbot_cpp/modules/cudnn/include`
		- `CUDA_ARCH_BIN` = go to [cuda wiki](https://en.wikipedia.org/wiki/CUDA) and find your Nvidia GPU architecture. For me with my `rtx 3080ti` i write `8.6`
		- `OPENCV_DNN_CUDA`
		- `OPENCV_EXTRA_MODULES_PATH` = `enter full path/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules`
		- `BUILD_opencv_world`
	- Click configure and check if the flag has been reset from the `CUDA_FAST_MATH` option.
	- Hit Generate button to build C++ solution.
	- Close cmake and open `sunone_aimbot_cpp/modules/opencv/build/OpenCV.sln'.
	- Switch the build to x64 and release.
	- Open CMakeTargets folder in solution.
	- Hit right click on `ALL_BUILD` and click `build`. (Building a project can take up to two hours)
	- After building hit right click on `INSTALL` and click `build`.
	- Now check the folders with the built files.
		- `sunone_aimbot_cpp/modules/opencv/build/install/include/opencv2` - folders with .hpp and .h files.
		- `sunone_aimbot_cpp/modules/opencv/build/install/x64/vc16/bin` - .dll files.
		- `sunone_aimbot_cpp/modules/opencv/build/install/x64/vc16/lib` - .lib files.

6. **Download Required Libraries**  
   - [Boost](https://disk.yandex.ru/d/O8XkcKeQ3vNDFg)
   - TensorRT from [Yandex](https://disk.yandex.ru/d/S16C9oDSuF1_EQ) or [Developer Nvidia](https://developer.nvidia.com/tensorrt/download/10x)

7. **Extract Libraries**  
	Extract the downloaded libraries into the respective directories:
		- `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\boost_1_82_0`
		- `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\TensorRT-10.6.0.26`

8. **Compile Boost Libraries**  
	- Navigate to the Boost directory:
	```bash
	cd sunone_aimbot_cpp/sunone_aimbot_cpp/modules/boost_1_82_0
	```
	- Run the bootstrap script:
	```bash
	bootstrap.bat vc142
	```
	- After successful bootstrapping, build Boost:
	```bash
	b2.exe --build-type=complete link=static runtime-link=static threading=multi variant=release
	```

9. **Check the correct folders location**
	This is roughly what the project hierarchy should look like
	sunone_aimbot_cpp/
	├── .gitattributes
	├── .gitignore
	├── LICENSE
	├── README.md
	├── sunone_aimbot_cpp.sln
	├── include/
	├── models/
	├── modules/
	│   ├── boost_1_82_0/
	│   ├── imgui-1.91.2/
	│   ├── opencv/
	│   │   ├── build/
	│   │   ├── opencv_contrib-4.10.0/
	│   │   └── opencv-4.10.0/
	│   ├── stb/
	│   ├── TensorRT-10.6.0.26/
	│   └── tools/
	├── scr/
	├── screenshots/
	├── config.ini
	└── ghub_mouse.dll

10. **Configure Project Settings**  
	- Open the project in Visual Studio.
	- Ensure all library paths are correctly set in **Project Properties** under **Library Directories**.
	- Go to Nuget packages and install `Microsoft.Windows.CppWinRT`.
	
11. **Verify CUDA Integration**  
	- Right-click on the project in Visual Studio.
	- Navigate to **Build Dependencies** > **Build Customizations**.
	- Ensure that **CUDA 12.4** (.targets, .props) is included.
	
12. **Build the Project**  
	- Switch the build configuration to **Release**.
	- Build the project by selecting **Build** > **Build Solution**.

## 📋 Config docs
- The config documentation is available in a separate [repository](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md).

## 📚 References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/d1/dfb/intro.html)
- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
- [Boost](https://www.boost.org/)
- [ImGui](https://github.com/ocornut/imgui)
- [CppWinRT](https://github.com/microsoft/cppwinrt)
- [Python AI AIMBOT](https://github.com/SunOner/sunone_aimbot)

## 📄 Licenses

### Boost
- **License:** [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt)

### OpenCV
- **License:** [Apache License 2.0](https://opencv.org/license.html)

### ImGui
- **License:** [MIT License](https://github.com/ocornut/imgui/blob/master/LICENSE)