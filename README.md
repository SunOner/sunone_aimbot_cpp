<div align="center">

# Sunone Aimbot C++

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://github.com/SunOner/sunone_aimbot_cpp)
[![License MIT](https://badgen.net/github/license/SunOner/sunone_aimbot_cpp)](https://github.com/SunOner/sunone_aimbot_cpp/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SunOner/sunone_aimbot_cpp?color=ffb500)](https://github.com/SunOner/sunone_aimbot_cpp)
[![Discord server](https://badgen.net/discord/online-members/sunone)](https://discord.gg/sunone)

  <p>
    <a href="https://github.com/SunOner/sunone_aimbot_cpp/releases" target="_blank">
      <img width="75%" src="https://github.com/SunOner/sunone_aimbot/blob/main/media/one.gif">
    </a>
  </p>
</div>

- **This project is actively being developed thanks to the people who support on [Boosty](https://boosty.to/sunone) and [Patreon](https://www.patreon.com/sunone). By providing active support, you receive enhanced AI models.**

> **⚠️ WARNING:** TensorRT version 10 does not support the Pascal architecture (10 series graphics cards). Use only with GPUs of at least the 20 series.

## How to Use
1. **Download CUDA**
	- Download and install [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive).

2. **Download the Latest Release**  
	- Download the latest release from [here](https://disk.yandex.ru/d/Mpql8LBfjag2fA) (v2.4).

3. **Unpack Aimbot**  
   - Extract the contents of the Aimbot.

4. **First Launch and Model Export**  
	- Run `ai.exe` and wait until the standard `.onnx` model is exported, usually taking no more than two minutes.
	- To export another model, simply place it in `.onnx` format in the `models` folder. Then, in the AI tab, select this model, and it will be exported automatically.

5. **Settings**  
	- After successfully exporting the model, you can configure the program.
	- All settings are available in the overlay (default key is `Home`).
	- A list of settings can be found in the [config documentation](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md).

6. **Controls**
	- **Right Mouse Button:** Aim at the detected target.
	- **F2:** Exit the program.
	- **F3:** Activate pause for aiming.
	- **F4:** Reload config.
	- **Home:** Show overlay.

## 🛠 Build the Project from Source

> **ℹ️ NOTE:** This guide is intended for advanced users. If you encounter errors while building the modules, please report them on the [Discord server](https://discord.gg/sunone).

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

5. **Build OpenCV with CUDA Support**
	- Download and install [CMake](https://cmake.org/) and [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive).
	- Download [OpenCV](https://github.com/opencv/opencv).
	- Download [OpenCV Contrib](https://github.com/opencv/opencv_contrib/tags).
	- Create new directories: `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/` and `sunone_aimbot_cpp/modules/opencv/build`.
	- Extract `opencv-4.10.0` to `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv-4.10.0` and `opencv_contrib-4.10.0` to `sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0`.
	- Extract cuDNN to `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn`.
	- Open CMake and set the source code location to `sunone_aimbot_cpp/modules/opencv/opencv-4.10.0`.
	- Set the build directory to `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build`.
	- Click `Configure`.
	- (Some options will appear after the next configuration application. For example, to configure the CUDNN_LIBRARY paths, you first need to activate the WITH_CUDA option and click configure.)
	- Check or configure:
		- `WITH_CUDA`
		- `WITH_CUBLAS`
		- `ENABLE_FAST_MATH`
		- `CUDA_FAST_MATH`
		- `WITH_CUDNN`
		- `CUDNN_LIBRARY` = `<full path>sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/lib/x64/cudnn.lib`
		- `CUDNN_INCLUDE_DIR` = `<full path>sunone_aimbot_cpp/sunone_aimbot_cpp/modules/cudnn/include`
		- `CUDA_ARCH_BIN` = Visit the [CUDA Wiki](https://en.wikipedia.org/wiki/CUDA) to find your Nvidia GPU architecture. For example, for `RTX 3080ti`, enter `8.6`.
		- `OPENCV_DNN_CUDA`
		- `OPENCV_EXTRA_MODULES_PATH` = `<full path>sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules`
		- `BUILD_opencv_world`
		
		- Uncheck:
			- `WITH_NVCUVENC`
			- `WITH_NVCUVID`
   - Click `Configure` again and ensure that the flags for `CUDA_FAST_MATH` and `ENABLE_FAST_MATH` are not reset.
   - Click `Generate` to build the C++ solution.
   - Close CMake and open `sunone_aimbot_cpp/modules/opencv/build/OpenCV.sln`, or click `Open Project` in cmake.
   - Switch the build configuration to `x64` and `Release`.
   - Open the `CMakeTargets` folder in the solution.
   - Right-click on `ALL_BUILD` and select `Build`. (Building the project can take up to two hours.)
   - After building, right-click on `INSTALL` and select `Build`.
   - Verify the built files exist in the following folders:
     - `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/include/opencv2` - Contains `.hpp` and `.h` files.
     - `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc16/bin` - Contains `.dll` files.
     - `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install/x64/vc16/lib` - Contains `.lib` files.

6. **Download Required Libraries**  
	- [Boost](https://disk.yandex.ru/d/O8XkcKeQ3vNDFg)
	- TensorRT from [Yandex](https://disk.yandex.ru/d/S16C9oDSuF1_EQ) or [NVIDIA Developer](https://developer.nvidia.com/tensorrt/download/10x)

7. **Extract Libraries**  
	Extract the downloaded libraries into the respective directories:
	- `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/boost_1_82_0`
	- `sunone_aimbot_cpp/sunone_aimbot_cpp/modules/TensorRT-10.6.0.26`

8. **Compile Boost Libraries**
	- Navigate to the Boost directory:
		```bash
		cd sunone_aimbot_cpp/sunone_aimbot_cpp/modules/boost_1_82_0
		```
	- Run the bootstrap script (from PowerShell):
		```bash
		bootstrap.bat vc142
		```
	- After successful bootstrapping, build Boost:
		```bash
		b2.exe --build-type=complete link=static runtime-link=static threading=multi variant=release
		```

9. **Verify Folder Structure**  
   Ensure your project hierarchy looks like this:
   ```
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
   ```

10. **Configure Project Settings**
	- Open the project in Visual Studio.
	- Ensure all library paths are correctly set in **Project Properties** under **Library Directories**.
	- Go to NuGet packages and install `Microsoft.Windows.CppWinRT`.

11. **Verify CUDA Integration**
	- Right-click on the project in Visual Studio.
	- Navigate to **Build Dependencies** > **Build Customizations**.
	- Ensure that **CUDA 12.4** (.targets, .props) is included.

12. **Build the Project**
    - Switch the build configuration to **Release**.
    - Build the project by selecting **Build** > **Build Solution**.

## Old releases
- Stored [here](https://disk.yandex.ru/d/m0jbkiLEFvnZKg).
	
## 📋 Config Documentation
- The config documentation is available in a separate [repository](https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md).

## 📚 References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/d1/dfb/intro.html)
- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
- [Boost](https://www.boost.org/)
- [ImGui](https://github.com/ocornut/imgui)
- [CppWinRT](https://github.com/microsoft/cppwinrt)
- [Python AI AIMBOT](https://github.com/SunOner/sunone_aimbot)
- [Snowflake.cpp](https://github.com/BaconToaster/Snowflake.cpp)

## 📄 Licenses

### Boost
- **License:** [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt)

### OpenCV
- **License:** [Apache License 2.0](https://opencv.org/license.html)

### ImGui
- **License:** [MIT License](https://github.com/ocornut/imgui/blob/master/LICENSE)