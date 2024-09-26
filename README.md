# sunone_aimbot_cpp

- This is a developer version, and it is currently under development!

## Current Dependencies
- Boost 1.82.0
- OpenCV
- TensorRT 10.3.0.26
- CUDA 12.4
- cuDNN 9.1

## Model Export Command
```bash
TensorRT-10.3.0.26/bin/trtexec.exe --onnx=sunxds_0.5.6.onnx --saveEngine=sunxds_0.5.6.engine --fp16
```
> **WARNING:** TensorRT version 10 does not support the Pascal architecture (10 series graphics card). Use only on video cards of at least 20 series.

> **NOTE:** This guide for building the application is intended for advanced users. If you encounter errors while building the modules, please report them on the [Discord server](https://discord.gg/sunone).

## Build a Project from Source
1. Download and install Visual Studio 2019 Community.
2. Install the Windows SDK (10.0.22000.194).
3. Install CUDA 12.4 and cuDNN 9.1.
4. Create a folder named `modules` in the directory `sunone_aimbot_cpp\sunone_aimbot_cpp\modules`.
5. Download the libraries:
	- [Boost](https://disk.yandex.ru/d/O8XkcKeQ3vNDFg)
	- [OpenCV](https://github.com/opencv/opencv/releases/tag/4.10.0) (Windows)
	- [TensorRT](https://disk.yandex.ru/d/2W-CgOvLQy7OTw).
	- [ImGui-1.91.2](https://github.com/ocornut/imgui/releases/tag/v1.91.2)
6. Extract the files into the directories:
	- `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\boost_1_82_0`
	- `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\opencv` (Rename `opencv-4.10.0` to `opencv`)
	- `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\TensorRT-10.3.0.26`
	- `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\imgui-1.91.2`
	- Copy all `.h` and `.cpp` files from `imgui-1.91.2` and paste to `sunone_aimbot_cpp\sunone_aimbot_cpp\`
7. Compile the Boost libraries:
	- Navigate to the folder `cd /sunone_aimbot_cpp/sunone_aimbot_cpp/modules/boost_1_82_0`
	- Run `bootstrap.bat vc142`
	- Once everything completes without errors, run `b2.exe --build-type=complete link=static runtime-link=static threading=multi variant=release`
8. Check the project for correct library imports in the project settings (Project->Properties).
9. Verify the presence of CUDA customization files (Right-click on the project -> Build Dependencies -> Build Customizations). CUDA 12.4 (.targets, .props) should be included in the project.
10. Switch the build to Release and build the project.

## References
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/d1/dfb/intro.html)
- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
- [Boost](https://www.boost.org/)
- [ImGui](https://github.com/ocornut/imgui)
- [Desktop Duplication API](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api)
- [Python AI AIMBOT](https://github.com/SunOner/sunone_aimbot)

## Licenses
### Boost
- License: [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt)

### OpenCV
- License: [Apache License 2.0](https://opencv.org/license.html)

### ImGui
- License: [MIT License](https://github.com/ocornut/imgui/blob/master/LICENSE)