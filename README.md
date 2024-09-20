# sunone_aimbot_cpp
 
- This is a developer version, and it is currently under development!
- Current dependencies:
	- boost_1_82_0
	- opencv
	- TensorRT-10.3.0.26
	- Cuda 12.4
	- Cudnn 9.1

- Model export command `TensorRT-10.3.0.26/bin/trtexec.exe --onnx=sunxds_0.5.6.onnx --saveEngine=sunxds_0.5.6.engine --fp16`
> [!NOTE]
> TensorRT version 10 does not support the Pascal architecture (10 series graphics card). Use only on video cards of at least 20 series.

- How to build a project from source code?
> [!WARNING]
> This guide for building the application is intended for advanced users. If you encounter errors while building the modules, please report them on the [Discord server](https://discord.gg/sunone).

- Download and install Visual Studio 2019 Community.
- Install the Windows SDK.
- Install Cuda 12.4 and CudaNN 9.1.
- Create a folder named `modules` in the directory `sunone_aimbot_cpp\sunone_aimbot_cpp\modules`.
- Download the libraries:
  - [Boost](https://disk.yandex.ru/d/O8XkcKeQ3vNDFg)
  - [Opencv](https://disk.yandex.ru/d/TV7XNTwQn3VXPQ)
  - [TensorRT](https://disk.yandex.ru/d/2W-CgOvLQy7OTw).
- Extract the files into the directories:
  - `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\boost_1_82_0`
  - `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\opencv` (Rename `opencv-4.10.0` to `opencv`)
  - `sunone_aimbot_cpp\sunone_aimbot_cpp\modules\TensorRT-10.3.0.26`
- Compile the Boost libraries:
  - Navigate to the folder `cd /sunone_aimbot_cpp/sunone_aimbot_cpp/modules/boost_1_82_0`
  - Run `bootstrap.bat vc142`
  - Once everything completes without errors, run `b2.exe link=static runtime-link=static threading=multi variant=release`
- Check the project for correct library imports in the project settings (Project->Properties).
- Verify the presence of Cuda customization files (Right-click on the project -> Build Dependencies -> Build Customizations). CUDA 12.4 (.targets, .props) should be included in the project.
- Switch the build to Release and build the project.

- Docs:
	- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
	- [OpenCV](https://docs.opencv.org/4.x/d1/dfb/intro.html)
	- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
	- [Boost](https://www.boost.org/)
	- [Desktop Duplication API](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api)
	- [Python AI AIMBOT](https://github.com/SunOner/sunone_aimbot)