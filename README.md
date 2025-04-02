The project originates from: https://cznull.github.io/vsbm 

A Web graphics test program based on OpenGL. 

This project is a reimplementation of the original project using Vulkan API and C++. The template is based on https://github.com/naivisoftware/vulkansdldemo. 

At the same time, it is equipped with a console output FPS detection and benchmarking function, which records the frame rate for 20 seconds of operation and outputs the score. 

Dependencies: SDL2, Vulkan-Headers, glm 

Linux Compilation Instructions: 

CMake .


make


Run `./vulkansdldemo` 

Shader Compilation Instructions: 

The already compiled vert.spv and frag.spv binary files can be directly executed together with the main program. 

Compared with the original vsbm parameters, the polygon accuracy is weakened, but it can achieve a considerable frame rate on devices with lower performance, such as Android phones running through Termux. 

To adjust the computational cost, modify the constant at line 31 of shader.frag: float step = 0.01 * ubo.len; (the smaller the constant, the higher the precision). 

And recompile the shader files using glslc. 

Such as 

glslc shader.frag -o frag.spv


The running effect is as follows:

![20250403_00h31m20s_grim](https://github.com/user-attachments/assets/e4298385-21a6-428e-9286-dcaab88cdd1f)

中文说明：

项目源于：https://cznull.github.io/vsbm

一个基于OpenGL的Web图形测试程序。

本项目是基于原项目使用Vulkan api和C++的重新实现。模板使用了https://github.com/naivisoftware/vulkansdldemo

同时附带一个控制台输出的fps检测和跑分功能，记录运行20秒的帧数并输出分数。

依赖：sdl2 vulkan-headers glm

Linux编译说明：

CMake .

make

运行./vulkansdldemo

着色器编译说明：

已经编译好的vert.spv和frag.spv二进制文件直接和主程序放一起执行即可。

相较于原始的vsbm参数削弱了多边形精度，可以在性能较低的设备上（如安卓手机通过termux运行）获得可观的帧数。

如果要调节计算开销，修改shader.frag第31行float step = 0.01 * ubo.len;处的常量（越小精度越高）

并使用glslc重新编译着色器文件。

如

glslc shader.frag -o frag.spv

运行效果如下：


