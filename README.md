项目源于：https://cznull.github.io/vsbm
一个基于OpenGL的Web图形测试程序。
本项目是基于原项目使用Vulkan api和C++的重新实现。模板使用了https://github.com/naivisoftware/vulkansdldemo
同时附带一个控制台输出的fps检测和跑分功能，记录运行20秒的帧数并输出分数。
依赖：sdl2 vulkan-headers glm
Linux编译说明：
CMake .
make
运行
./vulkansdldemo

着色器编译说明：
已经编译好的vert.spv和frag.spv二进制文件直接和主程序放一起执行即可。
如需要自己编译，修改shader.vert和shader.frag并使用glslc编译
如
glslc vert.spv -o vert.spv

运行效果如下：
![20250403_00h31m20s_grim](https://github.com/user-attachments/assets/e4298385-21a6-428e-9286-dcaab88cdd1f)
