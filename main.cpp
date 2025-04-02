#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <set>
#include <glm/glm.hpp>
#include <assert.h>
//***新增头文件
#include <fstream>
#include <array>
#include <stdexcept>
#include <chrono>          // 用于 std::chrono
#include <glm/gtc/matrix_transform.hpp>  // 用于 glm::rotate, glm::lookAt, glm::perspective 等函数

// Global Settings
const char                      gAppName[] = "VulkanDemo";
const char                      gEngineName[] = "VulkanDemoEngine";
int                             gWindowWidth = 800;
int                             gWindowHeight = 600;
VkPresentModeKHR                gPresentationMode = VK_PRESENT_MODE_IMMEDIATE_KHR; // 改为立即模式，可以禁用垂直同步解锁fps上限
VkSurfaceTransformFlagBitsKHR   gTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
VkFormat                        gFormat = VK_FORMAT_B8G8R8A8_SRGB;
VkColorSpaceKHR                 gColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
VkImageUsageFlags               gImageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

// ***立方体顶点数据

// 渲染相关全局变量
struct UniformBufferObject {
    glm::vec2 resolution;
    float len;
    glm::vec3 origin;
    glm::mat4 view;  // 添加视图矩阵用于旋转
};

VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer uniformBuffer;
VkDeviceMemory uniformBufferMemory;
VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool;
std::vector<VkDescriptorSet> descriptorSets;
VkPipelineLayout pipelineLayout;
VkRenderPass renderPass;
VkPipeline graphicsPipeline;
std::vector<VkFramebuffer> swapChainFramebuffers;
VkCommandPool commandPool;
std::vector<VkCommandBuffer> commandBuffers;
VkSemaphore imageAvailableSemaphore;
VkSemaphore renderFinishedSemaphore;
VkFence inFlightFence;
std::vector<VkImageView> swapChainImageViews;  // 添加到全局变量中，与 swapChainFramebuffers 对应

const std::vector<float> quadVertices = {
    -1.0f, -1.0f, 0.0f, // 左下
     1.0f, -1.0f, 0.0f, // 右下
     1.0f,  1.0f, 0.0f, // 右上
    -1.0f, -1.0f, 0.0f, // 左下
     1.0f,  1.0f, 0.0f, // 右上
    -1.0f,  1.0f, 0.0f  // 左上
};
// ***

/**
 * This demo attempts to create a window and vulkan compatible surface using SDL
 * Verified and tested using multiple CPUs under windows.
 * Should work on every other SDL / Vulkan supported operating system (OSX, Linux, Android)
 * main() clearly outlines all the specific steps taken to create a vulkan instance,
 * select a device, create a vulkan compatible surface (opaque) associated with a window.
 */

//////////////////////////////////////////////////////////////////////////
// Global Settings
//////////////////////////////////////////////////////////////////////////

/**
 *  @return the set of layers to be initialized with Vulkan
 */
const std::set<std::string>& getRequestedLayerNames()
{
    static std::set<std::string> layers;
    if (layers.empty())
    {
        layers.emplace("VK_LAYER_NV_optimus");
        layers.emplace("VK_LAYER_KHRONOS_validation");
    }
    return layers;
}


/**
 * @return the set of required device extension names
 */
const std::set<std::string>& getRequestedDeviceExtensionNames()
{
    static std::set<std::string> layers;
    if (layers.empty())
    {
        layers.emplace(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    return layers;
}


/**
 * @return the set of required image usage scenarios
 * that need to be supported by the surface and swap chain
 */
const std::vector<VkImageUsageFlags> getRequestedImageUsages()
{
    static std::vector<VkImageUsageFlags> usages;
    if (usages.empty())
    {
        usages.emplace_back(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    }
    return usages;
}


//////////////////////////////////////////////////////////////////////////
// Utilities
//////////////////////////////////////////////////////////////////////////

/**
 * Clamps value between min and max
 */
template<typename T>
T clamp(T value, T min, T max)
{
    return glm::clamp<T>(value, min, max);
}

// ***添加工具
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void createVertexBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue) {
    VkDeviceSize bufferSize = sizeof(quadVertices[0]) * quadVertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, quadVertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

    copyBuffer(device, commandPool, graphicsQueue, stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t swapChainImageCount) {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

void createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    // 将阶段标志改为同时包含顶点和片段阶段
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}
void createGraphicsPipeline(VkDevice device, VkExtent2D swapChainExtent) {
    // 硬编码着色器代码（这里使用SPIR-V二进制格式，需预编译）
    std::vector<char> vertShaderCode = readFile("vert.spv");
    std::vector<char> fragShaderCode = readFile("frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(float) * 3; // 仅位置(3)
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.binding = 0;
    attributeDescription.location = 0;
    attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void createRenderPass(VkDevice device, VkFormat swapChainImageFormat) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}
void createFramebuffers(VkDevice device, VkExtent2D swapChainExtent, const std::vector<VkImage>& swapChainImages) {
    swapChainImageViews.resize(swapChainImages.size());
    swapChainFramebuffers.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapChainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = gFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &swapChainImageViews[i];  // 使用全局存储的视图
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}
void createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void createCommandBuffers(VkDevice device, uint32_t swapChainImageCount) {
    commandBuffers.resize(swapChainImageCount);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void createSyncObjects(VkDevice device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
        vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create synchronization objects!");
    }
}

void createDescriptorPool(VkDevice device, uint32_t swapChainImageCount) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(swapChainImageCount);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImageCount);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void createDescriptorSets(VkDevice device, uint32_t swapChainImageCount) {
    std::vector<VkDescriptorSetLayout> layouts(swapChainImageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImageCount);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(swapChainImageCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < swapChainImageCount; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }
}

void updateUniformBuffer(VkDevice device, VkExtent2D swapChainExtent, uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    ubo.resolution = glm::vec2(swapChainExtent.width, swapChainExtent.height);
    ubo.len = 3.6f;  // 控制相机距离

    float ang1 = 2.8f + time * 0.01f;  // 自动旋转，与 ontimer 中的 ang1+=0.01 一致
    float ang2 = 0.4f;
    glm::vec3 origin = glm::vec3(
        ubo.len * cos(ang1) * cos(ang2),
        ubo.len * sin(ang2),
        ubo.len * sin(ang1) * cos(ang2)
    );
    ubo.origin = origin;

    // 视图矩阵保持相机朝向原点
    ubo.view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    void* data;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBufferMemory);
}

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, VkExtent2D swapChainExtent) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent; // 使用传入的参数

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);

    vkCmdDraw(commandBuffer, 6, 1, 0, 0); // 暂时保持 4 个顶点，稍后检查

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void drawFrame(VkDevice device, VkQueue graphicsQueue, VkQueue presentQueue, VkSwapchainKHR swapChain, VkExtent2D swapChainExtent, uint32_t& currentFrame) {
    vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &inFlightFence);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    updateUniformBuffer(device, swapChainExtent, imageIndex);

    vkResetCommandBuffer(commandBuffers[imageIndex], 0);
    recordCommandBuffer(commandBuffers[imageIndex], imageIndex, swapChainExtent); // 传入 swapChainExtent

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(presentQueue, &presentInfo);
}
//////////////////////////////////////////////////////////////////////////
// Setup
//////////////////////////////////////////////////////////////////////////

/**
* Initializes SDL
* @return true if SDL was initialized successfully
*/
bool initSDL()
{
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) == 0)
        return true;
    std::cout << "Unable to initialize SDL\n";
    return false;
}


/**
 * Callback that receives a debug message from Vulkan
 */
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType,
    uint64_t obj,
    size_t location,
    int32_t code,
    const char* layerPrefix,
    const char* msg,
    void* userData)
{
    std::cout << "validation layer: " << layerPrefix << ": " << msg << std::endl;
    return VK_FALSE;
}


VkResult createDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback)
{
    auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}


/**
 *  Sets up the vulkan messaging callback specified above
 */
bool setupDebugCallback(VkInstance instance, VkDebugReportCallbackEXT& callback)
{
    VkDebugReportCallbackCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    createInfo.pfnCallback = debugCallback;

    if (createDebugReportCallbackEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS)
    {
        std::cout << "unable to create debug report callback extension\n";
        return false;
    }
    return true;
}


/**
 * Destroys the callback extension object
 */
void destroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
    if (func != nullptr)
    {
        func(instance, callback, pAllocator);
    }
}


bool getAvailableVulkanLayers(std::vector<std::string>& outLayers)
{
    // Figure out the amount of available layers
    // Layers are used for debugging / validation etc / profiling..
    unsigned int instance_layer_count = 0;
    VkResult res = vkEnumerateInstanceLayerProperties(&instance_layer_count, NULL);
    if (res != VK_SUCCESS)
    {
        std::cout << "unable to query vulkan instance layer property count\n";
        return false;
    }

    std::vector<VkLayerProperties> instance_layer_names(instance_layer_count);
    res = vkEnumerateInstanceLayerProperties(&instance_layer_count, instance_layer_names.data());
    if (res != VK_SUCCESS)
    {
        std::cout << "unable to retrieve vulkan instance layer names\n";
        return false;
    }

    // Display layer names and find the ones we specified above
    std::cout << "found " << instance_layer_count << " instance layers:\n";
    std::vector<const char*> valid_instance_layer_names;
    const std::set<std::string>& lookup_layers = getRequestedLayerNames();
    int count(0);
    outLayers.clear();
    for (const auto& name : instance_layer_names)
    {
        std::cout << count << ": " << name.layerName << ": " << name.description << "\n";
        auto it = lookup_layers.find(std::string(name.layerName));
        if (it != lookup_layers.end())
            outLayers.emplace_back(name.layerName);
        count++;
    }

    // Print the ones we're enabling
    std::cout << "\n";
    for (const auto& layer : outLayers)
        std::cout << "applying layer: " << layer.c_str() << "\n";
    return true;
}


bool getAvailableVulkanExtensions(SDL_Window* window, std::vector<std::string>& outExtensions)
{
    // Figure out the amount of extensions vulkan needs to interface with the os windowing system
    // This is necessary because vulkan is a platform agnostic API and needs to know how to interface with the windowing system
    unsigned int ext_count = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &ext_count, nullptr))
    {
        std::cout << "Unable to query the number of Vulkan instance extensions\n";
        return false;
    }

    // Use the amount of extensions queried before to retrieve the names of the extensions
    std::vector<const char*> ext_names(ext_count);
    if (!SDL_Vulkan_GetInstanceExtensions(window, &ext_count, ext_names.data()))
    {
        std::cout << "Unable to query the number of Vulkan instance extension names\n";
        return false;
    }

    // Display names
    std::cout << "found " << ext_count << " Vulkan instance extensions:\n";
    for (unsigned int i = 0; i < ext_count; i++)
    {
        std::cout << i << ": " << ext_names[i] << "\n";
        outExtensions.emplace_back(ext_names[i]);
    }

    // Add debug display extension, we need this to relay debug messages
    outExtensions.emplace_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    std::cout << "\n";
    return true;
}


/**
 * Creates a vulkan instance using all the available instance extensions and layers
 * @return if the instance was created successfully
 */
bool createVulkanInstance(const std::vector<std::string>& layerNames, const std::vector<std::string>& extensionNames, VkInstance& outInstance)
{
    // Copy layers
    std::vector<const char*> layer_names;
    for (const auto& layer : layerNames)
        layer_names.emplace_back(layer.c_str());

    // Copy extensions
    std::vector<const char*> ext_names;
    for (const auto& ext : extensionNames)
        ext_names.emplace_back(ext.c_str());

    // Get the suppoerted vulkan instance version
    unsigned int api_version;
    vkEnumerateInstanceVersion(&api_version);

    // initialize the VkApplicationInfo structure
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = NULL;
    app_info.pApplicationName = gAppName;
    app_info.applicationVersion = 1;
    app_info.pEngineName = gEngineName;
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

    // initialize the VkInstanceCreateInfo structure
    VkInstanceCreateInfo inst_info = {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = NULL;
    inst_info.flags = 0;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledExtensionCount = static_cast<uint32_t>(ext_names.size());
    inst_info.ppEnabledExtensionNames = ext_names.data();
    inst_info.enabledLayerCount = static_cast<uint32_t>(layer_names.size());
    inst_info.ppEnabledLayerNames = layer_names.data();

    // Create vulkan runtime instance
    std::cout << "initializing Vulkan instance\n\n";
    VkResult res = vkCreateInstance(&inst_info, NULL, &outInstance);
    switch (res)
    {
    case VK_SUCCESS:
        break;
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        std::cout << "unable to create vulkan instance, cannot find a compatible Vulkan ICD\n";
        return false;
    default:
        std::cout << "unable to create Vulkan instance: unknown error\n";
        return false;
    }
    return true;
}


/**
 * Allows the user to select a GPU (physical device)
 * @return if query, selection and assignment was successful
 * @param outDevice the selected physical device (gpu)
 * @param outQueueFamilyIndex queue command family that can handle graphics commands
 */
bool selectGPU(VkInstance instance, VkPhysicalDevice& outDevice, unsigned int& outQueueFamilyIndex)
{
    // Get number of available physical devices, needs to be at least 1
    unsigned int physical_device_count(0);
    vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
    if (physical_device_count == 0)
    {
        std::cout << "No physical devices found\n";
        return false;
    }

    // Now get the devices
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

    // Show device information
    std::cout << "found " << physical_device_count << " GPU(s):\n";
    int count(0);
    std::vector<VkPhysicalDeviceProperties> physical_device_properties(physical_devices.size());
    for (auto& physical_device : physical_devices)
    {
        vkGetPhysicalDeviceProperties(physical_device, &(physical_device_properties[count]));
        std::cout << count << ": " << physical_device_properties[count].deviceName << "\n";
        count++;
    }

    // Select one if more than 1 is available
    unsigned int selection_id = 0;
    // ***取消GPU设备选择（默认0号gpu运行）
    // if (physical_device_count > 1)
    // {
    //     while (true)
    //     {
    //         std::cout << "select device: ";
    //         std::cin  >> selection_id;
    //         if (selection_id >= physical_device_count || selection_id < 0)
    //         {
    //             std::cout << "invalid selection, expected a value between 0 and " << physical_device_count - 1 << "\n";
    //             continue;
    //         }
    //         break;
    //     }
    // }
    std::cout << "selected: " << physical_device_properties[selection_id].deviceName << "\n";
    VkPhysicalDevice selected_device = physical_devices[selection_id];

    // Find the number queues this device supports, we want to make sure that we have a queue that supports graphics commands
    unsigned int family_queue_count(0);
    vkGetPhysicalDeviceQueueFamilyProperties(selected_device, &family_queue_count, nullptr);
    if (family_queue_count == 0)
    {
        std::cout << "device has no family of queues associated with it\n";
        return false;
    }

    // Extract the properties of all the queue families
    std::vector<VkQueueFamilyProperties> queue_properties(family_queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(selected_device, &family_queue_count, queue_properties.data());

    // Make sure the family of commands contains an option to issue graphical commands.
    unsigned int queue_node_index = -1;
    for (unsigned int i = 0; i < family_queue_count; i++)
    {
        if (queue_properties[i].queueCount > 0 && queue_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            queue_node_index = i;
            break;
        }
    }

    if (queue_node_index < 0)
    {
        std::cout << "Unable to find a queue command family that accepts graphics commands\n";
        return false;
    }

    // Set the output variables
    outDevice = selected_device;
    outQueueFamilyIndex = queue_node_index;
    return true;
}


/**
 *  Creates a logical device
 */
bool createLogicalDevice(VkPhysicalDevice& physicalDevice,
    unsigned int queueFamilyIndex,
    const std::vector<std::string>& layerNames,
    VkDevice& outDevice)
{
    // Copy layer names
    std::vector<const char*> layer_names;
    for (const auto& layer : layerNames)
        layer_names.emplace_back(layer.c_str());


    // Get the number of available extensions for our graphics card
    uint32_t device_property_count(0);
    if (vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &device_property_count, NULL) != VK_SUCCESS)
    {
        std::cout << "Unable to acquire device extension property count\n";
        return false;
    }
    std::cout << "\nfound " << device_property_count << " device extensions\n";

    // Acquire their actual names
    std::vector<VkExtensionProperties> device_properties(device_property_count);
    if (vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &device_property_count, device_properties.data()) != VK_SUCCESS)
    {
        std::cout << "Unable to acquire device extension property names\n";
        return false;
    }

    // Match names against requested extension
    std::vector<const char*> device_property_names;
    const std::set<std::string>& required_extension_names = getRequestedDeviceExtensionNames();
    int count = 0;
    for (const auto& ext_property : device_properties)
    {
        std::cout << count << ": " << ext_property.extensionName << "\n";
        auto it = required_extension_names.find(std::string(ext_property.extensionName));
        if (it != required_extension_names.end())
        {
            device_property_names.emplace_back(ext_property.extensionName);
        }
        count++;
    }

    // Warn if not all required extensions were found
    if (required_extension_names.size() != device_property_names.size())
    {
        std::cout << "not all required device extensions are supported!\n";
        return false;
    }

    std::cout << "\n";
    for (const auto& name : device_property_names)
        std::cout << "applying device extension: " << name << "\n";

    // Create queue information structure used by device based on the previously fetched queue information from the physical device
    // We create one command processing queue for graphics
    VkDeviceQueueCreateInfo queue_create_info;
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queueFamilyIndex;
    queue_create_info.queueCount = 1;
    std::vector<float> queue_prio = { 1.0f };
    queue_create_info.pQueuePriorities = queue_prio.data();
    queue_create_info.pNext = NULL;
    queue_create_info.flags = 0;

    // Device creation information
    VkDeviceCreateInfo create_info;
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.ppEnabledLayerNames = layer_names.data();
    create_info.enabledLayerCount = static_cast<uint32_t>(layer_names.size());
    create_info.ppEnabledExtensionNames = device_property_names.data();
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_property_names.size());
    create_info.pNext = NULL;
    create_info.pEnabledFeatures = NULL;
    create_info.flags = 0;

    // Finally we're ready to create a new device
    VkResult res = vkCreateDevice(physicalDevice, &create_info, nullptr, &outDevice);
    if (res != VK_SUCCESS)
    {
        std::cout << "failed to create logical device!\n";
        return false;
    }
    return true;
}


/**
 *  Returns the vulkan device queue associtated with the previously created device
 */
void getDeviceQueue(VkDevice device, int familyQueueIndex, VkQueue& outGraphicsQueue)
{
    vkGetDeviceQueue(device, familyQueueIndex, 0, &outGraphicsQueue);
}


/**
 *  Creates the vulkan surface that is rendered to by the device using SDL
 */
bool createSurface(SDL_Window* window, VkInstance instance, VkPhysicalDevice gpu, uint32_t graphicsFamilyQueueIndex, VkSurfaceKHR& outSurface)
{
    if (!SDL_Vulkan_CreateSurface(window, instance, &outSurface))
    {
        std::cout << "Unable to create Vulkan compatible surface using SDL\n";
        return false;
    }

    // Make sure the surface is compatible with the queue family and gpu
    VkBool32 supported = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(gpu, graphicsFamilyQueueIndex, outSurface, &supported);
    if (!supported)
    {
        std::cout << "Surface is not supported by physical device!\n";
        return false;
    }

    return true;
}


/**
 * @return if the present modes could be queried and ioMode is set
 * @param outMode the mode that is requested, will contain FIFO when requested mode is not available
 */
bool getPresentationMode(VkSurfaceKHR surface, VkPhysicalDevice device, VkPresentModeKHR& ioMode)
{
    uint32_t mode_count(0);
    if(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, NULL) != VK_SUCCESS)
    {
        std::cout << "unable to query present mode count for physical device\n";
        return false;
    }

    std::vector<VkPresentModeKHR> available_modes(mode_count);
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, available_modes.data()) != VK_SUCCESS)
    {
        std::cout << "unable to query the various present modes for physical device\n";
        return false;
    }

    for (auto& mode : available_modes)
    {
        if (mode == ioMode)
            return true;
    }
    std::cout << "unable to obtain preferred display mode, fallback to FIFO\n";
    ioMode = VK_PRESENT_MODE_FIFO_KHR;
    return true;
}


/**
 * Obtain the surface properties that are required for the creation of the swap chain
 */
bool getSurfaceProperties(VkPhysicalDevice device, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR& capabilities)
{
    if(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities) != VK_SUCCESS)
    {
        std::cout << "unable to acquire surface capabilities\n";
        return false;
    }
    return true;
}


/**
 * Figure out the number of images that are used by the swapchain and
 * available to us in the application, based on the minimum amount of necessary images
 * provided by the capabilities struct.
 */
unsigned int getNumberOfSwapImages(const VkSurfaceCapabilitiesKHR& capabilities)
{
    unsigned int number = capabilities.minImageCount + 1;
    return number > capabilities.maxImageCount ? capabilities.minImageCount : number;
}


/**
 *  Returns the size of a swapchain image based on the current surface
 */
VkExtent2D getSwapImageSize(const VkSurfaceCapabilitiesKHR& capabilities)
{
    // Default size = window size
    VkExtent2D size = { (unsigned int)gWindowWidth, (unsigned int)gWindowHeight };

    // This happens when the window scales based on the size of an image
    if (capabilities.currentExtent.width == 0xFFFFFFF)
    {
        size.width  = glm::clamp<unsigned int>(size.width,  capabilities.minImageExtent.width,  capabilities.maxImageExtent.width);
        size.height = glm::clamp<unsigned int>(size.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    else
    {
        size = capabilities.currentExtent;
    }
    return size;
}


/**
 * Checks if the surface supports color and other required surface bits
 * If so constructs a ImageUsageFlags bitmask that is returned in outUsage
 * @return if the surface supports all the previously defined bits
 */
bool getImageUsage(const VkSurfaceCapabilitiesKHR& capabilities, VkImageUsageFlags& outUsage)
{
    const std::vector<VkImageUsageFlags>& desir_usages = getRequestedImageUsages();
    assert(desir_usages.size() > 0);

    // Needs to be always present
    outUsage = desir_usages[0];

    for (const auto& desired_usage : desir_usages)
    {
        VkImageUsageFlags image_usage = desired_usage & capabilities.supportedUsageFlags;
        if (image_usage != desired_usage)
        {
            std::cout << "unsupported image usage flag: " << desired_usage << "\n";
            return false;
        }

        // Add bit if found as supported color
        outUsage = (outUsage | desired_usage);
    }

    return true;
}


/**
 * @return transform based on global declared above, current transform if that transform isn't available
 */
VkSurfaceTransformFlagBitsKHR getTransform(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.supportedTransforms & gTransform)
        return gTransform;
    std::cout << "unsupported surface transform: " << gTransform;
    return capabilities.currentTransform;
}


/**
 * @return the most appropriate color space based on the globals provided above
 */
bool getFormat(VkPhysicalDevice device, VkSurfaceKHR surface, VkSurfaceFormatKHR& outFormat)
{
    unsigned int count(0);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr) != VK_SUCCESS)
    {
        std::cout << "unable to query number of supported surface formats";
        return false;
    }

    std::vector<VkSurfaceFormatKHR> found_formats(count);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, found_formats.data()) != VK_SUCCESS)
    {
        std::cout << "unable to query all supported surface formats\n";
        return false;
    }

    // This means there are no restrictions on the supported format.
    // Preference would work
    if (found_formats.size() == 1 && found_formats[0].format == VK_FORMAT_UNDEFINED)
    {
        outFormat.format = gFormat;
        outFormat.colorSpace = gColorSpace;
        return true;
    }

    // Otherwise check if both are supported
    for (const auto& found_format_outer : found_formats)
    {
        // Format found
        if (found_format_outer.format == gFormat)
        {
            outFormat.format = found_format_outer.format;
            for (const auto& found_format_inner : found_formats)
            {
                // Color space found
                if (found_format_inner.colorSpace == gColorSpace)
                {
                    outFormat.colorSpace = found_format_inner.colorSpace;
                    return true;
                }
            }

            // No matching color space, pick first one
            std::cout << "warning: no matching color space found, picking first available one\n!";
            outFormat.colorSpace = found_formats[0].colorSpace;
            return true;
        }
    }

    // No matching formats found
    std::cout << "warning: no matching color format found, picking first available one\n";
    outFormat = found_formats[0];
    return true;
}


/**
 * creates the swap chain using utility functions above to retrieve swap chain properties
 * Swap chain is associated with a single window (surface) and allows us to display images to screen
 */
bool createSwapChain(VkSurfaceKHR surface, VkPhysicalDevice physicalDevice, VkDevice device, VkSwapchainKHR& outSwapChain)
{
    // Get properties of surface, necessary for creation of swap-chain
    VkSurfaceCapabilitiesKHR surface_properties;
    if (!getSurfaceProperties(physicalDevice, surface, surface_properties))
        return false;

    // Get the image presentation mode (synced, immediate etc.)
    VkPresentModeKHR presentation_mode = gPresentationMode;
    if (!getPresentationMode(surface, physicalDevice, presentation_mode))
        return false;

    // Get other swap chain related features
    unsigned int swap_image_count = getNumberOfSwapImages(surface_properties);

    // Size of the images
    VkExtent2D swap_image_extent = getSwapImageSize(surface_properties);

    // Get image usage (color etc.)
    VkImageUsageFlags usage_flags;
    if (!getImageUsage(surface_properties, usage_flags))
        return false;

    // Get the transform, falls back on current transform when transform is not supported
    VkSurfaceTransformFlagBitsKHR transform = getTransform(surface_properties);

    // Get swapchain image format
    VkSurfaceFormatKHR image_format;
    if (!getFormat(physicalDevice, surface, image_format))
        return false;

    // Old swap chain
    VkSwapchainKHR old_swap_chain = outSwapChain;

    // Populate swapchain creation info
    VkSwapchainCreateInfoKHR swap_info;
    swap_info.pNext = nullptr;
    swap_info.flags = 0;
    swap_info.surface = surface;
    swap_info.minImageCount = swap_image_count;
    swap_info.imageFormat = image_format.format;
    swap_info.imageColorSpace = image_format.colorSpace;
    swap_info.imageExtent = swap_image_extent;
    swap_info.imageArrayLayers = 1;
    swap_info.imageUsage = usage_flags;
    swap_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap_info.queueFamilyIndexCount = 0;
    swap_info.pQueueFamilyIndices = nullptr;
    swap_info.preTransform = transform;
    swap_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap_info.presentMode = presentation_mode;
    swap_info.clipped = true;
    swap_info.oldSwapchain = NULL;
    swap_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;

    // Destroy old swap chain
    if (old_swap_chain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device, old_swap_chain, nullptr);
        old_swap_chain = VK_NULL_HANDLE;
    }

    // Create new one
    if (vkCreateSwapchainKHR(device, &swap_info, nullptr, &old_swap_chain) != VK_SUCCESS)
    {
        std::cout << "unable to create swap chain\n";
        return false;
    }

    // Store handle
    outSwapChain = old_swap_chain;
    return true;
}


/**
 *  Returns the handles of all the images in a swap chain, result is stored in outImageHandles
 */
bool getSwapChainImageHandles(VkDevice device, VkSwapchainKHR chain, std::vector<VkImage>& outImageHandles)
{
    unsigned int image_count(0);
    VkResult res = vkGetSwapchainImagesKHR(device, chain, &image_count, nullptr);
    if (res != VK_SUCCESS)
    {
        std::cout << "unable to get number of images in swap chain\n";
        return false;
    }

    outImageHandles.clear();
    outImageHandles.resize(image_count);
    if (vkGetSwapchainImagesKHR(device, chain, &image_count, outImageHandles.data()) != VK_SUCCESS)
    {
        std::cout << "unable to get image handles from swap chain\n";
        return false;
    }
    return true;
}


/**
 * Create a vulkan window
 */
SDL_Window* createWindow()
{
    return SDL_CreateWindow(gAppName, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, gWindowWidth, gWindowHeight, SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN);
}


/**
 *  Destroys the vulkan instance
 */
void quit(VkInstance instance, VkDevice device, VkDebugReportCallbackEXT callback, VkSwapchainKHR chain, VkSurfaceKHR presentation_surface)
{
    vkDestroySwapchainKHR(device, chain, nullptr);
    vkDestroyDevice(device, nullptr);
    destroyDebugReportCallbackEXT(instance, callback, nullptr);
    vkDestroySurfaceKHR(instance, presentation_surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    SDL_Quit();
}



// *** 修改后的main
int main(int argc, char *argv[]) {
    // Initialize SDL
    if (!initSDL())
        return -1;

    // Create vulkan compatible window
    SDL_Window* window {createWindow()};
    if (!window) {
        SDL_Quit();
        return -1;
    }

    // Get available vulkan extensions
    std::vector<std::string> found_extensions;
    if (!getAvailableVulkanExtensions(window, found_extensions))
        return -1;

    // Get available vulkan layers
    std::vector<std::string> found_layers;
    if (!getAvailableVulkanLayers(found_layers))
        return -1;

    if (found_layers.size() != getRequestedLayerNames().size())
        std::cout << "warning! not all requested layers could be found!\n";

    // Create Vulkan Instance
    VkInstance instance;
    if (!createVulkanInstance(found_layers, found_extensions, instance))
        return -1;

    // Vulkan messaging callback
    VkDebugReportCallbackEXT callback;
    setupDebugCallback(instance, callback);

    // Select GPU
    VkPhysicalDevice gpu;
    unsigned int graphics_queue_index(-1);
    if (!selectGPU(instance, gpu, graphics_queue_index))
        return -1;

    // Create a logical device
    VkDevice device;
    if (!createLogicalDevice(gpu, graphics_queue_index, found_layers, device))
        return -1;

    // Create the surface
    VkSurfaceKHR presentation_surface;
    if (!createSurface(window, instance, gpu, graphics_queue_index, presentation_surface))
        return -1;

    // Create swap chain
    VkSwapchainKHR swap_chain = VK_NULL_HANDLE;
    if (!createSwapChain(presentation_surface, gpu, device, swap_chain))
        return -1;

    // Get image handles from swap chain
    std::vector<VkImage> chain_images;
    if (!getSwapChainImageHandles(device, swap_chain, chain_images))
        return -1;

    // Fetch the queue
    VkQueue graphics_queue;
    getDeviceQueue(device, graphics_queue_index, graphics_queue);
    VkQueue present_queue = graphics_queue; // 假设图形队列和呈现队列相同

    // Create render pass
    createRenderPass(device, gFormat);

    // Create descriptor set layout
    createDescriptorSetLayout(device);

    // Create graphics pipeline
    VkSurfaceCapabilitiesKHR surfaceCaps;
    getSurfaceProperties(gpu, presentation_surface, surfaceCaps);
    VkExtent2D swapChainExtent = getSwapImageSize(surfaceCaps);
    createGraphicsPipeline(device, swapChainExtent);

    // Create framebuffers
    createFramebuffers(device, swapChainExtent, chain_images);

    // Create command pool
    createCommandPool(device, graphics_queue_index);

    // Create vertex and index buffers
    createVertexBuffer(gpu, device, commandPool, graphics_queue);

    // Create uniform buffers
    createUniformBuffers(gpu, device, chain_images.size());

    // Create descriptor pool and sets
    createDescriptorPool(device, chain_images.size());
    createDescriptorSets(device, chain_images.size());

    // Create command buffers
    createCommandBuffers(device, chain_images.size());

    // Create sync objects
    createSyncObjects(device);

    std::cout << "\nsuccessfully initialized vulkan and physical device (gpu).\n";
    std::cout << "successfully created a window and compatible surface\n";
    std::cout << "successfully created swapchain and rendering pipeline\n";
    std::cout << "ready to render!\n";

// Main render loop
    bool run = true;
    uint32_t currentFrame = 0;
    auto startTime = std::chrono::high_resolution_clock::now();  // 记录开始时间
    auto lastTime = std::chrono::high_resolution_clock::now();
    uint32_t frameCount = 0;  // 帧计数器
    uint32_t totalFrame = 0;  // 总帧数

    while (run) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                run = false;
            }
        }

        drawFrame(device, graphics_queue, present_queue, swap_chain, swapChainExtent, currentFrame);
        frameCount++;  // 每帧计数
        totalFrame++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        if (deltaTime >= 2.0f) {
            std::cout << "FPS: " << frameCount / deltaTime << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
        // 检查运行时间是否超过20秒
        float elapsedTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        if (elapsedTime >= 20.0f) {
            std::cout << "Score: " << totalFrame << std::endl;
            run = false;  // 退出循环
        }
    }

    // Wait for rendering to finish before cleanup
    vkDeviceWaitIdle(device);

    // Cleanup
    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    quit(instance, device, callback, swap_chain, presentation_surface);

    return 1;
}
