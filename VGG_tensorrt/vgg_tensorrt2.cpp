#include<iostream>
#include<NvInfer.h>
#include<NvInferRuntimeCommon.h>
#include<vector>
#include<fstream>
#include<string>
#include"utils.h"
#include"logger.h"



int main()
{
	std::string trt_path = "F:/C++ projects/VGG_tensorrt/VGG_tensorrt/MyVGG.trt";		// trt模型地址
	std::string img_path = "F:/C++ projects/VGG_tensorrt/VGG_tensorrt/2.bmp";		// 预测图片文件夹位置

	const int batchSize = 1;				// 预测batch_size
	const int input_channel = 3;			// 输入的通道数
	const int input_height = 48;
	const int input_width = 48;				// 输入的高和宽
	const int outputSize = 1;				

	// 构建engine
	std::vector<unsigned char> model_data = utils::loadModel(trt_path);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());	// 创建推理的runtime实例化接口
	nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());	// 反序列化模型得到engine
	nvinfer1::IExecutionContext *context = engine->createExecutionContext();		// 构建context保存参数

	// 分配输入和输出内存
	//void* buffers[2];	// 为输入、输出设置缓冲器
	const char* INPUT_BLOB_NAME = "input";
	const char* OUTPUT_BLOB_NAME = "output";
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);	// 得到engine中输入和输出的索引

	// cuda malloc
	void* input_mem{ nullptr };
	cudaMalloc(&input_mem, batchSize * input_channel * input_height * input_width * sizeof(float));
	void* output_mem{ nullptr };
	cudaMalloc(&output_mem, batchSize * outputSize * sizeof(float));

	std::vector<float> outputData(batchSize * outputSize);
	//float outputData[4];

	// 输入图片预处理
	float* inputData = utils::preprocess(img_path, input_height, input_width);
	/*std::cout << *inputData << std::endl;*/
	//for (int i = 0; i < 3 * input_height * input_width; i++)
	//{
	//	std::cout << i <<":"<< *&inputData[i] << std::endl;
	//}

	// 创建cuda流
	cudaStream_t stream;
	cudaStreamCreate(&stream);
		// cudaMemcpy 同步执行	cudaMemcpyAsync 异步执行	将输入数据从cpu拷贝到gpu中
	cudaMemcpyAsync(input_mem, inputData, batchSize * input_channel * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 执行推理
	void* bindings[] = { input_mem, output_mem };
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < outputSize; i++)
	{
		//context->enqueue(batchSize, buffers, stream, nullptr);
		context->enqueueV2(bindings, stream, nullptr);
	}
	auto end = std::chrono::system_clock::now(); // 结束时间
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	// 将输出从gpu拷贝到cpu中
	cudaMemcpyAsync(outputData.data(), output_mem, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
	std::cout << outputData[0] << std::endl;

	std::cout << "Top Predictions:" << std::endl;
	for (int i = 0; i < 1; i++)
	{
		float maxVal = -0.5f;
		float maxIndex = -1;
		for (int j = 0; j < outputSize; j++)
		{
			if (outputData[j] > maxVal)
			{
				maxVal = outputData[j];
				maxIndex = j;
			}
		}
		std::cout << "Class " << maxIndex << ": " << maxVal << std::endl;
		outputData[maxIndex] = -1.0f; // 将该类别得分置为负数，确保不会重复输出
	}

	// 释放
	context->destroy();
	engine->destroy();
	runtime->destroy();
}