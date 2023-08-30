#include"utils.h"
#include<opencv.hpp>



std::vector<unsigned char> utils::loadModel(const std::string& file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

float* utils::preprocess(const std::string imgPath, const int input_h, int input_w)
{
	cv::Mat origin_img = cv::imread(imgPath);
	//std::cout << origin_img << std::endl;
	//std::cout << "----------------------------------------------------------------------" << std::endl;
	//std::cout << origin_img.data << std::endl;	// cv::Mat.data：unchar类型指针，指向Mat数据矩阵的首地址
	cv::Mat img;
	cv::cvtColor(origin_img, img, cv::COLOR_BGR2RGB);
	//std::cout << img << std::endl;
	//std::cout << "----------------------------------------------------------------------" << std::endl;
	float* data = new float[3 * input_h * input_w];
	cv::Mat src_img;	// 构建输入的实体
	cv::resize(img, src_img, cv::Size(input_w, input_h));
	//std::cout << src_img << std::endl;
	//std::cout << "----------------------------------------------------------------------" << std::endl;
	// 获取Mat对象的像素块的数据指针，基于指针操作，实现快速像素方法
	// noramlize
	int count = 0;
	std::vector<float> mean_value = { 0.3150, 0.0619, 0.1087 };
	std::vector<float> std_value = { 0.2419, 0.0771, 0.2342 };
	for (int row = 0; row < input_h; row++)
	{
		uchar* uc_pixel = src_img.data + row * src_img.step;
		for (int col = 0; col < input_w; col++)
		{	// bgr存放
			data[count] = (uc_pixel[0] / 255. - mean_value[0]) / std_value[0];	// B
			data[count + src_img.rows * src_img.cols] = (uc_pixel[1] / 255. - mean_value[1]) / std_value[1];	// G
			data[count + src_img.rows * src_img.cols * 2] = (uc_pixel[2] / 255. - mean_value[2]) / std_value[2];	// R
			uc_pixel += 3;
			count++;
		}
	}
	//std::cout << *data << std::endl;
	return data;
}

//float* utils::preprocess(const std::string imgPath, int input_h, int input_w)
//{
//    cv::Mat MatBGRImage = cv::imread(imgPath);
//    cv::Mat RGBImg, ResizeImg;
//    cvtColor(MatBGRImage, RGBImg, cv::COLOR_BGR2RGB);
//    cv::resize(RGBImg, ResizeImg, cv::Size(input_h, input_w));
//
//    int channels = ResizeImg.channels(), height = ResizeImg.rows, width = ResizeImg.cols;
//
//    float* nchwMat = (float*)malloc(channels * height * width * sizeof(float));
//    memset(nchwMat, 0, channels * height * width * sizeof(float));
//
//    // Convert HWC to CHW and Normalize
//    float mean_rgb[3] = { 0.3150, 0.0619, 0.1087 };
//    float std_rgb[3] = { 0.2419, 0.0771, 0.2342 };
//    uint8_t* ptMat = ResizeImg.ptr<uint8_t>(0);
//    int area = height * width;
//    for (int c = 0; c < channels; ++c)
//    {
//        for (int h = 0; h < height; ++h)
//        {
//            for (int w = 0; w < width; ++w)
//            {
//                int srcIdx = c * area + h * width + w;
//                int divider = srcIdx / 3;  // 0, 1, 2
//                for (int i = 0; i < 3; ++i)
//                {
//                    nchwMat[divider + i * area] = static_cast<float>((ptMat[srcIdx] * 1.0f / 255.0f - mean_rgb[i]) * 1.0f / std_rgb[i]);
//                }
//            }
//        }
//    }
//    return nchwMat;
//}