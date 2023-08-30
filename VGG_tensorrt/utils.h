#pragma once
#include<iostream>
#include<vector>
#include<fstream>

namespace utils
{
	struct InitParameter
	{
		int img_size;
		float anomalyThresh;
		
	};

	std::vector<unsigned char> loadModel(const std::string& file);
	float* preprocess(const std::string imgPath, int input_h, int input_w);

}
