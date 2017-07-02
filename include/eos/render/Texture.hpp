/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/Texture.hpp
 *
 * Copyright 2017 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <algorithm>

namespace eos {
	namespace render {

// TODO: Should go to detail:: namespace, or texturing/utils or whatever.
inline unsigned int get_max_possible_mipmaps_num(unsigned int width, unsigned int height)
{
	unsigned int mipmapsNum = 1;
	unsigned int size = std::max(width, height);

	if (size == 1)
		return 1;

	do {
		size >>= 1;
		mipmapsNum++;
	} while (size != 1);

	return mipmapsNum;
};

inline bool is_power_of_two(int x)
{
	return !(x & (x - 1));
};

/**
 * @brief Represents a texture for rendering.
 * 
 * Represents a texture and mipmap levels for use in the renderer.
 * Todo: This whole class needs a major overhaul and documentation.
 */
class Texture
{
public:
	std::vector<cv::Mat> mipmaps;	// make Texture a friend class of renderer, then move this to private?
	unsigned char widthLog, heightLog; // log2 of width and height of the base mip-level

//private:
	//std::string filename;
	unsigned int mipmaps_num;
};

// throws: ocv exc,  runtime_ex
inline Texture create_mipmapped_texture(cv::Mat image, unsigned int mipmapsNum = 0)
{
	assert(image.type() == CV_8UC3 || image.type() == CV_8UC4);

	Texture texture;

	texture.mipmaps_num = (mipmapsNum == 0 ? get_max_possible_mipmaps_num(image.cols, image.rows) : mipmapsNum);
	/*if (mipmapsNum == 0)
	{
	uchar mmn = render::utils::MatrixUtils::getMaxPossibleMipmapsNum(image.cols, image.rows);
	this->mipmapsNum = mmn;
	} else
	{
	this->mipmapsNum = mipmapsNum;
	}*/

	if (texture.mipmaps_num > 1)
	{
		if (!is_power_of_two(image.cols) || !is_power_of_two(image.rows))
		{
			throw std::runtime_error("Error: Couldn't generate mipmaps, width or height not power of two.");
		}
	}
	if (image.type() == CV_8UC3)
	{
		image.convertTo(image, CV_8UC4); // Most often, the input img is CV_8UC3. Img is BGR. Add an alpha channel. TODO: Actually I think this doesn't do anything. Below line adds the 4th channel...
		cv::cvtColor(image, image, CV_BGR2BGRA);
	}

	int currWidth = image.cols;
	int currHeight = image.rows;
	std::vector<cv::Mat> mipmaps;
	for (int i = 0; i < texture.mipmaps_num; i++)
	{
		if (i == 0) {
			mipmaps.push_back(image);
		}
		else {
			cv::Mat currMipMap(currHeight, currWidth, CV_8UC4);
			cv::resize(mipmaps[i - 1], currMipMap, currMipMap.size());
			mipmaps.push_back(currMipMap);
		}

		if (currWidth > 1)
			currWidth >>= 1;
		if (currHeight > 1)
			currHeight >>= 1;
	}
	texture.mipmaps = mipmaps;
	texture.widthLog = (uchar)(std::log(mipmaps[0].cols) / CV_LOG2 + 0.0001f); // std::epsilon or something? or why 0.0001f here?
	texture.heightLog = (uchar)(std::log(mipmaps[0].rows) / CV_LOG2 + 0.0001f); // Changed std::logf to std::log because it doesnt compile in linux (gcc 4.8). CHECK THAT
	return texture;
};

	} /* namespace render */
} /* namespace eos */

#endif /* TEXTURE_HPP_ */
