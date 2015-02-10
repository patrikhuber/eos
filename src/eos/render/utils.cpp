/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: src/eos/render/utils.cpp
 *
 * Copyright 2014, 2015 Patrik Huber
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
#include "eos/render/utils.hpp"

using cv::Vec2f;

namespace eos {
	namespace render {

Vec2f clipToScreenSpace(Vec2f clipCoordinates, int screenWidth, int screenHeight)
{
	// Window transform:
	float x_ss = (clipCoordinates[0] + 1.0f) * (screenWidth / 2.0f);
	float y_ss = screenHeight - (clipCoordinates[1] + 1.0f) * (screenHeight / 2.0f); // also flip y; Qt: Origin top-left. OpenGL: bottom-left.
	return Vec2f(x_ss, y_ss);
	/* Note: What we do here is equivalent to
	   x_w = (x *  vW/2) + vW/2;
	   However, Shirley says we should do:
	   x_w = (x *  vW/2) + (vW-1)/2;
	   (analogous for y)
	   Todo: Check the consequences.
	*/
}

Vec2f screenToClipSpace(Vec2f screenCoordinates, int screenWidth, int screenHeight)
{
	float x_cs = screenCoordinates[0] / (screenWidth / 2.0f) - 1.0f;
	float y_cs = screenCoordinates[1] / (screenHeight / 2.0f) - 1.0f;
	y_cs *= -1.0f;
	return Vec2f(x_cs, y_cs);
}

	} /* namespace render */
} /* namespace eos */
