/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/utils.hpp
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
#pragma once

#ifndef RENDER_UTILS_HPP_
#define RENDER_UTILS_HPP_

#include "opencv2/core/core.hpp"

namespace eos {
	namespace render {

/**
 * Transforms a point from clip space ([-1, 1] x [-1, 1]) to
 * image (screen) coordinates, i.e. the window transform.
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 * No z-division is performed.
 * Note: It should rather be called from NDC to screen space?
 *
 * Exactly conforming to the OpenGL viewport transform, except that
 * we flip y at the end.
 * Qt: Origin top-left. OpenGL: bottom-left. OCV: top-left.
 *
 * @param[in] clip_coordinates A point in clip coordinates.
 * @param[in] screen_width Width of the screen or window.
 * @param[in] screen_height Height of the screen or window.
 * @return A vector with x and y coordinates transformed to screen space.
 */
inline cv::Vec2f clip_to_screen_space(cv::Vec2f clip_coordinates, int screen_width, int screen_height)
{
	// Window transform:
	float x_ss = (clip_coordinates[0] + 1.0f) * (screen_width / 2.0f);
	float y_ss = screen_height - (clip_coordinates[1] + 1.0f) * (screen_height / 2.0f); // also flip y; Qt: Origin top-left. OpenGL: bottom-left.
	return cv::Vec2f(x_ss, y_ss);
	/* Note: What we do here is equivalent to
	   x_w = (x *  vW/2) + vW/2;
	   However, Shirley says we should do:
	   x_w = (x *  vW/2) + (vW-1)/2;
	   (analogous for y)
	   Todo: Check the consequences.
	*/
};

/**
 * Transforms a point from image (screen) coordinates to
 * clip space ([-1, 1] x [-1, 1]).
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 *
 * @param[in] screenCoordinates A point in screen coordinates.
 * @param[in] screenWidth Width of the screen or window.
 * @param[in] screenHeight Height of the screen or window.
 * @return A vector with x and y coordinates transformed to clip space.
 */
cv::Vec2f screenToClipSpace(cv::Vec2f screenCoordinates, int screenWidth, int screenHeight);

	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_UTILS_HPP_ */
