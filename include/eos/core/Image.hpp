/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/Image.hpp
 *
 * Copyright 2017, 2018 Patrik Huber
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

#ifndef EOS_IMAGE_HPP
#define EOS_IMAGE_HPP

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace eos {
namespace core {

/**
 * @brief Representation of an image with \p num_channels channels.
 *
 * This is a quickly hacked-together implementation of an image class, to be able
 * to represent 1 and 3-channel images. The class was mainly created to be able to
 * remove OpenCV and cv::Mat as a dependency for the core of the eos headers.
 *
 * The class currently uses col-major storage order.
 *
 * Note: Ideally we'd have a template param for row and col major too?
 *       And then we could do stuff like strides and all too but then we'd end up with a matrix class like Eigen::Matrix...?
 *       It means we probably can't accept ROI's from Python into larger images without copying, but doesn't matter.
 *       Also what about vector<Eigen::Matrix>(num_channels)? But having an own type would be good?
 */
template <class T, int num_channels>
class Image
{
    // we store RGB RGB RGB ... col or rows first? Do the same as Eigen:col-major?
    // numpy: 'f' = fortran = col-major
    //        'c' = c = row-major = default I think.
public:
    // using element_type = T;

    Image() = default;

    Image(std::size_t rows, std::size_t cols) : rows(rows), cols(cols)
    {
        // Question: Encode the channel into the plain array or not?
        // data.reserve(rows * cols * num_channels);
        // data.reserve(rows * cols);
        data.resize(rows * cols); // This actually zero-initialises, with std::array<>.
    };

    // We can't just return T. Should be T* to return RGB or RGBA?
    // And actually we may want to return a reference, or not? Well, it's only 1-4 values, we can copy them?
    // But this must also act as a setter actually! So needs to return a reference? And a const& overload?
    // Can we do a operator<>()? But we don't need it as in comparison to OpenCV, we know the type at compile
    // time...
    // hmm... SFINAE with num_channels == 1 or something?
    // => I'm over-engineering. Go for simple solution first!
    // If we use array<uint8_t, 3> as type T for a 3-channel image, then this operator works out of the box.
    T& operator()(std::size_t row, std::size_t col)
    {
        assert(row < rows);
        assert(col < cols);
        assert(row + col * rows < data.size());
        return data[row + col * rows]; // Col-major for now.
        // return data[col + row * cols]; // Ok, switching to row-major, to test with numpy.
        /*
         * [a b c]
         * [d e f]
         * // => [a d b e c f]
         *
         * (0, 0) => a
         * (0, 1) => b
         * (1, 0) => d
         * (1, 2) => f
         */
    };

    const T& operator()(std::size_t row, std::size_t col) const
    {
        assert(row < rows);
        assert(col < cols);
        assert(row + col * rows < data.size());
        return data[row + col * rows]; // Col-major for now.
        // return data[col + row * cols]; // Ok, switching to row-major, to test with numpy.
    };

    /*	static Image<T, num_channels> zeros(std::size_t rows, std::size_t cols)
            {
                    Image<T, num_channels> image(rows, cols);
                    for (int c=0; c < image.cols; ++c) {
                            for (int r = 0; r < image.rows; ++r) {
                                    image(r, c) = T{ 1 }; // Does this create an std::array with all zeros,
       like we expect? No it doesn't, it only sets the first component to 0, and leaves the rest.
                            }
                    }
                    return image;
            }; */

    // private:
    std::vector<T> data;  // Maybe not too ideal, see the zeros(...) function. Should rather encode [RGB...]
                          // etc in here too, directly. Maybe.
    std::size_t rows = 0; // nobody should be able to set these directly, so.. yea... private.
    std::size_t cols = 0;
    // Maybe array_view (span) is the way to go...? But nothing coming to the standard anytime soon...
};

// Note: The num_channels number needs to be repeated, not so nice.
using Image1u = Image<std::uint8_t, 1>;
using Image3u = Image<std::array<std::uint8_t, 3>, 3>;
using Image4u = Image<std::array<std::uint8_t, 4>, 4>;
using Image1d = Image<double, 1>;

} /* namespace core */
} /* namespace eos */

#endif /* EOS_IMAGE_HPP */
