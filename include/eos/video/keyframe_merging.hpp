/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/video/keyframe_merging.hpp
 *
 * Copyright 2018 Patrik Huber
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

#ifndef KEYFRAME_MERGING_HPP_
#define KEYFRAME_MERGING_HPP_

#include "eos/core/Image_opencv_interop.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/video/Keyframe.hpp"

#include "Eigen/Core"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cassert>
#include <vector>

namespace eos {
namespace video {

/**
 * @brief Extracts texture from each keyframe and merges them using a weighted mean.
 *
 * Uses the view angle as weighting.
 *
 * Note 1: Would be nice to eventually return a 4-channel texture map, with a sensible weight in the 4th
 * channel (i.e. the max of all weights for a given pixel).
 *
 * Note 2: On each call to this, it generates all isomaps. This is quite time-consuming (and we could compute
 * the weighted mean incrementally). But caching them is not trivial (maybe with a hashing or comparing the
 * cv::Mat frame data* member?).
 * On the other hand, for the more complex merging techniques (super-res, involving ceres, or a median
 * cost-func?), there might be no caching possible anyway and we will recompute the merged isomap from scratch
 * each time anyway, but not by first extracting all isomaps - instead we would just do a lookup of the
 * required pixel value(s) in the original image.
 *
 * // struct KeyframeMerger {};
 *
 * @param[in] keyframes The keyframes that will be merged.
 * @param[in] morphable_model The Morphable Model with which the keyframes have been fitted.
 * @param[in] blendshapes The blendshapes with which the keyframes have been fitted.
 * @return Merged texture map (isomap), 3-channel uchar.
 */
inline cv::Mat merge_weighted_mean(const std::vector<Keyframe<cv::Mat>>& keyframes,
                            const morphablemodel::MorphableModel& morphable_model,
                            const std::vector<morphablemodel::Blendshape>& blendshapes)
{
    assert(keyframes.size() >= 1);

    using cv::Mat;
    using Eigen::VectorXf;
    using std::vector;

    vector<Mat> isomaps;
    for (const auto& frame_data : keyframes)
    {
        const VectorXf shape =
            morphable_model.get_shape_model().draw_sample(frame_data.fitting_result.pca_shape_coefficients) +
            morphablemodel::to_matrix(blendshapes) *
                Eigen::Map<const Eigen::VectorXf>(frame_data.fitting_result.expression_coefficients.data(),
                                                  frame_data.fitting_result.expression_coefficients.size());
        const auto mesh =
            morphablemodel::sample_to_mesh(shape, {}, morphable_model.get_shape_model().get_triangle_list(),
                                           {}, morphable_model.get_texture_coordinates());
        const auto affine_camera_matrix = fitting::get_3x4_affine_camera_matrix(
            frame_data.fitting_result.rendering_parameters, frame_data.frame.cols, frame_data.frame.rows);
        const Mat isomap = core::to_mat(render::extract_texture(mesh, affine_camera_matrix, core::from_mat(frame_data.frame), true,
                                                   render::TextureInterpolation::NearestNeighbour, 1024));
        isomaps.push_back(isomap);
    }

    // Now do the actual merging:
    Mat r = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    Mat g = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    Mat b = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    Mat accumulated_weight = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    // Currently, this just uses the weights in the alpha channel for weighting - they contain only the
    // view-angle. We should use the keyframe's score as well. Plus the area of the source triangle.
    for (auto&& isomap : isomaps)
    {
        vector<Mat> channels;
        cv::split(isomap, channels);
        // channels[0].convertTo(channels[0], CV_32FC1);
        // We could avoid this explicit temporary, but then we'd have to convert both matrices
        // to CV_32FC1 first - and manually multiply with 1/255. Not sure which one is faster.
        // If we do it like this, the add just becomes '+=' - so I think it's fine like this.
        // The final formula is:
        // b += chan_0 * alpha * 1/255; (and the same for g and r respectively)
        Mat weighted_b, weighted_g, weighted_r;
        // // we scale the weights from [0, 255] to [0, 1]:
        cv::multiply(channels[0], channels[3], weighted_b, 1 / 255.0, CV_32FC1);
        cv::multiply(channels[1], channels[3], weighted_g, 1 / 255.0, CV_32FC1);
        cv::multiply(channels[2], channels[3], weighted_r, 1 / 255.0, CV_32FC1);
        b += weighted_b;
        g += weighted_g;
        r += weighted_r;
        channels[3].convertTo(channels[3], CV_32FC1); // needed for the '/ 255.0f' below to work
        cv::add(accumulated_weight, channels[3] / 255.0f, accumulated_weight, cv::noArray(), CV_32FC1);
    }
    b = b.mul(1.0 / (accumulated_weight)); // divide by number of frames used too?
    g = g.mul(1.0 / (accumulated_weight));
    r = r.mul(1.0 / (accumulated_weight));

    // Let's return accumulated_weight too: Normalise by num_isomaps * 255 (=maximum weight)
    // This sets the returned weight to the average from all the isomaps. Maybe the maximum would make more
    // sense? => Not returning anything for now.
    // accumulated_weight = (accumulated_weight / isomaps.size()) * 255;

    Mat merged_isomap;
    cv::merge(vector<Mat>{b, g, r}, merged_isomap);
    merged_isomap.convertTo(merged_isomap, CV_8UC3);
    return merged_isomap;
};

/**
 * @brief Computes the variance of laplacian of the given image or patch.
 *
 * This should compute the variance of the laplacian of a given image or patch, according to the 'LAPV'
 * algorithm of Pech 2000.
 * It is used as a focus or blurriness measure, i.e. to assess the quality of the given patch.
 *
 * @param[in] image Input image or patch.
 * @return The computed variance of laplacian score.
 */
inline double variance_of_laplacian(const cv::Mat& image)
{
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_64F);

    cv::Scalar mu, sigma;
    cv::meanStdDev(laplacian, mu, sigma);

    const double focus_measure = sigma.val[0] * sigma.val[0];
    return focus_measure;
};

} /* namespace video */
} /* namespace eos */

#endif /* KEYFRAME_MERGING_HPP_ */
