/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/TriangleToRasterize.hpp
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

#ifndef EOS_TRIANGLE_TO_RASTERIZE_HPP
#define EOS_TRIANGLE_TO_RASTERIZE_HPP

#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/plane.hpp"

/**
 * The detail namespace contains implementations of internal functions, not part of the API we expose and not
 * meant to be used by a user.
 */
namespace eos {
namespace render {
namespace detail {

/**
 * A representation for a triangle that is to be rasterised.
 * Stores the enclosing bounding box of the triangle that is
 * calculated during rendering and used during rasterisation.
 *
 * Used in render_affine and render.
 */
struct TriangleToRasterize
{
    Vertex<float> v0, v1, v2;
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    // Everything below is only used in the "normal" renderer, but not
    // in render_affine.
    double one_over_z0;
    double one_over_z1;
    double one_over_z2;
    // double one_over_v0ToLine12;
    // double one_over_v1ToLine20;
    // double one_over_v2ToLine01;
    plane alphaPlane;
    plane betaPlane;
    plane gammaPlane;
    double one_over_alpha_c; // those are only used for texturing -> float
    double one_over_beta_c;
    double one_over_gamma_c;
    float alpha_ffx;
    float beta_ffx;
    float gamma_ffx;
    float alpha_ffy;
    float beta_ffy;
    float gamma_ffy;
};

} /* namespace detail */
} /* namespace render */
} /* namespace eos */

#endif /* EOS_TRIANGLE_TO_RASTERIZE_HPP */
