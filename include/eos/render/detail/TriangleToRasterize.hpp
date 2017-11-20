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

#ifndef TRIANGLETORASTERIZE_HPP_
#define TRIANGLETORASTERIZE_HPP_

#include "eos/render/detail/Vertex.hpp"

#include "glm/glm.hpp"

#include <cmath>

/**
 * The detail namespace contains implementations of internal functions, not part of the API we expose and not
 * meant to be used by a user.
 */
namespace eos {
namespace render {
namespace detail {

// plane should probably go into its own file as well.
class plane
{
public:
    plane() {}

    plane(float a, float b, float c, float d)
    {
        this->a = a;
        this->b = b;
        this->c = c;
        this->d = d;
    }

    plane(const glm::vec3& normal, float d = 0.0f)
    {
        this->a = normal[0];
        this->b = normal[1];
        this->c = normal[2];
        this->d = d;
    }

    plane(const glm::vec3& point, const glm::vec3& normal)
    {
        a = normal[0];
        b = normal[1];
        c = normal[2];
        d = -glm::dot(point, normal);
    }

    template <typename T, glm::precision P = glm::defaultp>
    plane(const glm::tvec3<T, P>& point1, const glm::tvec3<T, P>& point2, const glm::tvec3<T, P>& point3)
    {
        const glm::tvec3<T, P> v1 = point2 - point1;
        const glm::tvec3<T, P> v2 = point3 - point1;
        glm::tvec3<T, P> normal = glm::cross(v1, v2);
        normal = glm::normalize(normal);

        a = normal[0];
        b = normal[1];
        c = normal[2];
        d = -glm::dot(point1, normal);
    }

    void normalize()
    {
        float length = std::sqrt(a * a + b * b + c * c);

        a /= length;
        b /= length;
        c /= length;
    }

    float getSignedDistanceFromPoint(const glm::vec3& point) const
    {
        return a * point[0] + b * point[1] + c * point[2] + d;
    }

    float getSignedDistanceFromPoint(const glm::vec4& point) const
    {
        return a * point[0] + b * point[1] + c * point[2] + d;
    }

public:
    float a, b, c;
    float d;
};

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

#endif /* TRIANGLETORASTERIZE_HPP_ */
