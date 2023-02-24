/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/plane.hpp
 *
 * Copyright 2017, 2023 Patrik Huber
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

#ifndef EOS_RENDER_DETAIL_PLANE_HPP
#define EOS_RENDER_DETAIL_PLANE_HPP

#include "Eigen/Core"

#include <cmath>

/**
 * The detail namespace contains implementations of internal functions, not part of the API we expose and not
 * meant to be used by a user.
 */
namespace eos {
namespace render {
namespace detail {

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

    plane(const Eigen::Vector3f& normal, float d = 0.0f)
    {
        this->a = normal[0];
        this->b = normal[1];
        this->c = normal[2];
        this->d = d;
    }

    plane(const Eigen::Vector3f& point, const Eigen::Vector3f& normal)
    {
        a = normal[0];
        b = normal[1];
        c = normal[2];
        d = -point.dot(normal);
    }

    template <typename T>
    plane(const Eigen::Vector3<T>& point1, const Eigen::Vector3<T>& point2, const Eigen::Vector3<T>& point3)
    {
        const Eigen::Vector3<T> v1 = point2 - point1;
        const Eigen::Vector3<T> v2 = point3 - point1;
        Eigen::Vector3<T> normal = v1.cross(v2);

        normal = normal.normalized();

        a = normal[0];
        b = normal[1];
        c = normal[2];
        d = -point1.dot(normal);
    }

    void normalize()
    {
        const float length = std::sqrt(a * a + b * b + c * c);

        a /= length;
        b /= length;
        c /= length;
    }

    float getSignedDistanceFromPoint(const Eigen::Vector3f& point) const
    {
        return a * point[0] + b * point[1] + c * point[2] + d;
    }

    float getSignedDistanceFromPoint(const Eigen::Vector4f& point) const
    {
        return a * point[0] + b * point[1] + c * point[2] + d;
    }

public:
    float a, b, c;
    float d;
};

} /* namespace detail */
} /* namespace render */
} /* namespace eos */

#endif /* EOS_RENDER_DETAIL_PLANE_HPP */
