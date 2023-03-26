/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/RenderingParameters.hpp
 *
 * Copyright 2016, 2023 Patrik Huber
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

#ifndef EOS_RENDERING_PARAMETERS_HPP
#define EOS_RENDERING_PARAMETERS_HPP

#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/rotation_angles.hpp"
#include "eos/render/matrix_projection.hpp"
#include "eos/cpp17/optional.hpp"
#include "eos/cpp17/optional_serialization.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"

#include "eos/fitting/detail/glm_cerealisation.hpp"
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"

#include "Eigen/Core"

#include <string>
#include <fstream>
#include <stdexcept>

namespace eos {
namespace fitting {

/**
 * @brief A class representing a camera viewing frustum. At the moment only
 * fully tested with orthographic camera.
 */
struct Frustum
{
    Frustum(){};
    Frustum(float l, float r, float b, float t) : l(l), r(r), b(b), t(t){};
    // Frustum(float l, float r, float b, float t, float n, float f) : l(l), r(r), b(b), t(t), n(n),
    // f(f) {};
    float l = -1.0f;
    float r = 1.0f;
    float b = -1.0f;
    float t = 1.0f;
    // std::optional<float> n; // These are not needed yet but probably will in the future,
    // std::optional<float> f; // and then it's good if the old serialised files stay compatible.

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] archive The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(CEREAL_NVP(l), CEREAL_NVP(r), CEREAL_NVP(b), CEREAL_NVP(t));
    };
};

enum class CameraType
{
    Orthographic,
    Perspective
};

/**
 * @brief Represents a set of estimated model parameters (rotation, translation) and
 * camera parameters (viewing frustum).
 *
 * The estimated rotation and translation transform the model from model-space to camera-space,
 * and, if one wishes to use OpenGL, can be used to build the model-view matrix, i.e. the parameter
 * representation is OpenGL compliant.
 *
 * The parameters are the inverse of the camera position in 3D space.
 *
 * The camera frustum describes the size of the viewing plane of the camera, and
 * can be used to build an OpenGL-conformant orthographic projection matrix.
 *
 * Together, these parameters fully describe the imaging process of a given model instance.
 *
 * The rotation is represented using a quaternion. However, the quaternion can be converted to a rotation
 * matrix, and subsequently Euler angles can be extracted. In our coordinate system, the x axis points to the
 * right, y points upwards, and z into/out of the screen. When extracting Euler angles (or Tait-Bryan angles,
 * to be more precise), we assume rotation around the y axis first (yaw), then around x axis (pitch), then
 * around z (roll).
 */
class RenderingParameters
{
public:
    // Creates with default frustum...
    RenderingParameters() = default;

    // Initialisation for Eigen::LevMarq
    // This creates the correct rotation quaternion in the case the angles were estimated/given by
    // R*P*Y*v. Angles given in radian. Note: If you subsequently use RP::get_rotation() and
    // glm::eulerAngles() on it, the angles you get out will be slightly different from the ones you
    // put in here. But they will describe the same rotation! Just in a different order. (i.e. the
    // rotation matrix or quaternion for both of these two sets of angles is identical.)
    RenderingParameters(CameraType camera_type, Frustum camera_frustum, float r_x, float r_y, float r_z,
                        float tx, float ty, int screen_width, int screen_height)
        : camera_type(camera_type), frustum(camera_frustum), t_x(tx), t_y(ty), screen_width(screen_width),
          screen_height(screen_height)
    {
        const auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), r_x, glm::vec3{1.0f, 0.0f, 0.0f});
        const auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), r_y, glm::vec3{0.0f, 1.0f, 0.0f});
        const auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), r_z, glm::vec3{0.0f, 0.0f, 1.0f});
        const auto zxy_rotation_matrix = rot_mtx_z * rot_mtx_x * rot_mtx_y;
        glm::quat glm_rotation_quat = glm::quat(zxy_rotation_matrix);
        rotation = Eigen::Quaternionf(glm_rotation_quat.w, glm_rotation_quat.x, glm_rotation_quat.y,
                                      glm_rotation_quat.z);
    };

    // This assumes estimate_sop was run on points with OpenCV viewport! I.e. y flipped.
    RenderingParameters(ScaledOrthoProjectionParameters ortho_params, int screen_width, int screen_height)
        : camera_type(CameraType::Orthographic), t_x(ortho_params.tx), t_y(ortho_params.ty),
          screen_width(screen_width), screen_height(screen_height)
    {
        rotation = ortho_params.R; // converts the rotation matrix to a quaternion

        const float l = 0.0f;
        const float r = screen_width / ortho_params.s;
        const float b = 0.0f; // The b and t values are not tested for what happens if the SOP parameters
        const float t = screen_height / ortho_params.s; // were estimated on a non-flipped viewport.
        frustum = Frustum(l, r, b, t);
    };

    // Note: Doesn't set up the Frustum currently. Need to re-think this design a bit anyway.
    RenderingParameters(Eigen::Quaternionf rotation, Eigen::Vector3f translation, float fov_y, int image_width,
                        int image_height)
        : camera_type(CameraType::Perspective), rotation(rotation), t_x(translation.x()), t_y(translation.y()),
          t_z(translation.z()), fov_y(fov_y), screen_width(image_width), screen_height(image_height){};

    auto get_camera_type() const
    {
        return camera_type;
    };

    Eigen::Quaternionf get_rotation() const
    {
        return rotation;
    };

    void set_rotation(Eigen::Quaternionf rotation_quaternion)
    {
        rotation = rotation_quaternion;
    };

    /**
     * @brief Returns the intrinsic rotation angles, also called Tait-Bryan angles, in degrees. The returned
     * Vector3f contains [yaw, pitch, roll].
     *
     * The order of rotations is yaw, pitch, roll (in our coordinate system, y, x, z).
     *
     * - Positive pitch means the head is looking down
     * - Positive yaw means the head is looking to the left (from the subject's perspective, i.e. we see their
     *   right cheek)
     * - Positive roll means the head's right eye is further down than the other one (the head is tilted to
     *   the right, from the subject's perspective).
     */
    Eigen::Vector3f get_yaw_pitch_roll()
    {
        // We use Eigen's Matrix3::eulerAngles() function with axes (1, 0, 2). In our coordinate system, the x
        // axis is to the right, y up, and z into/out of the screen. So to get yaw first, our first axis is y
        // ('1'), then our x axis ('0') to get pitch, then our z axis ('2') to get roll.
        // We are guessing that the function returns intrinsic angles (the unsupported EulerAngles() module
        // does, but it's an unrelated module).
        // Because eulerAngles() constrains the angles in a different way to what we want, we correct for
        // that in tait_bryan_angles() (which calls eulerAngles() and then modifies the solution accordingly).
        Eigen::Vector3f ea = fitting::tait_bryan_angles(rotation.normalized().toRotationMatrix(), 1, 0, 2);
        ea(0) = core::degrees(ea(0));
        ea(1) = core::degrees(ea(1));
        ea(2) = core::degrees(ea(2));
        return ea;
    };

    void set_translation(float t_x, float t_y, cpp17::optional<float> t_z = cpp17::nullopt)
    {
        this->t_x = t_x;
        this->t_y = t_y;
        if (t_z)
        {
            this->t_z = t_z;
        }
    };

    // Third coord is 0.0 for ortho (==no t_z).
    Eigen::Vector3f get_translation() const
    {
        return {t_x, t_y, t_z.value_or(0.0f)};
    };

    cpp17::optional<float> get_fov_y() const
    {
        return fov_y;
    };

    /**
     * @brief Construct a model-view matrix from the RenderingParameters' rotation and translation, and return
     * it.
     */
    Eigen::Matrix4f get_modelview() const
    {
        if (camera_type == CameraType::Orthographic)
        {
            Eigen::Matrix4f modelview = Eigen::Matrix4f::Identity();
            modelview.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
            modelview(0, 3) = t_x;
            modelview(1, 3) = t_y;
            return modelview;
        } else
        {
            assert(t_z.has_value()); // Might be worth throwing an exception instead.
            Eigen::Matrix4f modelview = Eigen::Matrix4f::Identity();
            modelview.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
            modelview(0, 3) = t_x;
            modelview(1, 3) = t_y;
            modelview(2, 3) = t_z.value();
            return modelview;
        }
    };

    /**
     * @brief Construct an orthographic or perspective projection matrix from the RenderingParameters' frustum
     * (orthographic) or fov_y and aspect ration (perspective), and return it.
     */
    Eigen::Matrix4f get_projection() const
    {
        if (camera_type == CameraType::Orthographic)
        {
            const auto ortho = render::ortho(frustum.l, frustum.r, frustum.b, frustum.t);
            return ortho;
        } else
        {
            assert(fov_y.has_value()); // Might be worth throwing an exception instead.
            const float aspect_ratio = static_cast<double>(screen_width) / screen_height;
            const auto persp_mtx = render::perspective(fov_y.value(), aspect_ratio, 0.1f, 1000.0f);
            return persp_mtx;
        }
    };

    Frustum get_frustum() const { return frustum; };

    void set_frustum(Frustum frustum) { this->frustum = frustum; };

    int get_screen_width() const { return screen_width; };

    void set_screen_width(int screen_width) { this->screen_width = screen_width; };

    int get_screen_height() const { return screen_height; };

    void set_screen_height(int screen_height) { this->screen_height = screen_height; };

private:
    CameraType camera_type = CameraType::Orthographic;
    Frustum frustum; // Can construct a glm::ortho or glm::perspective matrix from this.

    Eigen::Quaternionf rotation;

    float t_x;
    float t_y;
    cpp17::optional<float> t_z;

    cpp17::optional<float> fov_y; // Field of view in the y direction. Degree or radians? Only for certain
                                  // camera types. Should it go into Frustum?

    int screen_width; // (why) do we need these?
    int screen_height;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(CEREAL_NVP(camera_type), CEREAL_NVP(frustum), CEREAL_NVP(rotation), CEREAL_NVP(t_x),
                CEREAL_NVP(t_y), CEREAL_NVP(t_z), CEREAL_NVP(fov_y), CEREAL_NVP(screen_width),
                CEREAL_NVP(screen_height));
    };
};

/**
 * Saves the rendering parameters for an image to a json file.
 *
 * @param[in] rendering_parameters An instance of class RenderingParameters.
 * @param[in] filename The file to write.
 * @throws std::runtime_error if unable to open the given file for writing.
 */
inline void save_rendering_parameters(RenderingParameters rendering_parameters, std::string filename)
{
    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Error opening file for writing: " + filename);
    }
    cereal::JSONOutputArchive output_archive(file);
    output_archive(cereal::make_nvp("rendering_parameters", rendering_parameters));
};

/**
 * @brief Returns a glm/OpenGL compatible viewport vector that flips y and
 * has the origin on the top-left, like in OpenCV.
 */
inline Eigen::Vector4f get_opencv_viewport(int width, int height)
{
    return Eigen::Vector4f(0, height, width, -height);
};

/**
 * @brief Creates a 3x4 affine camera matrix from given fitting parameters. The
 * matrix transforms points directly from model-space to screen-space.
 *
 * This function is mainly used since the linear shape fitting fitting::fit_shape_to_landmarks_linear
 * expects one of these 3x4 affine camera matrices, as well as render::extract_texture.
 */
inline Eigen::Matrix<float, 3, 4> get_3x4_affine_camera_matrix(RenderingParameters params, int width,
                                                               int height)
{
    // Note: We should perhaps throw instead?
    assert(params.get_camera_type() == CameraType::Orthographic);

    using MatrixXf3x4 = Eigen::Matrix<float, 3, 4>;
    using Eigen::Matrix4f;
    const Matrix4f mvp = params.get_projection() * params.get_modelview();

    const Eigen::Vector4f viewport =
        get_opencv_viewport(width, height); // flips y, origin top-left, like in OpenCV
    // Equivalent to what glm::project's viewport does, but we don't change z and w:
    Eigen::Matrix4f viewport_mat;
    // clang-format off
    viewport_mat << viewport(2) / 2.0f, 0.0f,               0.0f, viewport(2) / 2.0f + viewport(0),
                    0.0f,               viewport(3) / 2.0f, 0.0f, viewport(3) / 2.0f + viewport(1),
                    0.0f,               0.0f,               1.0f, 0.0f,
                    0.0f,               0.0f,               0.0f, 1.0f;
    // clang-format on

    const Matrix4f full_projection_4x4 = viewport_mat * mvp;
    MatrixXf3x4 full_projection_3x4 = full_projection_4x4.block<3, 4>(0, 0); // we take the first 3 rows, but then set the last one to [0 0 0 1]
    // Use .block, possibly with the static template arguments!
    full_projection_3x4(2, 0) = 0.0f;
    full_projection_3x4(2, 1) = 0.0f;
    full_projection_3x4(2, 2) = 0.0f;
    full_projection_3x4(2, 3) = 1.0f;

    return full_projection_3x4;
};

} /* namespace fitting */
} /* namespace eos */

#endif /* EOS_RENDERING_PARAMETERS_HPP */
