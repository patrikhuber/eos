/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/ceres_nonlinear.hpp
 *
 * Copyright 2016 Patrik Huber
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

#ifndef CERESNONLINEAR_HPP_
#define CERESNONLINEAR_HPP_
#endif

#include "eos/core/Image.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"

#include "Eigen/Core"

#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/problem.h"

#include <array>
#include <vector>

#define get_num_cam_params(use_perspective)                                                                  \
    static_cast<std::size_t>(3) + static_cast<std::size_t>(use_perspective)

template <std::size_t N>
using darray = std::array<double, N>;

namespace eos {
namespace fitting {

// Forward declarations:
template <typename T>
std::array<T, 3> get_shape_point(const morphablemodel::PcaModel& shape_model,
                                 const morphablemodel::Blendshapes& blendshapes, int vertex_id,
                                 const T* const shape_coeffs, const T* const blendshape_coeffs,
                                 std::size_t num_coeffs_fitting);

template <typename T>
std::array<T, 3> get_vertex_colour(const morphablemodel::PcaModel& colour_model, int vertex_id,
                                   const T* const colour_coeffs, std::size_t num_coeffs_fitting);

/**
 * Cost function for a prior on the parameters.
 *
 * Prior towards zero (0, 0...) for the parameters.
 * Note: The weight is inside the norm, so may not correspond to the "usual"
 * formulas. However I think it's equivalent up to a scaling factor, but it
 * should be checked.
 */
struct PriorCost
{

    /**
     * Creates a new prior object with set number of variables and a weight.
     *
     * @param[in] num_variables Number of variables that the parameter vector contains.
     * @param[in] weight A weight that the parameters are multiplied with.
     */
    explicit PriorCost(std::size_t num_variables, double weight = 1.0)
        : num_variables(num_variables), weight(weight){};

    /**
     * Cost function implementation.
     *
     * @param[in] x An array of parameters.
     * @param[in] residual An array of the resulting residuals.
     * @return Returns true. The ceres documentation is not clear about that I think.
     */
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        for (int i = 0; i < num_variables; ++i)
        {
            residual[i] = weight * x[i];
        }
        return true;
    };

private:
    std::size_t num_variables;
    double weight;
};

/**
 * 2D landmark error cost function.
 *
 * Computes the landmark reprojection error in 2D.
 * Models the cost for one landmark. The residual is 2-dim, [x, y].
 * Its input params are camera parameters, shape coefficients and
 * blendshape coefficients.
 */
struct LandmarkCost
{

    /**
     * Constructs a new landmark cost function object with for a particular landmark/vertex id.
     *
     * Warning: Don't put in temporaries for \c shape_model and \c blendshapes! We don't make a copy, we store
     * a reference to what is given to the function!
     *
     * @param[in] shape_model A PCA 3D shape model. Do not use a temporary.
     * @param[in] blendshapes A set of 3D blendshapes. Do not use a temporary.
     * @param[in] observed_landmark An observed 2D landmark in an image.
     * @param[in] vertex_id The vertex id that the given observed landmark corresponds to.
     * @param[in] image_width Width of the image that the 2D landmark is from (needed for the model
     * projection).
     * @param[in] image_height Height of the image.
     * @param[in] use_perspective Whether a perspective or an orthographic projection should be used.
     */
    LandmarkCost(const morphablemodel::PcaModel& shape_model, const morphablemodel::Blendshapes& blendshapes,
                 Eigen::Vector2f observed_landmark, int vertex_id, int image_width, int image_height,
                 bool use_perspective)
        : shape_model(shape_model), blendshapes(blendshapes), observed_landmark(std::move(observed_landmark)),
          vertex_id(vertex_id), image_width(image_width), image_height(image_height),
          aspect_ratio(static_cast<double>(image_width) / image_height), use_perspective(use_perspective){};

    /**
     * Landmark cost function implementation.
     *
     * Measures the landmark reprojection error of the model with the estimated parameters and the observed 2D
     * landmarks. For one single landmark.
     *
     * @param[in] camera_rotation A set of camera parameters, parameterised as a quaternion [w x y z].
     * @param[in] camera_translation_and_intrinsics Camera translation and intrinsic parameters. Ortho: [t_x
     * t_y frustum_scale]. Perspective: [t_x t_y t_z fov].
     * @param[in] shape_coeffs A set of PCA shape coefficients.
     * @param[in] blendshape_coeffs A set of blendshape coefficients.
     * @param[in] residual An array of the resulting residuals.
     * @return Returns true. The ceres documentation is not clear about that I think.
     */
    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation_and_intrinsics,
                    const T* const shape_coeffs, const T* const blendshape_coeffs, T* residual) const
    {
        using namespace glm;
        // Generate shape instance (of only one vertex id!) using current parameters and 10 shape
        // coefficients: Note: Why are we not returning a glm::tvec3<T>?
        const auto point_arr = get_shape_point<T>(shape_model, blendshapes, vertex_id, shape_coeffs,
                                                  blendshape_coeffs, get_num_cam_params(use_perspective));

        // Project the point to 2D:
        const tvec3<T> point_3d(point_arr[0], point_arr[1], point_arr[2]);
        // I think the quaternion is always normalised because we run Ceres with QuaternionParameterization
        const tquat<T> rot_quat(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
        // We rotate ZXY*p, which is RPY*p. I'm not sure this matrix still corresponds to RPY - probably if we
        // use glm::eulerAngles(), these are not RPY anymore and we'd have to adjust if we were to use
        // rotation matrices.
        const auto rot_mtx = glm::mat4_cast(rot_quat);

        // Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
        const tvec4<T> viewport(0, image_height, image_width, -image_height); // OpenCV convention

        tvec3<T> projected_point; // Note: could avoid default construction by using a lambda and
                                           // immediate invocation
        if (use_perspective)
        {
            const auto t_mtx = glm::translate(tvec3<T>(camera_translation_and_intrinsics[0],
                                                       camera_translation_and_intrinsics[1],
                                                       camera_translation_and_intrinsics[2]));
            const auto& fov = camera_translation_and_intrinsics[3];
            const auto persp_mtx = glm::perspective(fov, static_cast<T>(aspect_ratio), static_cast<T>(0.1),
                                                    static_cast<T>(1000.0));
            projected_point = glm::project(point_3d, t_mtx * rot_mtx, persp_mtx, viewport);
        } else
        {
            const auto& frustum_scale = camera_translation_and_intrinsics[2];
            const auto t_mtx = glm::translate(
                tvec3<T>(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1],
                         0.0)); // we don't have t_z in ortho camera, it doesn't matter where it is
            const auto ortho_mtx =
                glm::ortho(-1.0 * aspect_ratio * frustum_scale, 1.0 * aspect_ratio * frustum_scale,
                           -1.0 * frustum_scale, 1.0 * frustum_scale);
            projected_point = glm::project(point_3d, t_mtx * rot_mtx, ortho_mtx, viewport);
        }
        // Residual: Projected point minus the observed 2D landmark point
        residual[0] = projected_point.x - static_cast<T>(observed_landmark[0]);
        residual[1] = projected_point.y - static_cast<T>(observed_landmark[1]);
        return true;
    };

private:
    const morphablemodel::PcaModel&
        shape_model; // Or store as pointer (non-owning) or std::reference_wrapper?
    const morphablemodel::Blendshapes& blendshapes;
    const Eigen::Vector2f observed_landmark;
    const int vertex_id;
    const int image_width;
    const int image_height;
    const double aspect_ratio;
    const bool use_perspective;
};

/**
 * Image error cost function (at vertex locations).
 *
 * Measures the RGB image error between a particular vertex point of the 3D
 * model at its projected location and the observed input image.
 * Models the cost for 1 vertex. The residual is 3-dim, [r, g, b].
 * Its input params are cam, shape-coeffs, BS-coeffs and colour coeffs.
 * This projects the vertex locations - so not a full rendering pass.
 */
struct ImageCost
{
    /**
     * Constructs a new cost function object for a particular vertex id that measures the RGB image error
     * between the estimated model point and the observed input image.
     *
     * Warning: Don't put in temporaries for \c morphable_model and \c blendshapes! We don't make a copy, we
     * store a reference to what is given to the function!
     *
     * @param[in] morphable_model A 3D Morphable Model. Do not use a temporary.
     * @param[in] blendshapes A set of 3D blendshapes. Do not use a temporary.
     * @param[in] image The observed image. TODO: We should assert that the image we get is 8UC3!
     * @param[in] vertex_id Vertex id of the 3D model that should be projected and measured.
     * @param[in] use_perspective Whether a perspective or an orthographic projection should be used.
     * @throws std::runtime_error if the given \c image is not of type CV_8UC3.
     */
    ImageCost(const morphablemodel::MorphableModel& morphable_model,
              const morphablemodel::Blendshapes& blendshapes, const core::Image3u& image, int vertex_id,
              bool use_perspective)
        : morphable_model(morphable_model), blendshapes(blendshapes), image(image),
          aspect_ratio(static_cast<double>(image.width()) / image.height()), vertex_id(vertex_id),
          use_perspective(use_perspective)
    {
        if (!morphable_model.has_color_model())
        {
            throw std::runtime_error("The MorphableModel used does not contain a colour (albedo) model. "
                                     "ImageCost requires a model that contains a colour PCA model. You may "
                                     "want to use the full Surrey Face Model.");
        }
    };

    /**
     * Image cost function implementation.
     *
     * Measures the image pixel error between the given model vertex projected to 2D and the observed input
     * image.
     *
     * Todo: We should deal with visibility! Don't evaluate when the vertex is self-occluded.
     *
     * @param[in] camera_rotation A set of camera parameters, parameterised as a quaternion [w x y z].
     * @param[in] camera_translation_and_intrinsics Camera translation and intrinsic parameters. Ortho: [t_x
     * t_y frustum_scale]. Perspective: [t_x t_y t_z fov].
     * @param[in] shape_coeffs A set of PCA shape coefficients.
     * @param[in] blendshape_coeffs A set of blendshape coefficients.
     * @param[in] color_coeffs A set of PCA colour (albedo) coefficients.
     * @param[in] residual An array of the resulting residuals.
     * @return Returns true. The ceres documentation is not clear about that I think.
     */
    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation_and_intrinsics,
                    const T* const shape_coeffs, const T* const blendshape_coeffs,
                    const T* const color_coeffs, T* residual) const
    {
        using namespace glm;
        // Note: The following is all duplicated code with LandmarkCost. Fix if possible performance-wise.
        // Generate 3D shape point using the current parameters:
        const auto point_arr =
            get_shape_point<T>(morphable_model.get_shape_model(), blendshapes, vertex_id, shape_coeffs,
                               blendshape_coeffs, get_num_cam_params(use_perspective));

        // Project the point to 2D:
        const tvec3<T> point_3d(point_arr[0], point_arr[1], point_arr[2]);
        // I think the quaternion is always normalised because we run Ceres with QuaternionParameterization
        const auto rot_quat =
            tquat<T>(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
        // We rotate ZXY*p, which is RPY*p. I'm not sure this matrix still corresponds to RPY - probably if we
        // use glm::eulerAngles(), these are not RPY anymore and we'd have to adjust if we were to use
        // rotation matrices.
        const auto rot_mtx = glm::mat4_cast(rot_quat);

        // Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
        const auto viewport = tvec4<T>(0, image.height(), image.width(), -image.height()); // OpenCV convention

        tvec3<T> projected_point;
        if (use_perspective)
        {
            const auto t_mtx = glm::translate(tvec3<T>(camera_translation_and_intrinsics[0],
                                                       camera_translation_and_intrinsics[1],
                                                       camera_translation_and_intrinsics[2]));
            const auto& focal = camera_translation_and_intrinsics[3];
            const auto persp_mtx = glm::perspective(focal, static_cast<T>(aspect_ratio), static_cast<T>(0.1),
                                                    static_cast<T>(1000.0));
            projected_point = glm::project(point_3d, t_mtx * rot_mtx, persp_mtx, viewport);
        } else
        {
            const auto& frustum_scale = camera_translation_and_intrinsics[2];
            const auto t_mtx = glm::translate(
                tvec3<T>(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1],
                         0.0)); // we don't have t_z in ortho camera, it doesn't matter where it is
            const auto ortho_mtx =
                glm::ortho(-1.0 * aspect_ratio * frustum_scale, 1.0 * aspect_ratio * frustum_scale,
                           -1.0 * frustum_scale, 1.0 * frustum_scale);
            projected_point = glm::project(point_3d, t_mtx * rot_mtx, ortho_mtx, viewport);
        }

        // Access the image colour value at the projected pixel location, if inside the image - otherwise set
        // to (127, 127, 127) (maybe not ideal...):
        if (projected_point.y < static_cast<T>(0) || projected_point.y >= static_cast<T>(image.height()) ||
            projected_point.x < static_cast<T>(0) || projected_point.x >= static_cast<T>(image.width()))
        {
            // The point is outside the image.
            residual[0] = static_cast<T>(127.0);
            residual[1] = static_cast<T>(127.0);
            residual[2] = static_cast<T>(127.0);

            // TODO: What does interpolator.Evaluate() return in this case?
            /*	Grid2D<uchar, 3> grid(image.ptr(0), 0, image.rows, 0, image.cols);
            BiCubicInterpolator<Grid2D<uchar, 3>> interpolator(grid);
            T observed_colour[3];
            interpolator.Evaluate(projected_y, projected_x, &observed_colour[0]); // says it returns false
            when (r, c) is out of bounds... but it returns void?
            //std::cout << observed_colour[0] << ", " << observed_colour[1] << ", " << observed_colour[2] <<
            "\n";
            */
            // It kind of looks like as if when it's out of bounds, there will be a vector out of bound access
            // and an assert/crash? No, in debugging, it looks like it just interpolates or something. Not
            // clear currently.
        } else
        {
            // Note: We could store the BiCubicInterpolator as member variable.
            // The default template arguments for Grid2D are <T, kDataDim=1, kRowMajor=true,
            // kInterleaved=true> and (except for the dimension), they're the right ones for us.

            using Grid2D3d = ceres::Grid2D<std::uint8_t, 3>;

            Grid2D3d grid(image.ptr<std::uint8_t>(0, 0), 0, image.height(), 0, image.width());
            auto interpolator = ceres::BiCubicInterpolator<Grid2D3d>(grid);
            auto observed_colour = std::array<T, 3>();
            interpolator.Evaluate(projected_point.y, projected_point.x, &observed_colour[0]);

            // This probably needs to be modified if we add a light model.
            auto model_colour = get_vertex_colour(morphable_model.get_color_model(), vertex_id, color_coeffs,
                                                  get_num_cam_params(use_perspective));
            // I think this returns RGB, and between [0, 1].

            // Residual: Vertex colour of model point minus the observed colour in the 2D image
            // observed_colour is BGR, model_colour is RGB. Residual will be RGB.
            residual[0] = model_colour[0] * static_cast<T>(255.0) - static_cast<T>(observed_colour[2]);
            residual[1] = model_colour[1] * static_cast<T>(255.0) - static_cast<T>(observed_colour[1]);
            residual[2] = model_colour[2] * static_cast<T>(255.0) - static_cast<T>(observed_colour[0]);
        }
        return true;
    };

private:
    const morphablemodel::MorphableModel&
        morphable_model; // Or store as pointer (non-owning) or std::reference_wrapper?
    const morphablemodel::Blendshapes& blendshapes;
    const core::Image3u image; // the observed image
    const double aspect_ratio;
    const int vertex_id;
    const bool use_perspective;
};

/**
 * Returns the 3D position of a single point of the 3D shape generated by the parameters given.
 *
 * @param[in] shape_model A PCA 3D shape model.
 * @param[in] blendshapes A set of 3D blendshapes.
 * @param[in] vertex_id Vertex id of the 3D model that should be projected.
 * @param[in] shape_coeffs A set of PCA shape coefficients used to generate the point.
 * @param[in] blendshape_coeffs A set of blendshape coefficients used to generate the point.
 * @return The 3D point.
 */
template <typename T>
std::array<T, 3> get_shape_point(const morphablemodel::PcaModel& shape_model,
                                 const morphablemodel::Blendshapes& blendshapes, int vertex_id,
                                 const T* const shape_coeffs, const T* const blendshape_coeffs,
                                 std::size_t num_coeffs_fitting)
{
    auto mean = shape_model.get_mean_at_point(vertex_id);
    auto basis = shape_model.get_rescaled_pca_basis_at_point(vertex_id);
    // Computing Shape = mean + basis * coeffs:
    // Note: Could use an Eigen matrix with type T to see if it gives a speedup.
    std::array<T, 3> point{static_cast<T>(mean[0]), static_cast<T>(mean[1]), static_cast<T>(mean[2])};
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[0] += static_cast<T>(basis.row(0).col(i)(0)) * shape_coeffs[i];
    }
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[1] += static_cast<T>(basis.row(1).col(i)(0)) * shape_coeffs[i];
    }
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[2] += static_cast<T>(basis.row(2).col(i)(0)) * shape_coeffs[i];
    }
    // Adding the blendshape offsets:
    // Shape = mean + basis * coeffs + blendshapes * bs_coeffs:
    auto num_blendshapes = blendshapes.size();
    for (int i = 0; i < num_blendshapes; ++i)
    {
        point[0] += static_cast<T>(blendshapes[i].deformation(3 * vertex_id + 0)) * blendshape_coeffs[i];
    }
    for (int i = 0; i < num_blendshapes; ++i)
    {
        point[1] += static_cast<T>(blendshapes[i].deformation(3 * vertex_id + 1)) * blendshape_coeffs[i];
    }
    for (int i = 0; i < num_blendshapes; ++i)
    {
        point[2] += static_cast<T>(blendshapes[i].deformation(3 * vertex_id + 2)) * blendshape_coeffs[i];
    }
    return point;
};

/**
 * Returns the colour value of a single point of the 3D model generated by the parameters given.
 *
 * @param[in] color_model A PCA 3D colour (albedo) model.
 * @param[in] vertex_id Vertex id of the 3D model whose colour is to be returned.
 * @param[in] color_coeffs A set of PCA colour coefficients.
 * @return The colour. As RGB? In [0, 1]?
 */
template <typename T>
std::array<T, 3> get_vertex_colour(const morphablemodel::PcaModel& color_model, int vertex_id,
                                   const T* const color_coeffs, std::size_t num_coeffs_fitting)
{
    auto mean = color_model.get_mean_at_point(vertex_id);
    auto basis = color_model.get_rescaled_pca_basis_at_point(vertex_id);
    // Computing Colour = mean + basis * coeffs
    // Note: Could use an Eigen matrix with type T to see if it gives a speedup.
    std::array<T, 3> point{static_cast<T>(mean[0]), static_cast<T>(mean[1]), static_cast<T>(mean[2])};
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[0] += static_cast<T>(basis.row(0).col(i)(0)) * color_coeffs[i];
    }
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[1] += static_cast<T>(basis.row(1).col(i)(0)) * color_coeffs[i];
    }
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[2] += static_cast<T>(basis.row(2).col(i)(0)) * color_coeffs[i];
    }
    return point;
};

/*
 * Parameters of camera
 *
 * They are not stored in ModelFitter to support many cameras optimization.
 */
template <bool use_perspective>
struct CameraParameters
{
    CameraParameters(int image_cols, int image_rows)
        : translation_and_intrinsics(get_translation_and_intrinsics()), image_cols(image_cols),
          image_rows(image_rows)
    {
    }

    /*
     * Return viewport for given image rows num and cols num.
     */
    glm::dvec4 get_viewport() const
    {
        return glm::dvec4(0, image_rows, image_cols, -image_rows);
    }

    /**
     * Cast rotation_quaternion to euler angle
     */
    glm::dvec3 get_euler_rotation() const
    {
        return glm::eulerAngles(get_glm_rotation_quaternion());
    }

    /**
     * Calculate translation matrix from estimated camera parameters
     */
    glm::dmat4x4 calculate_translation_matrix() const
    {
        return glm::translate(glm::dvec3(translation_and_intrinsics[0], translation_and_intrinsics[1],
                                         use_perspective ? translation_and_intrinsics[2] : 0.0));
    }

    /**
     * Calculate rotation matrix from estimated camera parameters
     */
    glm::dmat4x4 calculate_rotation_matrix() const
    {
        return glm::mat4_cast(get_glm_rotation_quaternion());
    }

    /**
     * Calculate projection matrix from estimated camera parameters
     */
    glm::dmat4x4 calculate_projection_matrix() const
    {
        auto aspect = static_cast<double>(image_cols) / image_rows;
        if (use_perspective)
        {
            const auto& focal = translation_and_intrinsics[3];
            return glm::perspective(focal, aspect, 0.1, 1000.0);
        } else
        {
            const auto& frustum_scale = translation_and_intrinsics[2];
            return glm::ortho(-1.0 * aspect * frustum_scale, 1.0 * aspect * frustum_scale,
                              -1.0 * frustum_scale, 1.0 * frustum_scale);
        }
    }

    darray<4> rotation_quaternion = {1, 0, 0, 0}; // Quaternion, [w x y z].
    darray<get_num_cam_params(use_perspective)> translation_and_intrinsics;
    int image_cols, image_rows;

private:
    /*
     * Make array with initial camera parameters
     *
     * * @return std::array with 3 or 4 (for perspective projection) parameters.
     */
    auto get_translation_and_intrinsics() const
    {
        // Parameters for the orthographic projection: [t_x, t_y, frustum_scale]
        // And perspective projection: [t_x, t_y, t_z, fov].
        // Origin is assumed at center of image, and no lens distortions.
        // Note: Actually, we estimate the model-view matrix and not the camera position. But one defines the
        // other.

        darray<get_num_cam_params(use_perspective)> translation_and_intrinsics;
        if (use_perspective)
        {
            translation_and_intrinsics[2] = -400.0;              // Move the model back (along the -z axis)
            translation_and_intrinsics[3] = glm::radians(60.0f); // fov
        } else
        {
            translation_and_intrinsics[2] = 110.0; // frustum_scale
        }
        return translation_and_intrinsics;
    }

    /**
     * Cast rotation_quaternion to glm::quat
     */
    glm::dquat get_glm_rotation_quaternion() const
    {
        return glm::dquat(rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2],
                          rotation_quaternion[3]);
    }
};

struct PerspectiveCameraParameters : CameraParameters<true>
{
    PerspectiveCameraParameters(int image_cols, int image_rows)
        : CameraParameters<true>(image_cols, image_rows)
    {
    }
};

struct OrtogonalCameraParameters : CameraParameters<false>
{
    OrtogonalCameraParameters(int image_cols, int image_rows)
        : CameraParameters<false>(image_cols, image_rows)
    {
    }
};

/**
 * Class that maintain all model fitting process
 *
 * @tparam shapes_num number of shapes of model
 * @tparam blendshapes_num number of blendshapes of model
 * @tparam color_coeffs_num number of color coefficients of model
 * @tparam use_perspective will fitting use perspective projection
 */
template <std::size_t shapes_num, std::size_t blendshapes_num, std::size_t color_coeffs_num>
class ModelFitter
{
public:
    /**
     * @param[in] morphable_model morphable model instance
     * @param[in] blendshapes all blendshapes to fit
     */
    explicit ModelFitter(const morphablemodel::MorphableModel* const morphable_model,
                         const morphablemodel::Blendshapes* const blendshapes = nullptr)
        : problem(std::make_unique<ceres::Problem>()), morphable_model(morphable_model)
    {
        if (blendshapes == nullptr)
        {
            if (!morphable_model->has_separate_expression_model())
            {
                throw std::runtime_error(
                    "Blendshapes was not passed and morphable model does not contain them too."
                    "Try to pass blendshapes explicitly.");
            } else
            {
                this->blendshapes =
                    &cpp17::get<morphablemodel::Blendshapes>(morphable_model->get_expression_model().value());
            }
        } else
        {
            this->blendshapes = blendshapes;
        }
    }

    /*
     * Apply solver to overall problem
     *
     * @param[in] solver_options ceres solver options
     */
    ceres::Solver::Summary solve(const ceres::Solver::Options& solver_options)
    {
        // Fit position
        ceres::Solver::Summary solver_summary;
        ceres::Solve(solver_options, problem.get(), &solver_summary);

        return solver_summary;
    }

    /**
     * Clean up all added blocks and cost functions.
     */
    void reset_problem()
    {
        problem = std::make_unique<ceres::Problem>();
    }

    /**
     * Estimate contours using current fitting state and user lists of potential contours
     *
     * @param camera
     * @param landmarks_contour contours in landmarks
     * @param model_contour contours in model
     * @param landmarks to contour search
     * @return vector with indexed Landmarks
     */
    template <typename LandmarkType, bool use_perspective>
    auto estimate_contours(const CameraParameters<use_perspective>& camera,
                           const ContourLandmarks& landmarks_contour, const ModelContour& model_contour,
                           const core::LandmarkCollection<LandmarkType>& landmarks) const
    {
        std::vector<Eigen::Vector2f> image_points_contour; // the 2D landmark points
        std::vector<int> vertex_indices_contour;           // their corresponding 3D vertex indices

        // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
        std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
            eos::fitting::get_contour_correspondences(
                landmarks, landmarks_contour, model_contour,
                static_cast<float>(glm::degrees(camera.get_euler_rotation()[1])), morphable_model->get_mean(),
                camera.calculate_translation_matrix() * camera.calculate_rotation_matrix(),
                camera.calculate_projection_matrix(), camera.get_viewport());

        auto contour_landmarks = core::IndexedLandmarkCollection<LandmarkType>();
        for (int i = 0; i < image_points_contour.size(); ++i)
        {
            contour_landmarks.emplace_back("", image_points_contour[i], vertex_indices_contour[i]);
        }

        return contour_landmarks;
    }

    /**
     * Apply estimated shapes and blendshapes to morhable model and returns points.
     */
    Eigen::VectorXf calculate_estimated_points_positions() const
    {
        auto blendshape_coefficients_float =
            std::vector<float>(std::begin(blendshape_coefficients), std::end(blendshape_coefficients));
        return morphable_model->get_shape_model().draw_sample(shape_coefficients) +
               to_matrix(*blendshapes) *
                   Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients_float.data(),
                                                     blendshape_coefficients_float.size());
    }

    /**
     * Add residual block with camera cost function to the overall problem
     *
     * @param[in] camera camera to fit
     * @param[in] landmarks all landmarks with indices corresponding to model
     */
    template <typename LandmarkType, bool use_perspective>
    void add_camera_cost_function(CameraParameters<use_perspective>& camera,
                                  const core::IndexedLandmarkCollection<LandmarkType>& landmarks)
    {
        auto& camera_rotation = camera.rotation_quaternion;
        auto& camera_translation_and_intrinsics = camera.translation_and_intrinsics;
        for (const auto& landmark : landmarks)
        {
            /* Templates: CostFunctor, num residuals, camera rotation (quaternion),
                          camera translation & fov/frustum_scale, shape-coeffs, bs-coeffs */
            auto* cost_function =
                new ceres::AutoDiffCostFunction<fitting::LandmarkCost, 2, 4,
                                                get_num_cam_params(use_perspective), shapes_num,
                                                blendshapes_num>(new LandmarkCost(
                    morphable_model->get_shape_model(), *blendshapes, landmark.coordinates,
                    landmark.model_index, camera.image_cols, camera.image_rows, use_perspective));

            problem->AddResidualBlock(cost_function, nullptr, &camera_rotation[0],
                                      &camera_translation_and_intrinsics[0], &shape_coefficients[0],
                                      &blendshape_coefficients[0]);
        }
        if (use_perspective)
        {
            problem->SetParameterUpperBound(&camera_translation_and_intrinsics[0], 2,
                                            -std::numeric_limits<float>::epsilon()); // t_z has to be negative
            problem->SetParameterLowerBound(
                &camera_translation_and_intrinsics[0], 3,
                std::numeric_limits<float>::epsilon()); // fov in radians, must be > 0
        } else
        {
            problem->SetParameterLowerBound(&camera_translation_and_intrinsics[0], 2,
                                            1.0); // frustum_scale must be > 0
        }
        problem->SetParameterization(&camera_rotation[0], new ceres::QuaternionParameterization);
    }

    /**
     * Add residual block with shape prior cost function to the overall problem
     *
     * @param[in] weight constant to multiply all shapes
     * @param[in, out] upper and lower bound of each shape_coefficient
     */
    void add_shape_prior_cost_function(double weigth = 35.0, double shape_coeff_limit = 3.0)
    {
        /* Templates: CostFunctor, num residuals, shape-coeffs */
        auto* shape_prior_cost = new ceres::AutoDiffCostFunction<fitting::PriorCost, shapes_num, shapes_num>(
            new fitting::PriorCost(shapes_num, weigth));
        problem->AddResidualBlock(shape_prior_cost, nullptr, &shape_coefficients[0]);
        for (int i = 0; i < shapes_num; ++i)
        {
            problem->SetParameterLowerBound(&shape_coefficients[0], i, -shape_coeff_limit);
            problem->SetParameterUpperBound(&shape_coefficients[0], i, shape_coeff_limit);
        }
    }

    /**
     * Add residual block with shape prior cost function to the passed problem
     *
     * @param[in] weight constant to multiply all blendshapes
     */
    void add_blendshape_prior_cost_function(double weigth = 10.0)
    {
        /* Templates: CostFunctor, num residuals, blendshape-coeffs */
        auto* blendshapes_prior_cost =
            new ceres::AutoDiffCostFunction<fitting::PriorCost, blendshapes_num, blendshapes_num>(
                new fitting::PriorCost(blendshapes_num, weigth));
        problem->AddResidualBlock(blendshapes_prior_cost, nullptr, &blendshape_coefficients[0]);

        for (int i = 0; i < blendshapes_num; ++i)
        {
            problem->SetParameterLowerBound(&blendshape_coefficients[0], i, 0.0);
        }
    }

    /**
     * Add residual block with image cost function to the passed problem
     *
     * @param[in, out] camera camera to fit
     * @param[in] image image to fit
     */
    template <bool use_perspective>
    void add_image_cost_function(CameraParameters<use_perspective>& camera, const core::Image3u& image)
    {
        // Add a residual for each vertex:
        for (int i = 0; i < morphable_model->get_shape_model().get_data_dimension() / 3; ++i)
        {
            // Templates: CostFunctor, Residuals: [R, G, B], camera rotation (quaternion),
            //            camera translation & focal length, shape-coeffs, bs-coeffs, colour coeffs

            auto* cost_function =
                new ceres::AutoDiffCostFunction<fitting::ImageCost, 3, 4, get_num_cam_params(use_perspective),
                                                shapes_num, blendshapes_num, color_coeffs_num>(
                    new fitting::ImageCost(*morphable_model, *blendshapes, image, i, use_perspective));

            problem->AddResidualBlock(cost_function, nullptr, &camera.rotation_quaternion[0],
                                      &camera.translation_and_intrinsics[0], &shape_coefficients[0],
                                      &blendshape_coefficients[0], &colour_coefficients[0]);
        }
    }

    /**
     * Add residual block with shape prior cost function to the overall problem
     *
     * @param[in] weight constant to multiply all colours
     * @param[in, out] upper and lower bound of each colour_coefficient
     */
    void add_image_prior_cost_function(double weigth = 35.0, double color_coeff_limit = 3.0)
    {
        // Templates: CostFunctor, num residuals, colour-coeffs
        auto* colour_prior_cost =
            new ceres::AutoDiffCostFunction<fitting::PriorCost, color_coeffs_num, color_coeffs_num>(
                new fitting::PriorCost(color_coeffs_num, weigth));

        problem->AddResidualBlock(colour_prior_cost, nullptr, &colour_coefficients[0]);
        for (int i = 0; i < color_coeffs_num; ++i)
        {
            problem->SetParameterLowerBound(&colour_coefficients[0], i, -color_coeff_limit);
            problem->SetParameterUpperBound(&colour_coefficients[0], i, color_coeff_limit);
        }
    }

    /**
     * Block shapes fitting
     */
    void set_shape_coefficients_constant()
    {
        problem->SetParameterBlockConstant(&shape_coefficients[0]);
    }

    /**
     * Block blendshapes fitting
     */
    void set_blendshape_coefficients_constant()
    {
        problem->SetParameterBlockConstant(&blendshape_coefficients[0]);
    }

    /**
     * Block fov fitting
     *
     * Note: it is only for perspective projection. It have no sense in case of ortogonal projection.
     *
     * @param[in,out] camera perspective camera to fit
     * @param[in] fov_in_grad field of view to fix
     */
    void set_fov_constant(PerspectiveCameraParameters &camera, double fov_in_grad)
    {
        auto& camera_translation_and_intrinsics = camera.translation_and_intrinsics;
        camera_translation_and_intrinsics[3] = glm::radians(fov_in_grad);

        std::vector<int> vec_constant_extrinsic = {3};
        auto subset_parameterization = new ceres::SubsetParameterization(
            static_cast<int>(get_num_cam_params(true)), vec_constant_extrinsic);
        problem->SetParameterization(&camera_translation_and_intrinsics[0], subset_parameterization);
    }

    darray<shapes_num> shape_coefficients;
    darray<blendshapes_num> blendshape_coefficients;
    darray<color_coeffs_num> colour_coefficients;

    std::unique_ptr<ceres::Problem> problem;

private:
    const morphablemodel::MorphableModel* morphable_model;
    const morphablemodel::Blendshapes* blendshapes;
};

} /* namespace fitting */
} /* namespace eos */
