/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/ceres_nonlinear.hpp
 *
 * Copyright 2016-2023 Patrik Huber
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

#ifndef EOS_CERES_NONLINEAR_HPP
#define EOS_CERES_NONLINEAR_HPP

#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/core/Image.hpp"

#include "Eigen/Core"

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

#include <vector>

namespace eos {
namespace fitting {

// Forward declarations:
template <typename T>
Eigen::Vector3<T> get_shape_at_point(const eos::morphablemodel::PcaModel& shape_model,
                                     const eos::morphablemodel::Blendshapes& blendshapes, int vertex_id,
                                     Eigen::Map<const Eigen::VectorX<T>> shape_coeffs,
                                     Eigen::Map<const Eigen::VectorX<T>> blendshape_coeffs);

template <typename T>
Eigen::Vector3<T> get_vertex_color_at_point(const eos::morphablemodel::PcaModel& color_model, int vertex_id,
                                            Eigen::Map<const Eigen::VectorX<T>> color_coeffs);

template <typename T>
Eigen::Matrix4<T> perspective(T fov_y, T aspect, T z_near, T z_far);

template <typename T>
Eigen::Vector3<T> project(const Eigen::Vector3<T>& obj, const Eigen::Matrix4<T>& model,
                          const Eigen::Matrix4<T>& proj, const Eigen::Vector4<T>& viewport);

/**
 * Cost function that consists of the parameter values themselves as residual.
 *
 * If used with a squared loss, this corresponds to an L2 norm constraint on the parameters.
 * This class is implemented exactly like Ceres' NormalPrior.
 */
struct NormCost
{

    /**
     * Creates a new NormCost object with set number of parameters.
     *
     * @param[in] num_parameters Number of parameters that the parameter vector contains.
     */
    NormCost(int num_parameters) : num_parameters(num_parameters){};

    /**
     * Cost function implementation.
     *
     * @param[in] x An array of parameters.
     * @param[in] residual An array of the resulting residuals.
     * @return whether the computation of the residuals was successful. Always returns true.
     */
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        for (int i = 0; i < num_parameters; ++i)
        {
            residual[i] = x[i];
        }
        return true;
    };

    /**
     * Factory to hide the construction of the CostFunction object from the client code.
     *
     * The number of parameters is given as a template argument, so that we can use Ceres' fixed-size
     * constructor.
     */
    template <int num_parameters>
    static ceres::CostFunction* Create()
    {
        return (new ceres::AutoDiffCostFunction<NormCost, num_parameters /* num residuals */,
                                                num_parameters /* num parameters */>(
            new NormCost(num_parameters)));
    }

private:
    int num_parameters;
};

/**
 * 2D landmark error cost function, using perspective projection (in OpenGL convention).
 *
 * Computes the landmark reprojection error in 2D. Models the cost for one landmark. The residual is 2-dim,
 * [x, y]. Its input params are camera parameters, shape coefficients and blendshape coefficients.
 */
struct PerspectiveProjectionLandmarkCost
{

    /**
     * Constructs a new landmark cost function object with for a particular landmark/vertex id.
     *
     * Warning: Don't put in temporaries for \c shape_model and \c blendshapes! We don't make a copy, we store
     * a reference to what is given to the function!
     *
     * @param[in] shape_model A PCA 3D shape model. Do not use a temporary.
     * @param[in] blendshapes A set of 3D blendshapes. Do not use a temporary.
     * @param[in] num_shape_coeffs Number of shape coefficients that are being optimised for.
     * @param[in] num_blendshape_coeffs Number of blendshape coefficients that are being optimised for.
     * @param[in] observed_landmark An observed 2D landmark in an image.
     * @param[in] vertex_id The vertex id that the given observed landmark corresponds to.
     * @param[in] image_width Width of the image that the 2D landmark is from (needed for the model
     * projection).
     * @param[in] image_height Height of the image.
     */
    PerspectiveProjectionLandmarkCost(const morphablemodel::PcaModel& shape_model,
                                      const std::vector<morphablemodel::Blendshape>& blendshapes,
                                      int num_shape_coeffs, int num_blendshape_coeffs,
                                      Eigen::Vector2f observed_landmark, int vertex_id, int image_width,
                                      int image_height)
        : shape_model(shape_model), blendshapes(blendshapes), num_shape_coeffs(num_shape_coeffs),
          num_blendshape_coeffs(num_blendshape_coeffs), observed_landmark(observed_landmark),
          vertex_id(vertex_id), image_width(image_width), image_height(image_height),
          aspect_ratio(static_cast<double>(image_width) / image_height){};

    /**
     * Landmark cost function implementation.
     *
     * Measures the landmark reprojection error of the model with the estimated parameters and the observed 2D
     * landmarks. For one single landmark.
     *
     * @param[in] camera_rotation A set of camera rotation parameters, parameterised as an Eigen::Quaternion.
     * @param[in] camera_translation Camera translation parameters: [t_x t_y t_z].
     * @param[in] camera_intrinsics Camera intrinsics, containing the field of view (fov_y).
     * @param[in] shape_coeffs A set of PCA shape coefficients.
     * @param[in] blendshape_coeffs A set of blendshape coefficients.
     * @param[in] residual An array of the resulting residuals.
     * @return whether the computation of the residuals was successful. Always returns true.
     */
    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation,
                    const T* const camera_intrinsics, const T* const shape_coeffs,
                    const T* const blendshape_coeffs, T* residual) const
    {
        // Generate shape instance (of only one vertex id) using current coefficients:
        Eigen::Map<const Eigen::VectorX<T>> shape_coeffs_mapped(shape_coeffs, num_shape_coeffs);
        Eigen::Map<const Eigen::VectorX<T>> blendshape_coeffs_mapped(blendshape_coeffs,
                                                                     num_blendshape_coeffs);

        const Eigen::Vector3<T> point_3d = get_shape_at_point(shape_model, blendshapes, vertex_id,
                                                              shape_coeffs_mapped, blendshape_coeffs_mapped);
        // Project the point to 2D:
        // I think the quaternion is always normalised because we run Ceres with QuaternionParameterization
        // Previously, we used glm::tquat, which expects w, x, y, z, and called it with [0, 1, 2, 3].
        // Eigen stores it as x, y, z, w. So... are we sure this is right?
        Eigen::Map<const Eigen::Quaternion<T>> rotation(camera_rotation);
        Eigen::Map<const Eigen::Vector3<T>> translation(camera_translation);

        Eigen::Matrix4<T> model_view_mtx = Eigen::Matrix4<T>::Identity();
        model_view_mtx.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        model_view_mtx.col(3).head<3>() = translation;

        // Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
        const Eigen::Vector4<T> viewport(T(0), T(image_height), T(image_width),
                                         T(-image_height)); // OpenCV convention

        const T& fov = camera_intrinsics[0];
        const auto projection_mtx = perspective(fov, T(aspect_ratio), T(0.1), T(1000.0));

        Eigen::Vector3<T> projected_point = project(point_3d, model_view_mtx, projection_mtx, viewport);

        // Residual: Projected point minus the observed 2D landmark point
        residual[0] = projected_point.x() - T(observed_landmark[0]);
        residual[1] = projected_point.y() - T(observed_landmark[1]);
        return true;
    };

    /**
     * Factory to hide the construction of the CostFunction object from the client code.
     *
     * The number of parameters are given as template arguments, so that we can use Ceres' fixed-size
     * constructor.
     */
    template <int num_shape_coeffs, int num_blendshape_coeffs>
    static ceres::CostFunction* Create(const eos::morphablemodel::PcaModel& shape_model,
                                       const eos::morphablemodel::Blendshapes& blendshapes,
                                       Eigen::Vector2f observed_landmark, int vertex_id, int image_width,
                                       int image_height)
    {
        // Template arguments: 2 residuals, 4 (camera rotation), 3 (camera translation), 1 (camera
        // intrinsics), n (shape coeffs), m (blendshape coeffs)
        return (new ceres::AutoDiffCostFunction<PerspectiveProjectionLandmarkCost, 2, 4, 3, 1,
                                                num_shape_coeffs, num_blendshape_coeffs>(
            new PerspectiveProjectionLandmarkCost(shape_model, blendshapes, num_shape_coeffs,
                                                  num_blendshape_coeffs, observed_landmark, vertex_id,
                                                  image_width, image_height)));
    };

private:
    const morphablemodel::PcaModel&
        shape_model; // Or store as pointer (non-owning) or std::reference_wrapper?
    const std::vector<morphablemodel::Blendshape>& blendshapes;
    const int num_shape_coeffs;
    const int num_blendshape_coeffs;
    const Eigen::Vector2f observed_landmark;
    const int vertex_id;
    const int image_width;
    const int image_height;
    const double aspect_ratio;
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
struct VertexColorCost
{
    /**
     * Constructs a new cost function object for a particular vertex id that measures the RGB image error
     * between the estimated model point and the observed input image.
     *
     * Warning: Don't put in temporaries for \c shape_model, \c blendshapes and \c color_model. We don't make
     * a copy, we store a reference to what is given to the function.
     *
     * @param[in] shape_model A PCA 3D shape model. Do not use a temporary.
     * @param[in] blendshapes A set of 3D blendshapes. Do not use a temporary.
     * @param[in] shape_model A PCA 3D color model. Do not use a temporary.
     * @param[in] num_shape_coeffs Number of shape coefficients that are being optimised for.
     * @param[in] num_blendshape_coeffs Number of blendshape coefficients that are being optimised for.
     * @param[in] num_color_coeffs Number of colour coefficients that are being optimised for.
     * @param[in] vertex_id Vertex id of the 3D model that should be projected and measured.
     * @param[in] image The observed image.
     */
    VertexColorCost(const eos::morphablemodel::PcaModel& shape_model,
                    const eos::morphablemodel::Blendshapes& blendshapes,
                    const eos::morphablemodel::PcaModel& color_model, int num_shape_coeffs,
                    int num_blendshape_coeffs, int num_color_coeffs, int vertex_id,
                    const eos::core::Image3u& image)
        : shape_model(shape_model), blendshapes(blendshapes), color_model(color_model),
          num_shape_coeffs(num_shape_coeffs), num_blendshape_coeffs(num_blendshape_coeffs),
          num_color_coeffs(num_color_coeffs), vertex_id(vertex_id), image(image){};

    /**
     * Image cost function implementation.
     *
     * Measures the image pixel error between the given model vertex projected to 2D and the observed input
     * image.
     *
     * Todo: We should deal with visibility! Don't evaluate when the vertex is self-occluded.
     *
     * @param[in] camera_rotation A set of camera rotation parameters, parameterised as an Eigen::Quaternion.
     * @param[in] camera_translation Camera translation parameters: [t_x t_y t_z].
     * @param[in] camera_intrinsics Camera intrinsics, containing the field of view (fov_y).
     * @param[in] shape_coeffs A set of PCA shape coefficients.
     * @param[in] blendshape_coeffs A set of blendshape coefficients.
     * @param[in] color_coeffs A set of PCA colour (albedo) coefficients.
     * @param[in] residual An array of the resulting residuals.
     * @return whether the computation of the residuals was successful. Always returns true.
     */
    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation,
                    const T* const camera_intrinsics, const T* const shape_coeffs,
                    const T* const blendshape_coeffs, const T* const color_coeffs, T* residual) const
    {
        // The following code blocks are identical to PerspectiveProjectionLandmarkCost:

        // Generate shape instance (of only one vertex id) using current coefficients:
        Eigen::Map<const Eigen::VectorX<T>> shape_coeffs_mapped(shape_coeffs, num_shape_coeffs);
        Eigen::Map<const Eigen::VectorX<T>> blendshape_coeffs_mapped(blendshape_coeffs,
                                                                     num_blendshape_coeffs);

        const Eigen::Vector3<T> point_3d = get_shape_at_point(shape_model, blendshapes, vertex_id,
                                                              shape_coeffs_mapped, blendshape_coeffs_mapped);
        // Project the point to 2D:
        // I think the quaternion is always normalised because we run Ceres with QuaternionParameterization
        // Previously, we used glm::tquat, which expects w, x, y, z, and called it with [0, 1, 2, 3].
        // Eigen stores it as x, y, z, w. So... are we sure this is right?
        Eigen::Map<const Eigen::Quaternion<T>> rotation(camera_rotation);
        Eigen::Map<const Eigen::Vector3<T>> translation(camera_translation);

        Eigen::Matrix4<T> model_view_mtx = Eigen::Matrix4<T>::Identity();
        model_view_mtx.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        model_view_mtx.col(3).head<3>() = translation;

        // Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
        const Eigen::Vector4<T> viewport(T(0), T(image.height()), T(image.width()),
                                         T(-image.height())); // OpenCV convention

        const T& fov = camera_intrinsics[0];
        const double aspect_ratio = static_cast<double>(image.width()) / image.height();
        const auto projection_mtx = perspective(fov, T(aspect_ratio), T(0.1), T(1000.0));

        Eigen::Vector3<T> projected_point = project(point_3d, model_view_mtx, projection_mtx, viewport);
        // End of identical block with PerspectiveProjectionLandmarkCost.

        // Access the image colour value at the projected pixel location, if inside the image - otherwise
        // Ceres uses the value from the grid edge.
        // Note: We could store the BiCubicInterpolator as member variable.
        // The default template arguments for Grid2D are <T, kDataDim=1, kRowMajor=true,
        // kInterleaved=true> and (except for the dimension), they're the right ones for us.
        ceres::Grid2D<unsigned char, 3> grid(&image(0, 0).data()[0], 0, image.height(), 0, image.width());
        ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 3>> interpolator(grid);
        T observed_colour[3];
        interpolator.Evaluate(projected_point.y(), projected_point.x(), &observed_colour[0]);

        // Get the vertex's colour value:
        Eigen::Map<const Eigen::VectorX<T>> color_coeffs_mapped(color_coeffs, num_color_coeffs);
        const auto model_color = get_vertex_color_at_point(color_model, vertex_id, color_coeffs_mapped);
        // Returns RGB, between [0, 1].

        // Residual: Vertex colour of model point minus the observed colour in the 2D image
        // observed_colour is RGB, model_colour is RGB. Residual will be RGB.
        residual[0] = model_color[0] * 255.0 - T(observed_colour[0]); // r
        residual[1] = model_color[1] * 255.0 - T(observed_colour[1]); // g
        residual[2] = model_color[2] * 255.0 - T(observed_colour[2]); // b

        return true;
    };

    /**
     * Factory to hide the construction of the CostFunction object from the client code.
     *
     * The number of parameters are given as template arguments, so that we can use Ceres' fixed-size
     * constructor.
     */
    template <int num_shape_coeffs, int num_blendshape_coeffs, int num_color_coeffs>
    static ceres::CostFunction* Create(const eos::morphablemodel::PcaModel& shape_model,
                                       const eos::morphablemodel::Blendshapes& blendshapes,
                                       const eos::morphablemodel::PcaModel& color_model, int vertex_id,
                                       const eos::core::Image3u& image)
    {
        // Template arguments: 3 residuals (RGB), 4 (camera rotation), 3 (camera translation), 1 (camera
        // intrinsics), x (shape coeffs), x (expr coeffs), x (colour coeffs)
        return (new ceres::AutoDiffCostFunction<VertexColorCost, 3, 4, 3, 1, num_shape_coeffs,
                                                num_blendshape_coeffs, num_color_coeffs>(
            new VertexColorCost(shape_model, blendshapes, color_model, num_shape_coeffs,
                                num_blendshape_coeffs, num_color_coeffs, vertex_id, image)));
    };

private:
    const eos::morphablemodel::PcaModel&
        shape_model; // Or store as pointer (non-owning) or std::reference_wrapper?
    const eos::morphablemodel::Blendshapes& blendshapes;
    const eos::morphablemodel::PcaModel& color_model;
    const int num_shape_coeffs;
    const int num_blendshape_coeffs;
    const int num_color_coeffs;
    const int vertex_id;
    const eos::core::Image3u& image; // the observed image
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
Eigen::Vector3<T> get_shape_at_point(const eos::morphablemodel::PcaModel& shape_model,
                                     const eos::morphablemodel::Blendshapes& blendshapes, int vertex_id,
                                     Eigen::Map<const Eigen::VectorX<T>> shape_coeffs,
                                     Eigen::Map<const Eigen::VectorX<T>> blendshape_coeffs)
{
    // Computing Shape = mean + shape_basis*shape_coeffs + blendshapes*blendshape_coeffs:
    const Eigen::Vector3f mean = shape_model.get_mean_at_point(vertex_id);
    const Eigen::Vector3<T> shape_vector =
        shape_model.get_rescaled_pca_basis_at_point(vertex_id).leftCols(shape_coeffs.size()).cast<T>() *
        shape_coeffs;
    Eigen::Vector3<T> expression_vector(T(0.0), T(0.0), T(0.0));
    for (std::size_t i = 0; i < blendshape_coeffs.size(); i++)
    {
        expression_vector.x() += T(blendshapes[i].deformation(vertex_id * 3 + 0)) * blendshape_coeffs(i);
        expression_vector.y() += T(blendshapes[i].deformation(vertex_id * 3 + 1)) * blendshape_coeffs(i);
        expression_vector.z() += T(blendshapes[i].deformation(vertex_id * 3 + 2)) * blendshape_coeffs(i);
    }

    return Eigen::Vector3<T>(mean.cast<T>() + shape_vector + expression_vector);
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
Eigen::Vector3<T> get_vertex_color_at_point(const eos::morphablemodel::PcaModel& color_model, int vertex_id,
                                            Eigen::Map<const Eigen::VectorX<T>> color_coeffs)
{
    const Eigen::Vector3f mean = color_model.get_mean_at_point(vertex_id);
    const Eigen::Vector3<T> color_vector =
        color_model.get_rescaled_pca_basis_at_point(vertex_id).leftCols(color_coeffs.size()).cast<T>() *
        color_coeffs;

    return Eigen::Vector3<T>(mean.cast<T>() + color_vector);
};

/**
 * Creates a matrix for a right-handed, symmetric perspective-view frustrum.
 *
 * The function follows the OpenGL clip volume definition, which is also the GLM default. The near and far
 * clip planes correspond to z normalized device coordinates of -1 and +1 respectively.
 *
 * This function is equivalent to glm::perspectiveRH_NO(...).
 *
 * More details can be found on the gluPerspective man page:
 * https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml.
 *
 * @param[in] fov_y Specifies the field of view angle in the y direction. Expressed in radians.
 * @param[in] aspect Specifies the aspect ratio that determines the field of view in the x direction. The
 * aspect ratio is the ratio of x (width) to y (height).
 * @param[in] z_near Specifies the distance from the viewer to the near clipping plane (always positive).
 * @param[in] z_far Specifies the distance from the viewer to the far clipping plane (always positive).
 * @tparam T A floating-point scalar type, ceres::Jet, or similar compatible type.
 * @return The corresponding perspective projection matrix.
 */
template <typename T>
Eigen::Matrix4<T> perspective(T fov_y, T aspect, T z_near, T z_far)
{
    // Will this assert work? std::abs probably won't work on T?
    // assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

    const T tan_half_fov_y = tan(fov_y / static_cast<T>(2)); // ceres::tan?

    // Maybe construct with static_cast<T>(0)? => No, doesn't have c'tor.
    // Could do Eigen::Matrix4<T> result = {{1, 2, 3, 4}, {1, 2, 3, 4}...} I think.
    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Zero();
    result(0, 0) = static_cast<T>(1) / (aspect * tan_half_fov_y);
    result(1, 1) = static_cast<T>(1) / (tan_half_fov_y);
    result(2, 2) = -(z_far + z_near) / (z_far - z_near);
    result(2, 3) = -static_cast<T>(1);
    result(3, 2) = -(static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
    return result;
}

/**
 * Project the given point_3d (from object coordinates) into window coordinates.
 *
 * The function follows the OpenGL clip volume definition. The near and far clip planes correspond to
 * z normalized device coordinates of -1 and +1 respectively.
 * This function is equivalent to glm::projectNO(...).
 *
 * More details can be found on the gluProject man page:
 * https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluProject.xml.
 *
 * @param[in] point_3d A 3D point in object coordinates.
 * @param[in] modelview_matrix A model-view matrix, transforming the point into view (camera) space.
 * @param[in] projection_matrix The projection matrix to be used.
 * @param[in] viewport The viewport transformation to be used.
 * @tparam T A floating-point scalar type, ceres::Jet, or similar compatible type.
 * @return Return the computed window coordinates.
 */
template <typename T>
Eigen::Vector3<T> project(const Eigen::Vector3<T>& point_3d, const Eigen::Matrix4<T>& modelview_matrix,
                          const Eigen::Matrix4<T>& projection_matrix, const Eigen::Vector4<T>& viewport)
{
    Eigen::Vector4<T> projected_point = projection_matrix * modelview_matrix * point_3d.homogeneous();
    projected_point /= projected_point.w();
    projected_point =
        projected_point * static_cast<T>(0.5) +
        Eigen::Vector4<T>(static_cast<T>(0.5), static_cast<T>(0.5), static_cast<T>(0.5), static_cast<T>(0.5));
    projected_point.x() = projected_point.x() * T(viewport(2)) + T(viewport(0));
    projected_point.y() = projected_point.y() * T(viewport(3)) + T(viewport(1));

    return projected_point.head<3>();
}

} /* namespace fitting */
} /* namespace eos */

#endif /* EOS_CERES_NONLINEAR_HPP */
