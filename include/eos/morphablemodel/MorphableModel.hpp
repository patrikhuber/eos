/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/MorphableModel.hpp
 *
 * Copyright 2014-2017 Patrik Huber
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

#ifndef MORPHABLEMODEL_HPP_
#define MORPHABLEMODEL_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/morphablemodel/PcaModel.hpp"

#include "cereal/access.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#include "eos/morphablemodel/io/eigen_cerealisation.hpp"
#include "cereal/archives/binary.hpp"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "Eigen/Core"

#include <array>
#include <cstdint>
#include <vector>
#include <fstream>

namespace eos {
namespace morphablemodel {

// Forward declaration:
core::Mesh sample_to_mesh(
    const Eigen::VectorXf& shape_instance, const Eigen::VectorXf& color_instance,
    const std::vector<std::array<int, 3>>& tvi, const std::vector<std::array<int, 3>>& tci,
    const std::vector<std::array<double, 2>>& texture_coordinates = std::vector<std::array<double, 2>>());

/**
 * @brief A class representing a 3D Morphable Model, consisting
 * of a shape- and colour (albedo) PCA model.
 *
 * For the general idea of 3DMMs see T. Vetter, V. Blanz,
 * 'A Morphable Model for the Synthesis of 3D Faces', SIGGRAPH 1999.
 */
class MorphableModel
{
public:
    MorphableModel() = default;

    /**
     * Create a Morphable Model from a shape and a colour PCA model, and optional
     * texture coordinates.
     *
     * @param[in] shape_model A PCA model over the shape.
     * @param[in] color_model A PCA model over the colour (albedo).
     * @param[in] texture_coordinates Optional texture coordinates for every vertex.
     */
    MorphableModel(
        PcaModel shape_model, PcaModel color_model,
        std::vector<std::array<double, 2>> texture_coordinates = std::vector<std::array<double, 2>>())
        : shape_model(shape_model), color_model(color_model), texture_coordinates(texture_coordinates){};

    /**
     * Returns the PCA shape model of this Morphable Model.
     *
     * @return The shape model.
     */
    const PcaModel& get_shape_model() const
    {
        return shape_model;
    };

    /**
     * Returns the PCA colour (albedo) model of this Morphable Model.
     *
     * @return The colour model.
     */
    const PcaModel& get_color_model() const
    {
        return color_model;
    };

    /**
     * Returns the mean of the shape- and colour model as a Mesh.
     *
     * @return An mesh instance of the mean of the Morphable Model.
     */
    core::Mesh get_mean() const
    {
        assert(shape_model.get_data_dimension() == color_model.get_data_dimension() ||
               !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal
                                    // for both models, or, alternatively, it has to be a shape-only model.

        const Eigen::VectorXf shape = shape_model.get_mean();
        const Eigen::VectorXf color = color_model.get_mean();

        core::Mesh mesh;
        if (has_texture_coordinates())
        {
            mesh = sample_to_mesh(shape, color, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list(), texture_coordinates);
        } else
        {
            mesh = sample_to_mesh(shape, color, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list());
        }
        return mesh;
    };

    /**
     * Draws a random sample from the model, where the coefficients
     * for the shape- and colour models are both drawn from a standard
     * normal (or with the given standard deviation).
     *
     * If the Morphable Model is a shape-only model, the returned mesh will
     * not contain any colour data.
     *
     * @param[in] engine Random number engine used to draw random coefficients.
     * @param[in] shape_sigma The shape model standard deviation.
     * @param[in] color_sigma The colour model standard deviation.
     * @return A random sample from the model.
     */
    template <class RNG>
    core::Mesh draw_sample(RNG& engine, float shape_sigma = 1.0f, float color_sigma = 1.0f) const
    {
        assert(shape_model.get_data_dimension() == color_model.get_data_dimension() ||
               !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal
                                    // for both models, or, alternatively, it has to be a shape-only model.

        const Eigen::VectorXf shape_sample = shape_model.draw_sample(engine, shape_sigma);
        const Eigen::VectorXf color_sample = color_model.draw_sample(engine, color_sigma);

        core::Mesh mesh;
        if (has_texture_coordinates())
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list(), texture_coordinates);
        } else
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list());
        }
        return mesh;
    };

    /**
     * Returns a sample from the model with the given shape- and
     * colour PCA coefficients.
     *
     * If one of the given vectors is empty, the mean is used.
     * The coefficient vectors should contain normalised, i.e. standard normal distributed coefficients.
     * If the Morphable Model is a shape-only model (without colour model), make sure to
     * leave \c color_coefficients empty.
     * If a partial coefficient vector is given, it is filled with zeros up to the end.
     *
     * @param[in] shape_coefficients The PCA coefficients used to generate the shape sample.
     * @param[in] color_coefficients The PCA coefficients used to generate the vertex colouring.
     * @return A model instance with given coefficients.
     */
    core::Mesh draw_sample(std::vector<float> shape_coefficients, std::vector<float> color_coefficients) const
    {
        assert(shape_model.get_data_dimension() == color_model.get_data_dimension() ||
               !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal
                                    // for both models, or, alternatively, it has to be a shape-only model.

        Eigen::VectorXf shape_sample;
        Eigen::VectorXf color_sample;

        if (shape_coefficients.empty())
        {
            shape_sample = shape_model.get_mean();
        } else
        {
            shape_sample = shape_model.draw_sample(shape_coefficients);
        }
        if (color_coefficients.empty())
        {
            color_sample = color_model.get_mean();
        } else
        {
            color_sample = color_model.draw_sample(color_coefficients);
        }

        core::Mesh mesh;
        if (has_texture_coordinates())
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list(), texture_coordinates);
        } else
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list());
        }
        return mesh;
    };

    /**
     * Returns true if this Morphable Model contains a colour
     * model. Returns false if it is a shape-only model.
     *
     * @return True if the Morphable Model has a colour model (i.e. is not a shape-only model).
     */
    bool has_color_model() const
    {
        return (color_model.get_mean().size() > 0);
    };

    /**
     * Returns the texture coordinates for all the vertices in the model.
     *
     * @return The texture coordinates for the model vertices.
     */
    std::vector<std::array<double, 2>> get_texture_coordinates() const
    {
        return texture_coordinates;
    };

private:
    PcaModel shape_model;                                   ///< A PCA model of the shape
    PcaModel color_model;                                   ///< A PCA model of vertex colour information
    std::vector<std::array<double, 2>> texture_coordinates; ///< uv-coordinates for every vertex

    /**
     * Returns whether the model has texture mapping coordinates, i.e.
     * coordinates for every vertex.
     *
     * @return True if the model contains texture mapping coordinates.
     */
    bool has_texture_coordinates() const
    {
        return texture_coordinates.size() > 0 ? true : false;
    };

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] archive The archive to serialise to (or to serialise from).
     * @param[in] version Version number of the archive.
     * @throw std::runtime_error When the model file doesn't have the most recent version (=1).
     */
    template <class Archive>
    void serialize(Archive& archive, const std::uint32_t version)
    {
        if (version != 1)
        {
            throw std::runtime_error("The model file you are trying to load is in an old format. Please "
                                     "download the most recent model files.");
        }
        archive(CEREAL_NVP(shape_model), CEREAL_NVP(color_model), CEREAL_NVP(texture_coordinates));
    };
};

/**
 * Helper method to load a Morphable Model from
 * a cereal::BinaryInputArchive from the harddisk.
 *
 * @param[in] filename Filename to a model.
 * @return The loaded Morphable Model.
 * @throw std::runtime_error When the file given in \c filename fails to be opened (most likely because the
 * file doesn't exist).
 */
inline MorphableModel load_model(std::string filename)
{
    MorphableModel model;

    std::ifstream file(filename, std::ios::binary);
    if (file.fail())
    {
        throw std::runtime_error("Error opening given file: " + filename);
    }
    cereal::BinaryInputArchive input_archive(file);
    input_archive(model);

    return model;
};

/**
 * Helper method to save a Morphable Model to the
 * harddrive as cereal::BinaryOutputArchive.
 *
 * @param[in] model The model to be saved.
 * @param[in] filename Filename for the model.
 */
inline void save_model(MorphableModel model, std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(file);
    output_archive(model);
};

/**
 * Helper function that creates a Mesh from given shape and colour PCA
 * instances. Needs the vertex index lists as well to assemble the mesh -
 * and optional texture coordinates.
 *
 * If \c color is empty, it will create a mesh without vertex colouring.
 *
 * @param[in] shape_instance PCA shape model instance.
 * @param[in] color_instance PCA colour model instance.
 * @param[in] tvi Triangle vertex indices.
 * @param[in] tci Triangle colour indices (usually identical to the vertex indices).
 * @param[in] texture_coordinates Optional texture coordinates for each vertex.
 * @return A mesh created from given parameters.
 */
inline core::Mesh sample_to_mesh(const Eigen::VectorXf& shape_instance, const Eigen::VectorXf& color_instance,
                                 const std::vector<std::array<int, 3>>& tvi,
                                 const std::vector<std::array<int, 3>>& tci,
                                 const std::vector<std::array<double, 2>>&
                                     texture_coordinates /* = std::vector<std::array<double, 2>>() */)
{
    assert(shape_instance.rows() == color_instance.rows() ||
           color_instance.size() == 0); // The number of vertices (= model.getDataDimension() / 3) has to be
                                        // equal for both models, or, alternatively, it has to be a shape-only
                                        // model.

    const auto num_vertices = shape_instance.rows() / 3;

    core::Mesh mesh;

    // Construct the mesh vertices:
    mesh.vertices.resize(num_vertices);
    for (auto i = 0; i < num_vertices; ++i)
    {
        mesh.vertices[i] = glm::tvec4<float>(shape_instance(i * 3 + 0), shape_instance(i * 3 + 1),
                                             shape_instance(i * 3 + 2), 1.0f);
    }

    // Assign the vertex colour information if it's not a shape-only model:
    if (color_instance.size() > 0)
    {
        mesh.colors.resize(num_vertices);
        for (auto i = 0; i < num_vertices; ++i)
        {
            mesh.colors[i] = glm::tvec3<float>(color_instance(i * 3 + 0), color_instance(i * 3 + 1),
                                               color_instance(i * 3 + 2)); // We use RGB order everywhere
        }
    }

    // Assign the triangle lists:
    mesh.tvi = tvi;
    mesh.tci = tci; // tci will be empty in case of a shape-only model

    // Texture coordinates, if the model has them:
    if (!texture_coordinates.empty())
    {
        mesh.texcoords.resize(num_vertices);
        for (auto i = 0; i < num_vertices; ++i)
        {
            mesh.texcoords[i] = glm::tvec2<float>(texture_coordinates[i][0], texture_coordinates[i][1]);
        }
    }

    return mesh;
};

} /* namespace morphablemodel */
} /* namespace eos */

CEREAL_CLASS_VERSION(eos::morphablemodel::MorphableModel, 1);

#endif /* MORPHABLEMODEL_HPP_ */
