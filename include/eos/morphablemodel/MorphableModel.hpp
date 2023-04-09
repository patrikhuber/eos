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

#ifndef EOS_MORPHABLEMODEL_HPP
#define EOS_MORPHABLEMODEL_HPP

#include "eos/core/Mesh.hpp"
#include "eos/morphablemodel/PcaModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/ExpressionModel.hpp"
#include "eos/cpp17/optional.hpp"
#include "eos/cpp17/variant.hpp"
#include "eos/cpp17/clamp.hpp"

#include "cereal/access.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/unordered_map.hpp"
#include "eos/cpp17/optional_serialization.hpp"
#include "eos/cpp17/variant_serialization.hpp"
#include "eos/morphablemodel/io/eigen_cerealisation.hpp"
#include "cereal/archives/binary.hpp"

#include "Eigen/Core"

#include <array>
#include <cstdint>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <cassert>
#include <stdexcept>

namespace eos {
namespace morphablemodel {

// Forward declaration:
core::Mesh sample_to_mesh(
    const Eigen::VectorXf& shape_instance, const Eigen::VectorXf& color_instance,
    const std::vector<std::array<int, 3>>& tvi, const std::vector<std::array<int, 3>>& tci,
    const std::vector<std::array<double, 2>>& texture_coordinates = std::vector<std::array<double, 2>>(),
    const std::vector<std::array<int, 3>>& texture_triangle_indices = std::vector<std::array<int, 3>>());

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
     * @param[in] landmark_definitions A set of landmark definitions, mapping from identifiers to vertex
     * numbers.
     * @param[in] texture_coordinates Optional texture coordinates for every vertex.
     * @param[in] texture_triangle_indices Optional triangulation for the texture coordinates.
     */
    MorphableModel(
        PcaModel shape_model, PcaModel color_model,
        cpp17::optional<std::unordered_map<std::string, int>> landmark_definitions = cpp17::nullopt,
        std::vector<std::array<double, 2>> texture_coordinates = std::vector<std::array<double, 2>>(),
        std::vector<std::array<int, 3>> texture_triangle_indices = std::vector<std::array<int, 3>>())
        : shape_model(shape_model), color_model(color_model), landmark_definitions(landmark_definitions),
          texture_coordinates(texture_coordinates), texture_triangle_indices(texture_triangle_indices){};

    /**
     * Create a Morphable Model from a shape and a colour PCA model, an expression PCA model or blendshapes,
     * and optional texture coordinates.
     *
     * @param[in] shape_model A PCA model over the shape.
     * @param[in] expression_model A PCA model over expressions, or a set of blendshapes.
     * @param[in] color_model A PCA model over the colour (albedo).
     * @param[in] landmark_definitions A set of landmark definitions, mapping from identifiers to vertex
     * numbers.
     * @param[in] texture_coordinates Optional texture coordinates for every vertex.
     * @param[in] texture_triangle_indices Optional triangulation for the texture coordinates.
     */
    MorphableModel(
        PcaModel shape_model, ExpressionModel expression_model, PcaModel color_model,
        cpp17::optional<std::unordered_map<std::string, int>> landmark_definitions = cpp17::nullopt,
        std::vector<std::array<double, 2>> texture_coordinates = std::vector<std::array<double, 2>>(),
        std::vector<std::array<int, 3>> texture_triangle_indices = std::vector<std::array<int, 3>>())
        : shape_model(shape_model), color_model(color_model), landmark_definitions(landmark_definitions),
          texture_coordinates(texture_coordinates), texture_triangle_indices(texture_triangle_indices)
    {
        // Note: We may want to check/assert that the dimensions all match?
        this->expression_model = expression_model;
    };

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
     * Returns the shape expression model, if this Morphable Model has one.
     *
     * Returns an empty cpp17::optional if the Morphable Model does not have a separate expression
     * model (check with MorphableModel::has_separate_expression_model()).
     * If it does have an expression model, an std::variant<PcaModel, Blendshapes> is returned -
     * that is, either a PcaModel (if it is an expression PCA model), or Blendshapes.
     *
     * @return The expression model or an empty optional.
     */
    const auto& get_expression_model() const
    {
        return expression_model;
    }

    /**
     * Returns the mean of the shape (identity, and expressions, if present) and colour model as a Mesh.
     *
     * If the model contains a separate PCA expression model, the mean of that model is added and the overall
     * shape mean is returned.
     *
     * @return An mesh instance of the mean of the Morphable Model.
     */
    core::Mesh get_mean() const
    {
        assert(shape_model.get_data_dimension() == color_model.get_data_dimension() ||
               !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal
                                    // for both models, or, alternatively, it has to be a shape-only model.

        Eigen::VectorXf shape = shape_model.get_mean();
        const Eigen::VectorXf color = color_model.get_mean();

        // If there is a PCA expression model, add that model's mean. If there is no expression model, or
        // blendshapes, there's nothing to add.
        if (get_expression_model_type() == ExpressionModelType::PcaModel)
        {
            shape += cpp17::get<PcaModel>(expression_model.value()).get_mean();
        }

        core::Mesh mesh;
        if (has_texture_coordinates())
        {
            mesh =
                sample_to_mesh(shape, color, shape_model.get_triangle_list(), color_model.get_triangle_list(),
                               texture_coordinates, texture_triangle_indices);
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
                                  color_model.get_triangle_list(), texture_coordinates,
                                  texture_triangle_indices);
        } else
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list());
        }
        return mesh;
    };

    /**
     * Returns a sample from the model with the given shape and
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
                                  color_model.get_triangle_list(), texture_coordinates,
                                  texture_triangle_indices);
        } else
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list());
        }
        return mesh;
    };

    /**
     * Returns a sample from the model with the given shape, expression and colour PCA coefficients.
     *
     * If you call this method on a MorphableModel that doesn't contain an expression model, it'll throw an
     * exception.
     * If one of the given vectors is empty, the mean is used. The coefficient vectors should
     * contain normalised, i.e. standard normal distributed coefficients. If the Morphable Model is a
     * shape-only model (without colour model), make sure to leave \c color_coefficients empty. If a partial
     * coefficient vector is given, it is filled with zeros up to the end.
     *
     * @param[in] shape_coefficients The PCA coefficients used to generate the shape sample.
     * @param[in] expression_coefficients The PCA coefficients used to generate the expression sample.
     * @param[in] color_coefficients The PCA coefficients used to generate the vertex colouring.
     * @return A model instance with given coefficients.
     */
    core::Mesh draw_sample(std::vector<float> shape_coefficients, std::vector<float> expression_coefficients,
                           std::vector<float> color_coefficients) const
    {
        assert(shape_model.get_data_dimension() == color_model.get_data_dimension() ||
               !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal
                                    // for both models, or, alternatively, it has to be a shape-only model.
        assert(has_separate_expression_model());

        Eigen::VectorXf shape_sample;
        Eigen::VectorXf expression_sample;
        Eigen::VectorXf color_sample;

        if (shape_coefficients.empty())
        {
            shape_sample = shape_model.get_mean();
        } else
        {
            shape_sample = shape_model.draw_sample(shape_coefficients);
        }
        // Get a sample of the expression model, depending on whether it's a PcaModel or Blendshapes:
        if (get_expression_model_type() == ExpressionModelType::PcaModel)
        {
            const auto& pca_expression_model = cpp17::get<PcaModel>(expression_model.value());
            assert(pca_expression_model.get_data_dimension() == shape_model.get_data_dimension());
            if (expression_coefficients.empty())
            {
                expression_sample = pca_expression_model.get_mean();
            } else
            {
                expression_sample = pca_expression_model.draw_sample(expression_coefficients);
            }
        } else if (get_expression_model_type() == ExpressionModelType::Blendshapes)
        {
            const auto& expression_blendshapes = cpp17::get<Blendshapes>(expression_model.value());
            assert(expression_blendshapes.size() > 0);
            assert(expression_blendshapes[0].deformation.rows() == shape_model.get_data_dimension());
            if (expression_coefficients.empty())
            {
                expression_sample.setZero(expression_blendshapes[0].deformation.rows());
            } else
            {
                expression_sample = to_matrix(expression_blendshapes) *
                                    Eigen::Map<const Eigen::VectorXf>(expression_coefficients.data(),
                                                                      expression_coefficients.size());
            }
        } else
        {
            throw std::runtime_error("This MorphableModel doesn't contain an expression model in the form of "
                                     "a PcaModel or Blendshapes.");
        }
        shape_sample += expression_sample;

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
                                  color_model.get_triangle_list(), texture_coordinates,
                                  texture_triangle_indices);
        } else
        {
            mesh = sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(),
                                  color_model.get_triangle_list());
        }
        return mesh;
    };

    /**
     * Returns true if this Morphable Model contains a colour model. Returns false if it is a shape-only
     * model.
     *
     * @return True if the Morphable Model has a colour model (i.e. is not a shape-only model).
     */
    bool has_color_model() const
    {
        return (color_model.get_mean().size() > 0);
    };

    /**
     * Returns true if this Morphable Model contains a separate expression model, in the form of a PcaModel or
     * Blendshapes.
     *
     * @return True if the Morphable Model has a separate expression model.
     */
    bool has_separate_expression_model() const
    {
        return expression_model.has_value();
    };

    /**
     * Returns the landmark definitions for this Morphable Model, which might be an empty optional, if the
     * model doesn't contain any.
     *
     * The landmark definitions are define mappings from a set of global landmark identifiers, like for
     * example "eye.right.outer_corner", to the model's respective vertex indices.
     * A MorphableModel may or may not contain these landmark definitions, depending on how it was created.
     *
     * @return The landmark definitions of this MorphableModel.
     */
    const auto& get_landmark_definitions() const
    {
        return landmark_definitions;
    };

    /**
     * Sets the landmark definitions for this Morphable Model.
     *
     * The landmark definitions are define mappings from a set of global landmark identifiers, like for
     * example "eye.right.outer_corner", to the model's respective vertex indices.
     *
     */
    const void set_landmark_definitions(cpp17::optional<std::unordered_map<std::string, int>> updated_landmark_definitions)
    {
        landmark_definitions = updated_landmark_definitions;
    };

    /**
     * Returns the texture coordinates for all the vertices in the model.
     *
     * @return The texture coordinates for the model vertices.
     */
    const std::vector<std::array<double, 2>>& get_texture_coordinates() const
    {
        return texture_coordinates;
    };

    /**
     * Returns the triangulation (the triangles that make up the uv mapping) for the texture coordinates.
     *
     * @return The triangulation for the texture coordinates.
     */
    const std::vector<std::array<int, 3>>& get_texture_triangle_indices() const
    {
        return texture_triangle_indices;
    };

    /**
     * @brief The type of the expression model that this MorphableModel contains.
     *
     * A MorphableModel can contain no expression model, an expression model consisting of blendshapes, or a
     * PCA model of expressions.
     */
    enum class ExpressionModelType { None, Blendshapes, PcaModel };

    /**
     * Returns the type of the expression model: None, Blendshapes or PcaModel.
     *
     * @return The type of the expression model.
     */
    ExpressionModelType get_expression_model_type() const
    {
        if (!expression_model)
        {
            return ExpressionModelType::None;
        }
        if (cpp17::holds_alternative<Blendshapes>(expression_model.value()))
        {
            return ExpressionModelType::Blendshapes;
        } else if (cpp17::holds_alternative<PcaModel>(expression_model.value()))
        {
            return ExpressionModelType::PcaModel;
        } else
        {
            // We should never get here - but this silences the "not all control paths return a value"
            // compiler warning.
            throw std::runtime_error("The ExpressionModel contains something, but it's not Blendshapes or a "
                                     "PcaModel. This should not happen.");
        }
    };

private:
    PcaModel shape_model;                              ///< A PCA model of the shape
    PcaModel color_model;                              ///< A PCA model of vertex colour information
    cpp17::optional<ExpressionModel> expression_model; ///< Blendshapes or PcaModel
    cpp17::optional<std::unordered_map<std::string, int>> landmark_definitions; ///< A set of landmark
                                                                                ///< definitions for the
                                                                                ///< model, mapping from
                                                                                ///< identifiers to vertex
                                                                                ///< numbers
    std::vector<std::array<double, 2>> texture_coordinates;   ///< uv-coordinates for every vertex
    std::vector<std::array<int, 3>> texture_triangle_indices; ///< Triangulation for the uv-coordinates

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
     * @throw std::runtime_error When the model file has version <1 (the very old cv::Mat matrix format) or an
     *                           unknown value (e.g. >4).
     */
    template <class Archive>
    void serialize(Archive& archive, const std::uint32_t version)
    {
        if (version < 1)
        {
            throw std::runtime_error("The model file you are trying to load is in an old format. Please "
                                     "download the most recent model files.");
        }
        if (version == 1)
        {
            archive(CEREAL_NVP(shape_model), CEREAL_NVP(color_model), CEREAL_NVP(texture_coordinates));
        } else if (version == 2)
        {
            archive(CEREAL_NVP(shape_model), CEREAL_NVP(color_model), CEREAL_NVP(expression_model),
                    CEREAL_NVP(texture_coordinates));
        } else if (version == 3)
        {
            archive(CEREAL_NVP(shape_model), CEREAL_NVP(color_model), CEREAL_NVP(expression_model),
                    CEREAL_NVP(landmark_definitions), CEREAL_NVP(texture_coordinates));
        } else if (version == 4)
        {
            archive(CEREAL_NVP(shape_model), CEREAL_NVP(color_model), CEREAL_NVP(expression_model),
                    CEREAL_NVP(landmark_definitions), CEREAL_NVP(texture_coordinates),
                    CEREAL_NVP(texture_triangle_indices));
        } else
        {
            throw std::runtime_error("The model file you are trying to load has an unknown version number: " +
                                     std::to_string(version));
        }
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
    if (!file)
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
    if (!file)
    {
        throw std::runtime_error("Error creating given file: " + filename);
    }
    cereal::BinaryOutputArchive output_archive(file);
    output_archive(model);
};

/**
 * Helper function that creates a Mesh from given shape and colour PCA
 * instances. Needs the vertex index lists as well to assemble the mesh -
 * and optional texture coordinates.
 *
 * If \c color_instance is empty, it will create a mesh without vertex colouring.
 * Colour values are assumed to be in the range [0, 1] and will be clamped to [0, 1].
 *
 * @param[in] shape_instance PCA shape model instance.
 * @param[in] color_instance PCA colour model instance.
 * @param[in] tvi Triangle vertex indices.
 * @param[in] tci Triangle colour indices (usually identical to the vertex indices).
 * @param[in] texture_coordinates Optional texture coordinates for each vertex.
 * @param[in] texture_triangle_indices Optional triangulation for the texture coordinates.
 * @return A mesh created from given parameters.
 */
inline core::Mesh sample_to_mesh(
    const Eigen::VectorXf& shape_instance, const Eigen::VectorXf& color_instance,
    const std::vector<std::array<int, 3>>& tvi, const std::vector<std::array<int, 3>>& tci,
    const std::vector<std::array<double, 2>>&
        texture_coordinates /* = std::vector<std::array<double, 2>>() */,
    const std::vector<std::array<int, 3>>& texture_triangle_indices /* = std::vector<std::array<int, 3>>() */)
{
    assert(shape_instance.rows() == color_instance.rows() ||
           color_instance.size() == 0); // The number of vertices (= model.getDataDimension() / 3) has to be
                                        // equal for both models, or, alternatively, it has to be a shape-only
                                        // model.
    assert(texture_coordinates.size() == 0 || texture_coordinates.size() == (shape_instance.rows() / 3) ||
           !texture_triangle_indices
                .empty()); // No texture coordinates are ok. If there are texture
                           // coordinates given, their number needs to be identical to the
                           // number of vertices, or texture_triangle_indices needs to be given.

    const auto num_vertices = shape_instance.rows() / 3;

    core::Mesh mesh;

    // Construct the mesh vertices:
    mesh.vertices.resize(num_vertices);
    for (auto i = 0; i < num_vertices; ++i)
    {
        mesh.vertices[i] = Eigen::Vector3f(
            shape_instance(i * 3 + 0), shape_instance(i * 3 + 1),
            shape_instance(i * 3 + 2)); // Note: This can probably be simplified now, Eigen on both sides!
    }

    // Assign the vertex colour information if it's not a shape-only model:
    if (color_instance.size() > 0)
    {
        mesh.colors.resize(num_vertices);
        for (auto i = 0; i < num_vertices; ++i)
        {
            mesh.colors[i] = Eigen::Vector3f(
                cpp17::clamp(color_instance(i * 3 + 0), 0.0f, 1.0f),
                cpp17::clamp(color_instance(i * 3 + 1), 0.0f, 1.0f),
                cpp17::clamp(color_instance(i * 3 + 2), 0.0f, 1.0f)); // We use RGB order everywhere.
        }
    }

    // Assign the triangle lists:
    mesh.tvi = tvi;
    mesh.tci = tci; // tci will be empty in case of a shape-only model

    // Texture coordinates, if the model has them:
    if (!texture_coordinates.empty())
    {
        mesh.texcoords.resize(texture_coordinates.size());
        for (auto i = 0; i < texture_coordinates.size(); ++i)
        {
            mesh.texcoords[i] = Eigen::Vector2f(texture_coordinates[i][0], texture_coordinates[i][1]);
        }
        if (!texture_triangle_indices.empty())
        {
            mesh.tti = texture_triangle_indices;
        }
    }

    return mesh;
};

} /* namespace morphablemodel */
} /* namespace eos */

CEREAL_CLASS_VERSION(eos::morphablemodel::MorphableModel, 4);

#endif /* EOS_MORPHABLEMODEL_HPP */
