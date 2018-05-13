/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/ExpressionModel.hpp
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

#ifndef EOS_EXPRESSION_MODEL_HPP
#define EOS_EXPRESSION_MODEL_HPP

#include "eos/morphablemodel/PcaModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/cpp17/variant.hpp"

#include "Eigen/Core"

#include <vector>
#include <cassert>
#include <stdexcept>

namespace eos {
namespace morphablemodel {

/**
 * @brief Type alias to represent an expression model, which can either consist of blendshapes or a PCA model.
 *
 * Defining a type alias so that we don't have to spell out the type everywhere.
 */
using ExpressionModel = cpp17::variant<PcaModel, Blendshapes>;

/**
 * Returns a sample from the model with the given expression coefficients.
 *
 * The underlying expression model can be both a PcaModel as well as a Blendshapes model.
 * If a partial coefficient vector is given, it is filled with zeros up to the end.
 *
 * @param[in] expression_coefficients The coefficients used to generate the expression sample.
 * @return A model instance with given coefficients.
 */
Eigen::VectorXf draw_sample(const ExpressionModel& expression_model,
                            std::vector<float> expression_coefficients)
{
    Eigen::VectorXf expression_sample;
    // Get a sample of the expression model, depending on whether it's a PcaModel or Blendshapes:
    if (cpp17::holds_alternative<PcaModel>(expression_model))
    {
        const auto& pca_expression_model = cpp17::get<PcaModel>(expression_model);
        if (expression_coefficients.empty())
        {
            expression_sample = pca_expression_model.get_mean();
        } else
        {
            expression_sample = pca_expression_model.draw_sample(expression_coefficients);
        }
    } else if (cpp17::holds_alternative<Blendshapes>(expression_model))
    {
        const auto& expression_blendshapes = cpp17::get<Blendshapes>(expression_model);
        assert(expression_blendshapes.size() > 0);
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
        throw std::runtime_error("The given ExpressionModel doesn't contain an expression model in the form "
                                 "of a PcaModel or Blendshapes.");
    }
    return expression_sample;
};

} /* namespace morphablemodel */
} /* namespace eos */

#endif /* EOS_EXPRESSION_MODEL_HPP */
