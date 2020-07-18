/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/closest_edge_fitting.hpp
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

#ifndef EOS_CLOSEST_EDGE_FITTING_HPP
#define EOS_CLOSEST_EDGE_FITTING_HPP

#include "eos/core/Mesh.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/normals.hpp"
#include "eos/render/vertex_visibility.hpp"
#include "eos/render/ProjectionType.hpp"

#include "nanoflann.hpp"

#include "glm/common.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include "Eigen/Core"

#include <vector>
#include <algorithm>
#include <utility>
#include <cstddef>

namespace eos {
namespace fitting {

/**
 * @brief Computes the vertices that lie on occluding boundaries, given a particular pose.
 *
 * This algorithm computes the edges that lie on occluding boundaries of the mesh.
 * It performs a visibility test of each vertex, and returns a list of the (unique)
 * vertices that make up the boundary edges.
 * An edge is defined as the line whose two adjacent faces normals flip the sign.
 *
 * @param[in] mesh The mesh to use.
 * @param[in] edge_topology The edge topology of the given mesh.
 * @param[in] R The rotation (pose) under which the occluding boundaries should be computed.
 * @param[in] projection_type Indicates whether the projection used is orthographic or perspective.
 * @param[in] perform_self_occlusion_check Check the computed boundary vertices for self-occlusion and remove these.
 * @return A vector with unique vertex id's making up the edges.
 */
inline std::vector<int> occluding_boundary_vertices(const core::Mesh& mesh,
                                                    const morphablemodel::EdgeTopology& edge_topology,
                                                    glm::mat4x4 R, render::ProjectionType projection_type,
                                                    bool perform_self_occlusion_check = true)
{
    // Rotate the mesh:
    std::vector<glm::vec4> rotated_vertices;
    std::for_each(begin(mesh.vertices), end(mesh.vertices), [&rotated_vertices, &R](const auto& v) {
        rotated_vertices.push_back(R * glm::vec4(v[0], v[1], v[2], 1.0f));
    });

    // Compute the face normals of the rotated mesh:
    std::vector<glm::vec3> facenormals;
    for (const auto& f : mesh.tvi)
    { // for each face (triangle):
        const auto n =
            render::compute_face_normal(glm::vec3(rotated_vertices[f[0]]), glm::vec3(rotated_vertices[f[1]]),
                                        glm::vec3(rotated_vertices[f[2]]));
        facenormals.push_back(n);
    }

    // Find occluding edges:
    std::vector<int> occluding_edges_indices;
    for (int edge_idx = 0; edge_idx < edge_topology.adjacent_faces.size();
         ++edge_idx) // For each edge... Ef contains the indices of the two adjacent faces
    {
        const auto& edge = edge_topology.adjacent_faces[edge_idx];
        if (edge[0] == 0) // ==> NOTE/Todo Need to change this if we use 0-based indexing!
        {
            // Edges with a zero index lie on the mesh boundary, i.e. they are only
            // adjacent to one face.
            continue;
        }
        // Compute the occluding edges as those where the two adjacent face normals
        // differ in the sign of their z-component:
        // Changing from 1-based indexing to 0-based!
        if (glm::sign(facenormals[edge[0] - 1].z) != glm::sign(facenormals[edge[1] - 1].z))
        {
            // It's an occluding edge, store the index:
            occluding_edges_indices.push_back(edge_idx);
        }
    }
    // Select the vertices lying at the two ends of the occluding edges and remove duplicates:
    // (This is what EdgeTopology::adjacent_vertices is needed for).
    std::vector<int> occluding_vertices; // The model's contour vertices
    for (auto edge_idx : occluding_edges_indices)
    {
        // Changing from 1-based indexing to 0-based!
        occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][0] - 1);
        occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][1] - 1);
    }
    // Remove duplicate vertex id's (std::unique works only on sorted sequences):
    std::sort(begin(occluding_vertices), end(occluding_vertices));
    occluding_vertices.erase(std::unique(begin(occluding_vertices), end(occluding_vertices)),
                             end(occluding_vertices));

    if (perform_self_occlusion_check)
    {
        // Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
        // We have to know the projection type, and then use different vector directions depending on
        // the projection type:
        using render::detail::RayDirection;
        RayDirection ray_direction_type;
        if (projection_type == render::ProjectionType::Orthographic)
        {
            ray_direction_type = RayDirection::Parallel;
        } else
        {
            ray_direction_type = RayDirection::TowardsOrigin;
        }
        std::vector<bool> visibility;
        for (auto vertex_idx : occluding_vertices)
        {
            const bool visible = render::is_vertex_visible(rotated_vertices[vertex_idx], rotated_vertices,
                                                           mesh.tvi, ray_direction_type);
            visibility.push_back(visible);
        }

        // Remove vertices from occluding boundary list that are not visible:
        std::vector<int> final_vertex_ids;
        for (int i = 0; i < occluding_vertices.size(); ++i)
        {
            if (visibility[i] == true)
            {
                final_vertex_ids.push_back(occluding_vertices[i]);
            }
        }
        return final_vertex_ids;
    } else
    {
        return occluding_vertices;
    }
};

/** A simple vector-of-vectors adaptor for nanoflann, without duplicating the storage.
 *  The i'th vector represents a point in the state space.
 *
 * This adaptor is from the nanoflann examples and shows how to use it with these types of containers:
 *   typedef std::vector<std::vector<double> > my_vector_of_vectors_t;
 *   typedef std::vector<Eigen::VectorXd> my_vector_of_vectors_t;
 *
 * It works with any type inside the vector that has operator[] defined to access
 * its elements, as well as a ::size() operator to return its number of dimensions.
 * Eigen::VectorX is one of them. cv::Point is not (no [] and no size()), glm is also
 * not (no size()).
 *
 *  \tparam DIM If set to >0, it specifies a compile-time fixed dimensionality for the points in the data set, allowing more compiler optimizations.
 *  \tparam num_t The type of the point coordinates (typically, double or float).
 *  \tparam Distance The distance metric to use: nanoflann::metric_L1, nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc.
 *  \tparam IndexType The type for indices in the KD-tree index (typically, size_t of int)
 */
template <class VectorOfVectorsType, typename num_t = double, int DIM = -1,
          class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
    typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance> self_t;
    typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType> index_t;

    index_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

    /// Constructor: takes a const ref to the vector of vectors object with the data points
    // Make sure the data is kept alive while the kd-tree is in use.
    KDTreeVectorOfVectorsAdaptor(const int dimensionality, const VectorOfVectorsType& mat,
                                 const int leaf_max_size = 10)
        : m_data(mat)
    {
        assert(mat.size() != 0 && mat[0].size() != 0);
        const size_t dims = mat[0].size();
        if (DIM > 0 && static_cast<int>(dims) != DIM)
            throw std::runtime_error("Data set dimensionality does not match the 'DIM' template argument");
        index = new index_t(dims, *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }

    ~KDTreeVectorOfVectorsAdaptor() { delete index; }

    const VectorOfVectorsType& m_data;

    /** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
     *  Note that this is a short-cut method for index->findNeighbors().
     *  The user can also call index->... methods as desired.
     * \note nChecks_IGNORED is ignored but kept for compatibility with the original FLANN interface.
     */
    inline void query(const num_t* query_point, const size_t num_closest, IndexType* out_indices,
                      num_t* out_distances_sq, const int nChecks_IGNORED = 10) const
    {
        nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t& derived() const { return *this; }
    self_t& derived() { return *this; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return m_data.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in
    // the class:
    inline num_t kdtree_distance(const num_t* p1, const size_t idx_p2, size_t size) const
    {
        num_t s = 0;
        for (size_t i = 0; i < size; i++)
        {
            const num_t d = p1[i] - m_data[idx_p2][i];
            s += d * d;
        }
        return s;
    }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, int dim) const
    {
        return m_data[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

    /** @} */
};

/**
 * @brief For a given list of 2D edge points, find corresponding 3D vertex IDs.
 *
 * This algorithm first computes the 3D mesh's occluding boundary vertices under
 * the given pose. Then, for each 2D image edge point given, it searches for the
 * closest 3D edge vertex (projected to 2D). Correspondences lying further away
 * than \c distance_threshold (times a scale-factor) are discarded.
 * It returns a list of the remaining image edge points and their corresponding
 * 3D vertex ID.
 *
 * The given \c rendering_parameters camery_type must be CameraType::Orthographic.
 *
 * The units of \c distance_threshold are somewhat complicated. The function
 * uses squared distances, and the \c distance_threshold is further multiplied
 * with a face-size and image resolution dependent scale factor.
 * It's reasonable to use correspondences that are 10 to 15 pixels away on a
 * 1280x720 image with s=0.93. This would be a distance_threshold of around 200.
 * 64 might be a conservative default.
 *
 * @param[in] mesh The 3D mesh.
 * @param[in] edge_topology The mesh's edge topology (used for fast computation).
 * @param[in] rendering_parameters Rendering (pose) parameters of the mesh.
 * @param[in] image_edges A list of points that are edges.
 * @param[in] distance_threshold All correspondences below this threshold.
 * @param[in] perform_self_occlusion_check Check the computed boundary vertices for self-occlusion and remove these.
 * @return A pair consisting of the used image edge points and their associated 3D vertex index.
 */
inline std::pair<std::vector<Eigen::Vector2f>, std::vector<int>> find_occluding_edge_correspondences(
    const core::Mesh& mesh, const morphablemodel::EdgeTopology& edge_topology,
    const fitting::RenderingParameters& rendering_parameters, const std::vector<Eigen::Vector2f>& image_edges,
    float distance_threshold = 64.0f, bool perform_self_occlusion_check = true)
{
    assert(rendering_parameters.get_camera_type() == fitting::CameraType::Orthographic);
    using Eigen::Vector2f;
    using std::vector;

    // If there are no image_edges given, there's no point in computing anything:
    if (image_edges.empty())
    {
        return {};
    }

    // Compute vertices that lye on occluding boundaries:
    const render::ProjectionType projection_type = [&rendering_parameters]() {
        if (rendering_parameters.get_camera_type() == CameraType::Orthographic)
        {
            return render::ProjectionType::Orthographic;
        } else
        {
            return render::ProjectionType::Perspective;
        }
    }();
    const auto occluding_vertices =
        occluding_boundary_vertices(mesh, edge_topology, glm::mat4x4(rendering_parameters.get_rotation()),
                                    projection_type, perform_self_occlusion_check);

    // Project these occluding boundary vertices from 3D to 2D:
    vector<Vector2f> model_edges_projected;
    for (const auto& v : occluding_vertices)
    {
        const auto p =
            glm::project({mesh.vertices[v][0], mesh.vertices[v][1], mesh.vertices[v][2]},
                         rendering_parameters.get_modelview(), rendering_parameters.get_projection(),
                         fitting::get_opencv_viewport(rendering_parameters.get_screen_width(),
                                                      rendering_parameters.get_screen_height()));
        model_edges_projected.push_back({p.x, p.y});
    }

    // Find edge correspondences:
    // Build a kd-tree and use nearest neighbour search:
    using kd_tree_t = KDTreeVectorOfVectorsAdaptor<vector<Vector2f>, float, 2>;
    kd_tree_t tree(2, image_edges); // dim, samples, max_leaf
    tree.index->buildIndex();

    vector<std::pair<std::size_t, double>>
        idx_d; // will contain [ index to the 2D edge in 'image_edges', distance (L2^2) ]
    for (const auto& e : model_edges_projected)
    {
        std::size_t idx; // contains the indices into the original 'image_edges' vector
        double dist_sq;  // squared distances
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&idx, &dist_sq);
        tree.index->findNeighbors(resultSet, e.data(), nanoflann::SearchParams(10));
        idx_d.push_back({idx, dist_sq});
    }
    // Filter edge matches:
    // We filter below by discarding all correspondence that are a certain distance apart.
    // We could also (or in addition to) discard the worst 5% of the distances or something like that.

    // Filter and store the image (edge) points with their corresponding vertex id:
    vector<int> vertex_indices;
    vector<Vector2f> image_points;
    assert(occluding_vertices.size() == idx_d.size());
    for (int i = 0; i < occluding_vertices.size(); ++i)
    {
        const auto ortho_scale =
            rendering_parameters.get_screen_width() /
            rendering_parameters.get_frustum().r; // This might be a bit of a hack - we recover the "real"
                                                  // scaling from the SOP estimate
        if (idx_d[i].second <= distance_threshold * ortho_scale) // I think multiplying by the scale is good
                                                                 // here and gives us invariance w.r.t. the
                                                                 // image resolution and face size.
        {
            const auto edge_point = image_edges[idx_d[i].first];
            // Store the found 2D edge point, and the associated vertex id:
            vertex_indices.push_back(occluding_vertices[i]);
            image_points.push_back(edge_point);
        }
    }
    return {image_points, vertex_indices};
};

} /* namespace fitting */
} /* namespace eos */

#endif /* EOS_CLOSEST_EDGE_FITTING_HPP */
