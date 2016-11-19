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

#ifndef CLOSESTEDGEFITTING_HPP_
#define CLOSESTEDGEFITTING_HPP_

#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/render/Mesh.hpp"
#include "eos/render/utils.hpp"

#include "nanoflann.hpp"

#include "glm/common.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include "boost/optional.hpp"

#include <vector>
#include <algorithm>
#include <utility>

namespace eos {
	namespace fitting {

/**
 * @brief Computes the intersection of the given ray with the given triangle.
 *
 * Uses the Möller-Trumbore algorithm algorithm "Fast Minimum Storage
 * Ray/Triangle Intersection". Independent implementation, inspired by:
 * http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
 * The default eps (1e-6f) is from the paper.
 * When culling is on, rays intersecting triangles from the back will be discarded -
 * otherwise, the triangles normal direction w.r.t. the ray direction is just ignored.
 *
 * Note: The use of optional might turn out as a performance problem, as this
 * function is called loads of time - how costly is it to construct a boost::none optional?
 *
 * @param[in] ray_origin Ray origin.
 * @param[in] ray_direction Ray direction.
 * @param[in] v0 First vertex of a triangle.
 * @param[in] v1 Second vertex of a triangle.
 * @param[in] v2 Third vertex of a triangle.
 * @param[in] enable_backculling When culling is on, rays intersecting triangles from the back will be discarded.
 * @return Whether the ray intersects the triangle, and if yes, including the distance.
 */
std::pair<bool, boost::optional<float>> ray_triangle_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_direction, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, bool enable_backculling)
{
	using glm::vec3;
	const float epsilon = 1e-6f;

	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;

	vec3 pvec = glm::cross(ray_direction, v0v2);

	float det = glm::dot(v0v1, pvec);
	if (enable_backculling)
	{
		// If det is negative, the triangle is back-facing.
		// If det is close to 0, the ray misses the triangle.
		if (det < epsilon)
			return { false, boost::none };
	}
	else {
		// If det is close to 0, the ray and triangle are parallel.
		if (std::abs(det) < epsilon)
			return { false, boost::none };
	}
	float inv_det = 1 / det;

	vec3 tvec = ray_origin - v0;
	auto u = glm::dot(tvec, pvec) * inv_det;
	if (u < 0 || u > 1)
		return { false, boost::none };

	vec3 qvec = glm::cross(tvec, v0v1);
	auto v = glm::dot(ray_direction, qvec) * inv_det;
	if (v < 0 || u + v > 1)
		return { false, boost::none };

	auto t = glm::dot(v0v2, qvec) * inv_det;

	return { true, t };
};

/**
 * @brief Computes the vertices that lie on occluding boundaries, given a particular pose.
 *
 * This algorithm computes the edges that lie on occluding boundaries of the mesh.
 * It performs a visibility text of each vertex, and returns a list of the (unique)
 * vertices that make the boundary edges.
 * An edge is defined as the line whose two adjacent faces normals flip the sign.
 *
 * @param[in] mesh The mesh to use.
 * @param[in] edge_topology The edge topology of the given mesh.
 * @param[in] R The rotation (pose) under which the occluding boundaries should be computed.
 * @return A vector with unique vertex id's making up the edges.
 */
std::vector<int> occluding_boundary_vertices(const eos::render::Mesh& mesh, const morphablemodel::EdgeTopology& edge_topology, glm::mat4x4 R)
{
	// Rotate the mesh:
	std::vector<glm::vec4> rotated_vertices;
	std::for_each(begin(mesh.vertices), end(mesh.vertices), [&rotated_vertices, &R](auto&& v) { rotated_vertices.push_back(R * v); });

	// Compute the face normals of the rotated mesh:
	std::vector<glm::vec3> facenormals;
	for (auto&& f : mesh.tvi) { // for each face (triangle):
		auto n = render::compute_face_normal(rotated_vertices[f[0]], rotated_vertices[f[1]], rotated_vertices[f[2]]);
		facenormals.push_back(n);
	}

	// Find occluding edges:
	std::vector<int> occluding_edges_indices;
	for (int edge_idx = 0; edge_idx < edge_topology.adjacent_faces.size(); ++edge_idx) // For each edge... Ef contains the indices of the two adjacent faces
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
	for (auto&& edge_idx : occluding_edges_indices)
	{
		// Changing from 1-based indexing to 0-based!
		occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][0] - 1);
		occluding_vertices.push_back(edge_topology.adjacent_vertices[edge_idx][1] - 1);
	}
	// Remove duplicate vertex id's (std::unique works only on sorted sequences):
	std::sort(begin(occluding_vertices), end(occluding_vertices));
	occluding_vertices.erase(std::unique(begin(occluding_vertices), end(occluding_vertices)), end(occluding_vertices));

	// Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
	std::vector<bool> visibility;
	for (const auto& vertex_idx : occluding_vertices)
	{
		bool visible = true;
		// For every tri of the rotated mesh:
		for (auto&& tri : mesh.tvi)
		{
			auto& v0 = rotated_vertices[tri[0]];
			auto& v1 = rotated_vertices[tri[1]];
			auto& v2 = rotated_vertices[tri[2]];

			glm::vec3 ray_origin = rotated_vertices[vertex_idx];
			glm::vec3 ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
			auto intersect = ray_triangle_intersect(ray_origin, ray_direction, glm::vec3(v0), glm::vec3(v1), glm::vec3(v2), false);
			// first is bool intersect, second is the distance t
			if (intersect.first == true)
			{
				// We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the intersection:
				// Check if in front or behind?
				if (intersect.second.get() <= 1e-4)
				{
					continue; // the intersection is behind the vertex, we don't care about it
				}
				// Otherwise, we've hit a genuine triangle, and the vertex is not visible:
				visible = false;
				break;
			}
		}
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
template <class VectorOfVectorsType, typename num_t = double, int DIM = -1, class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
	typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance> self_t;
	typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
	typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>  index_t;

	index_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

	/// Constructor: takes a const ref to the vector of vectors object with the data points
	// Make sure the data is kept alive while the kd-tree is in use.
	KDTreeVectorOfVectorsAdaptor(const int dimensionality, const VectorOfVectorsType &mat, const int leaf_max_size = 10) : m_data(mat)
	{
		assert(mat.size() != 0 && mat[0].size() != 0);
		const size_t dims = mat[0].size();
		if (DIM>0 && static_cast<int>(dims) != DIM)
			throw std::runtime_error("Data set dimensionality does not match the 'DIM' template argument");
		index = new index_t(dims, *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
		index->buildIndex();
	}

	~KDTreeVectorOfVectorsAdaptor() {
		delete index;
	}

	const VectorOfVectorsType &m_data;

	/** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
	*  Note that this is a short-cut method for index->findNeighbors().
	*  The user can also call index->... methods as desired.
	* \note nChecks_IGNORED is ignored but kept for compatibility with the original FLANN interface.
	*/
	inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq, const int nChecks_IGNORED = 10) const
	{
		nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
		resultSet.init(out_indices, out_distances_sq);
		index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
	}

	/** @name Interface expected by KDTreeSingleIndexAdaptor
	* @{ */

	const self_t & derived() const {
		return *this;
	}
	self_t & derived() {
		return *this;
	}

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const {
		return m_data.size();
	}

	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline num_t kdtree_distance(const num_t *p1, const size_t idx_p2, size_t size) const
	{
		num_t s = 0;
		for (size_t i = 0; i<size; i++) {
			const num_t d = p1[i] - m_data[idx_p2][i];
			s += d*d;
		}
		return s;
	}

	// Returns the dim'th component of the idx'th point in the class:
	inline num_t kdtree_get_pt(const size_t idx, int dim) const {
		return m_data[idx][dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX & /*bb*/) const {
		return false;
	}

	/** @} */

};

	} /* namespace fitting */
} /* namespace eos */

#endif /* CLOSESTEDGEFITTING_HPP_ */
