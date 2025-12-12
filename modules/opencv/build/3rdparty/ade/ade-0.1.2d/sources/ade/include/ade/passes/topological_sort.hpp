// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file topological_sort.hpp

#ifndef ADE_TOPOLOGICAL_SORT_HPP
#define ADE_TOPOLOGICAL_SORT_HPP

#include <vector>
#include "utility"

#include "ade/node.hpp"

#include "ade/typed_graph.hpp"
#include "ade/passes/pass_base.hpp"

#include "ade/util/range.hpp"
#include "ade/util/filter_range.hpp"

namespace ade
{
namespace passes
{

struct TopologicalSortData final
{
    struct NodesFilter final
    {
        bool operator()(const ade::NodeHandle& node) const
        {
            return nullptr != node;
        }
    };

    using NodesList = std::vector<NodeHandle>;
    using NodesRange = util::FilterRange<util::IterRange<NodesList::const_iterator>, NodesFilter>;

    TopologicalSortData(const NodesList& nodes_):
        m_nodes(nodes_) {}

    TopologicalSortData(NodesList&& nodes_):
        m_nodes(std::move(nodes_)) {}

    NodesRange nodes() const
    {
        return util::filter<NodesFilter>(util::toRange(m_nodes));
    }

    static const char* name();

private:
    NodesList m_nodes;
};

struct TopologicalSort final
{
    void operator()(TypedPassContext<TopologicalSortData> context) const;
    static const char* name();
};

struct LazyTopologicalSortChecker final
{
    bool nodeCreated(const Graph& graph, const NodeHandle& node);
    bool nodeAboutToBeDestroyed(const Graph& graph, const NodeHandle& node);

    bool edgeCreated(const Graph&, const EdgeHandle& edge);
    bool edgeAboutToBeDestroyed(const Graph& graph, const EdgeHandle& edge);
    bool edgeAboutToBeRelinked(const Graph& graph,
                               const EdgeHandle& edge,
                               const NodeHandle& newSrcNode,
                               const NodeHandle& newDstNode);
};

}
}

#endif // ADE_TOPOLOGICAL_SORT_HPP
