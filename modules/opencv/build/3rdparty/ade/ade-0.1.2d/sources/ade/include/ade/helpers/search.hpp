// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file search.hpp

#ifndef ADE_SEARCH_HPP
#define ADE_SEARCH_HPP

#include <unordered_set>
#include <unordered_map>

#include "ade/node.hpp"

#include "ade/util/algorithm.hpp"
#include "ade/util/assert.hpp"
#include "ade/util/func_ref.hpp"

namespace ade
{
namespace traverse
{
using traverse_func_type
    = util::func_ref<void (const Node&,
                           util::func_ref<void (const NodeHandle&)>)>;
inline void forward(const Node& node,
                    util::func_ref<void (const NodeHandle&)> visitor)
{
    for (auto&& next : node.outNodes())
    {
        visitor(next);
    }
}
inline void backward(const Node& node,
                     util::func_ref<void (const NodeHandle&)> visitor)
{
    for (auto&& next : node.inNodes())
    {
        visitor(next);
    }
}
} // namespace ade::traverse

/// Depth first search through nodes
///
/// @param node - Start node, must not be null
/// @param visitor - Functor called for each found node,
/// must return true to continue search
/// @param direction - Traverse direction, this functor will be called for
/// current node and it must call its argument for each node we want to
/// visit next.
void dfs(const NodeHandle& node,
         util::func_ref<bool (const NodeHandle&)> visitor,
         traverse::traverse_func_type direction = traverse::forward);

namespace details
{
struct TransitiveClosureHelper
{
    using CacheT =
    std::unordered_map<NodeHandle,
                       std::unordered_set<NodeHandle, HandleHasher<Node>>,
                       HandleHasher<Node>>;
    void operator()(CacheT& cache,
                    const NodeHandle& node,
                    traverse::traverse_func_type direction) const;
};
}

/// Computes transitive closure for graph
/// https://en.wikipedia.org/wiki/Transitive_closure#In_graph_theory
///
/// @param nodes - List of nodes to check
/// @param visitor - functor which will be called for pairs of nodes,
/// first parameted is source node from provided list and
/// second parameter is node reachable from this source node
/// @param direction - Traverse direction, this functor will be called for
/// current node and it must call its argument for each node we want to
/// visit next.
template<typename Nodes, typename Visitor>
void transitiveClosure(
        Nodes&& nodes,
        Visitor&& visitor,
        traverse::traverse_func_type direction = traverse::forward)
{
    using Helper = details::TransitiveClosureHelper;
    Helper::CacheT visited;
    for (auto node : nodes)
    {
        ADE_ASSERT(nullptr != node);
        if (!util::contains(visited, node))
        {
            Helper()(visited, node, direction);
        }
        ADE_ASSERT(util::contains(visited, node));
        for (auto nextNode : visited[node])
        {
            visitor(node, nextNode);
        }
    }
}

}

#endif // ADE_SEARCH_HPP
