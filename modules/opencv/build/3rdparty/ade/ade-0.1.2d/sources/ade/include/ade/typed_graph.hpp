// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file typed_graph.hpp

#ifndef ADE_TYPED_GRAPH_HPP
#define ADE_TYPED_GRAPH_HPP

#include <cstdint>
#include <unordered_set>
#include <string>

#include <array>

#include "util/assert.hpp"
#include "util/memory_range.hpp"

#include "graph.hpp"
#include "typed_metadata.hpp"

namespace ade
{

template <typename T>
struct MetadataNameTag
{
};

/// User must provide means to get metadata name for each type he want to use
/// in graph metadata.
/// This can be done either via static member function:
/// \code{.cpp}
/// static const char* MyType::name();
/// \endcode
/// or via special function overload:
/// \code{.cpp}
/// const char* getMetadataName(ade::MetadataNameTag<MyType>);
/// \endcode
template<typename T>
auto getMetadataName(MetadataNameTag<T>)
->decltype(T::name())
{
    return T::name();
}

namespace details
{

template<typename T>
auto getMetadataNameImpl()
->decltype(getMetadataName(MetadataNameTag<T>()))
{
    return getMetadataName(MetadataNameTag<T>());
}

template<typename... Types>
void checkUniqueNames()
{
    std::unordered_multiset<std::string> names( { details::getMetadataNameImpl<Types>()... } );
    auto not_unique = std::find_if(names.begin(), names.end(),
                                  [&names](const std::string& n){ return names.count(n) != 1; });

    if (not_unique != names.end()) throw_error(std::logic_error("Name " + *not_unique + " is not unique in graph metadata"));
}

template<typename T, typename... Remaining>
struct InitIdsArray final
{
    inline void operator()(const ade::Graph& gr, util::MemoryRange<ade::details::MetadataId> ids) const
    {
        ADE_ASSERT(ids.size == (1 + sizeof...(Remaining)));
        ids[0] = gr.getMetadataId(getMetadataNameImpl<T>());
        InitIdsArray<Remaining...>()(gr, ids.Slice(1, ids.size - 1));
    }
};

template<typename T>
struct InitIdsArray<T> final
{
    inline void operator()(const ade::Graph& gr, util::MemoryRange<ade::details::MetadataId> ids) const
    {
        ADE_ASSERT(1 == ids.size);
        ids[0] = gr.getMetadataId(getMetadataNameImpl<T>());
    }
};

}

template<typename... Types>
class TypedGraph;

template<typename... Types>
class ConstTypedGraph
{
protected:
    static_assert(sizeof...(Types) > 0, "Type list is empty");

    // Store graph ptr as uintptr_t to avoid const_casts in derived class
    std::uintptr_t m_srcGraph;
    std::array<ade::details::MetadataId, sizeof...(Types)> m_ids;

    const ade::Graph& getCGraph() const
    {
        return *reinterpret_cast<const ade::Graph*>(m_srcGraph);
    }

    void initIds()
    {
        details::InitIdsArray<Types...>()(getCGraph(), util::memory_range(m_ids.data(), m_ids.size()));;
    }


public:
    template<typename... OtherTypes>
    friend class ConstTypedGraph;

    using CMetadataT = ade::TypedMetadata<true, Types...>;

    ConstTypedGraph(const ade::Graph& graph):
        m_srcGraph(reinterpret_cast<std::uintptr_t>(&graph))
    {
        details::checkUniqueNames<Types...>();
        initIds();
    }

    // explicitly define copy-constructor since we modify operator=
    ConstTypedGraph(const ConstTypedGraph<Types...>& other):
        m_srcGraph(other.m_srcGraph)
    {
        details::checkUniqueNames<Types...>();
        initIds();
    }

    template<typename... OtherTypes>
    ConstTypedGraph(const ConstTypedGraph<OtherTypes...>& other):
        m_srcGraph(other.m_srcGraph)
    {
        details::checkUniqueNames<Types...>();
        initIds();
    }

    ConstTypedGraph& operator=(const ConstTypedGraph&) = delete;

    ade::Graph::NodesListCRange nodes() const
    {
        return getCGraph().nodes();
    }

    CMetadataT metadata() const
    {
        return CMetadataT(m_ids, getCGraph().metadata());
    }

    CMetadataT metadata(const ade::NodeHandle& handle) const
    {
        return CMetadataT(m_ids, getCGraph().metadata(handle));
    }

    CMetadataT metadata(const ade::EdgeHandle& handle) const
    {
        return CMetadataT(m_ids, getCGraph().metadata(handle));
    }
};

template<typename... Types>
class TypedGraph final : public ConstTypedGraph<Types...>
{
protected:
    static_assert(sizeof...(Types) > 0, "Type list is empty");

    ade::Graph& getGraph() const
    {
        return *reinterpret_cast<ade::Graph*>(this->m_srcGraph);
    }

public:
    using MetadataT  = ade::TypedMetadata<false, Types...>;

    TypedGraph(ade::Graph& graph):
        ConstTypedGraph<Types...>(graph)
    {
    }

    // explicitly define copy-constructor since we modify operator=
    TypedGraph(TypedGraph& graph):
        ConstTypedGraph<Types...>(graph)
    {
    }

    template<typename... OtherTypes>
    TypedGraph(const TypedGraph<OtherTypes...>& other):
        ConstTypedGraph<Types...>(other)
    {

    }

    TypedGraph& operator=(const TypedGraph&) = delete;

    /// Create new node
    ade::NodeHandle createNode()
    {
        return getGraph().createNode();
    }

    /// Delete node and all connected edges from graph and null all handles pointing to it
    void erase(const ade::NodeHandle& node)
    {
        getGraph().erase(node);
    }

    /// Delete all edges, connected to this node
    void unlink(const ade::NodeHandle& node)
    {
        getGraph().unlink(node);
    }

    /// Delete node and all connected edges from graph and null all handles pointing to it
    void erase(const ade::EdgeHandle& edge)
    {
        getGraph().erase(edge);
    }

    /// Create new edge between src_node and dst_node
    ade::EdgeHandle link(const ade::NodeHandle& src_node, const ade::NodeHandle& dst_node)
    {
        return getGraph().link(src_node, dst_node);
    }

    /// Change src_edge destination node to dst_node
    /// noop if src_edge destination node is already dst_node
    /// returns src_edge
    ade::EdgeHandle link(const ade::EdgeHandle& src_edge, const ade::NodeHandle& dst_node)
    {
        return getGraph().link(src_edge, dst_node);
    }

    /// Change dst_edge source node to src_node
    /// noop if dst_edge source node is already src_node
    /// returns dst_edge
    ade::EdgeHandle link(const ade::NodeHandle& src_node, const ade::EdgeHandle& dst_edge)
    {
        return getGraph().link(src_node, dst_edge);
    }

    using ConstTypedGraph<Types...>::nodes;
    using ConstTypedGraph<Types...>::metadata;

    ade::Graph::NodesListRange nodes()
    {
        return getGraph().nodes();
    }

    MetadataT metadata()
    {
        return MetadataT(this->m_ids, getGraph().metadata());
    }

    MetadataT metadata(const ade::NodeHandle& handle)
    {
        return MetadataT(this->m_ids, getGraph().metadata(handle));
    }

    MetadataT metadata(const ade::EdgeHandle& handle)
    {
        return MetadataT(this->m_ids, getGraph().metadata(handle));
    }
};

}

#endif // ADE_TYPED_GRAPH_HPP
