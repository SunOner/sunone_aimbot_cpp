// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/graph.hpp"

#include <algorithm>

#include "ade/util/assert.hpp"
#include "ade/util/algorithm.hpp"

#include "ade/edge.hpp"
#include "ade/node.hpp"
#include "ade/typed_metadata.hpp"

#include "ade/graph_listener.hpp"

namespace ade
{

Graph::Graph()
{

}

Graph::~Graph()
{

}

NodeHandle Graph::createNode()
{
    NodePtr node(new Node(this), ElemDeleter{});
    NodeHandle ret(node);
    m_nodes.emplace_back(std::move(node));
    if (nullptr != m_listener)
    {
        m_listener->nodeCreated(*this, ret);
    }
    return ret;
}

void Graph::erase(const NodeHandle& node)
{
    ADE_ASSERT(nullptr != node);
    ADE_ASSERT(node->getParent() == this);
    removeNode(node.get());
}

void Graph::unlink(const NodeHandle& node)
{
    ADE_ASSERT(nullptr != node);
    ADE_ASSERT(node->getParent() == this);
    node->unlink();
}

void Graph::erase(const EdgeHandle& edge)
{
    ADE_ASSERT(nullptr != edge);
    ADE_ASSERT(edge->getParent() == this);
    removeEdge(edge.get());
}

EdgeHandle Graph::link(const NodeHandle& src_node, const NodeHandle& dst_node)
{
    ADE_ASSERT(nullptr != src_node);
    ADE_ASSERT(nullptr != dst_node);
    ADE_ASSERT(src_node->getParent() == dst_node->getParent());
    ADE_ASSERT(src_node->getParent() == this);
    return createEdge(src_node.get(), dst_node.get());
}

EdgeHandle Graph::link(const EdgeHandle& src_edge, const NodeHandle& dst_node)
{
    ADE_ASSERT(nullptr != src_edge);
    ADE_ASSERT(nullptr != dst_node);
    Edge* edge = src_edge.get();
    Node* node = dst_node.get();
    ADE_ASSERT_STRONG(edge != nullptr);
    ADE_ASSERT_STRONG(node != nullptr);
    if (nullptr != m_listener)
    {
        m_listener->edgeAboutToBeRelinked(*this, src_edge, edge->srcNode(), dst_node);
    }
    edge->resetNextNode(node);

    return src_edge;
}

EdgeHandle Graph::link(const NodeHandle& src_node, const EdgeHandle& dst_edge)
{
    ADE_ASSERT(nullptr != src_node);
    ADE_ASSERT(nullptr != dst_edge);
    Edge* edge = dst_edge.get();
    Node* node = src_node.get();
    ADE_ASSERT_STRONG(edge != nullptr);
    ADE_ASSERT_STRONG(node != nullptr);
    if (nullptr != m_listener)
    {
        m_listener->edgeAboutToBeRelinked(*this, dst_edge, src_node, edge->dstNode());
    }
    edge->resetPrevNode(node);

    return dst_edge;
}

Graph::NodesListRange Graph::nodes()
{
    return util::map<HandleMapper>(util::toRange(m_nodes));
}

Graph::NodesListCRange Graph::nodes() const
{
    return util::map<HandleMapper>(util::toRange(m_nodes));
}

details::MetadataId Graph::getMetadataId(const std::string& name) const
{
    return details::MetadataId(m_ids[name].p.get());
}

details::Metadata& Graph::metadata()
{
    return geMetadataImpl(nullptr);
}

const details::Metadata& Graph::metadata() const
{
    return geMetadataImpl(nullptr);
}

details::Metadata& Graph::metadata(const NodeHandle handle)
{
    ADE_ASSERT(nullptr != handle);
    return geMetadataImpl(handle.get());
}

const details::Metadata& Graph::metadata(const NodeHandle handle) const
{
    ADE_ASSERT(nullptr != handle);
    return geMetadataImpl(handle.get());
}

details::Metadata& Graph::metadata(const EdgeHandle handle)
{
    ADE_ASSERT(nullptr != handle);
    return geMetadataImpl(handle.get());
}

const details::Metadata& Graph::metadata(const EdgeHandle handle) const
{
    ADE_ASSERT(nullptr != handle);
    return geMetadataImpl(handle.get());
}

void Graph::setListener(IGraphListener* listener)
{
    m_listener = listener;
}

IGraphListener* Graph::getListener() const
{
    return m_listener;
}

EdgeHandle Graph::createEdge(Node* src_node, Node* dst_node)
{
    ADE_ASSERT(nullptr != src_node);
    ADE_ASSERT(nullptr != dst_node);
    ADE_ASSERT(this == src_node->getParent());
    ADE_ASSERT(this == dst_node->getParent());
    EdgePtr edge(new Edge(src_node, dst_node), ElemDeleter{});
    EdgePtr ret(edge);
    m_edges.emplace_back(edge);
    if (nullptr != m_listener)
    {
        m_listener->edgeCreated(*this, ret);
    }
    return ret;
}

void Graph::removeNode(Node* node)
{
    ADE_ASSERT(nullptr != node);
    if (nullptr != m_listener)
    {
        m_listener->nodeAboutToBeDestroyed(*this, node->shared_from_this());
    }
    m_metadata.erase(node);
    auto it = std::find_if(m_nodes.begin(), m_nodes.end(), [node](const NodePtr& n)
    {
        return n.get() == node;
    });
    //TODO: get rid of linear search
    ADE_ASSERT(m_nodes.end() != it);
    util::unstable_erase(m_nodes, it);
}

void Graph::removeEdge(Edge* edge)
{
    ADE_ASSERT(nullptr != edge);
    if (nullptr != m_listener)
    {
        m_listener->edgeAboutToBeDestroyed(*this, edge->shared_from_this());
    }
    m_metadata.erase(edge);
    auto it = std::find_if(m_edges.begin(), m_edges.end(), [edge](const EdgePtr& e)
    {
        return e.get() == edge;
    });
    //TODO: get rid of linear search
    ADE_ASSERT(m_edges.end() != it);
    util::unstable_erase(m_edges, it);
}

details::Metadata& Graph::geMetadataImpl(void* handle) const
{
    auto it = m_metadata.find(handle);
    if (m_metadata.end() != it)
    {
        return *it->second;
    }
    std::unique_ptr<details::Metadata> meta(new details::Metadata);
    auto res = m_metadata.insert(std::make_pair(handle, std::move(meta)));
    ADE_ASSERT(res.second);
    return *(res.first->second);
}

void Graph::ElemDeleter::operator()(Node* node) const
{
    delete node;
}

void Graph::ElemDeleter::operator()(Edge* edge) const
{
    delete edge;
}

}
