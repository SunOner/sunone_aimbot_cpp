// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file edge.hpp

#ifndef ADE_EDGE_HPP
#define ADE_EDGE_HPP

#include <memory>

#include "handle.hpp"

namespace ade
{

class Graph;
class Node;
class Edge;
using EdgeHandle = Handle<Edge>;
using NodeHandle = Handle<Node>;

class Edge final : public std::enable_shared_from_this<Edge>
{
public:
    NodeHandle srcNode() const;
    NodeHandle dstNode() const;
private:
    friend class Graph;
    friend class Node;

    Edge(Node* prev, Node* next);
    ~Edge();
    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;

    Graph* getParent() const;

    void unlink();
    void resetPrevNode(Node* newNode);
    void resetNextNode(Node* newNode);

    Node* m_prevNode = nullptr;
    Node* m_nextNode = nullptr;
};

}

#endif // ADE_EDGE_HPP
