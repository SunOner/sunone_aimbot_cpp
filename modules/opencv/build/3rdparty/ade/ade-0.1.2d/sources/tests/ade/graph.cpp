// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/node.hpp>
#include <ade/edge.hpp>
#include <ade/graph_listener.hpp>

using namespace ade;

TEST(Graph, Simple)
{
    Graph gr;
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();
    ASSERT_NE(nullptr, node1);
    ASSERT_NE(nullptr, node2);
    ASSERT_NE(nullptr, node3);
    ASSERT_EQ(3, gr.nodes().size());
    NodeHandle arr[] = {node1, node2, node3};
    for (NodeHandle h: gr.nodes())
    {
        ASSERT_NE(nullptr, h);
        auto it = std::find(std::begin(arr), std::end(arr), h);
        ASSERT_NE(std::end(arr), it);
    }
    gr.erase(node2);
    ASSERT_EQ(nullptr, node2);
    ASSERT_EQ(2, gr.nodes().size());
}

TEST(Graph, EraseMiddleNode)
{
    Graph gr;

    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();

    gr.link(node1, node2);
    gr.link(node2, node3);

    gr.erase(node2);

    EXPECT_EQ(2, gr.nodes().size());
}

namespace
{
class TestListener : public ade::IGraphListener
{
public:
    std::function<void(const ade::NodeHandle&)> createNode;
    std::function<void(const ade::NodeHandle&)> destroyNode;

    std::function<void(const ade::EdgeHandle&)> createEdge;
    std::function<void(const ade::EdgeHandle&)> destroyEdge;
    std::function<void(const ade::EdgeHandle&,const NodeHandle&,const NodeHandle&)> relinkEdge;

    virtual void nodeCreated(const Graph& /*graph*/, const NodeHandle& node) override
    {
        createNode(node);
    }
    virtual void nodeAboutToBeDestroyed(const Graph& /*graph*/, const NodeHandle& node) override
    {
        destroyNode(node);
    }

    virtual void edgeCreated(const Graph& /*graph*/, const EdgeHandle& edge) override
    {
        createEdge(edge);
    }
    virtual void edgeAboutToBeDestroyed(const Graph& /*graph*/, const EdgeHandle& edge) override
    {
        destroyEdge(edge);
    }
    virtual void edgeAboutToBeRelinked(const Graph& /*graph*/,
                                       const EdgeHandle& edge,
                                       const NodeHandle& newSrcNode,
                                       const NodeHandle& newDstNode) override
    {
        relinkEdge(edge, newSrcNode, newDstNode);
    }

    void reset()
    {
        createNode = nullptr;
        destroyNode = nullptr;

        createEdge = nullptr;
        destroyEdge = nullptr;

        relinkEdge = nullptr;
    }
};
}
TEST(Graph, Listener)
{
    NodeHandle tempNode;
    TestListener listener;
    bool called = false;

    Graph gr;
    gr.setListener(&listener);
    ASSERT_EQ(gr.getListener(), &listener);

    listener.reset();
    listener.createNode = [&](const ade::NodeHandle& node)
    {
        tempNode = node;
    };

    tempNode = nullptr;
    auto node1 = gr.createNode();
    EXPECT_EQ(node1, tempNode);

    tempNode = nullptr;
    auto node2 = gr.createNode();
    EXPECT_EQ(node2, tempNode);

    tempNode = nullptr;
    auto node3 = gr.createNode();
    EXPECT_EQ(node3, tempNode);


    listener.reset();
    called = false;
    listener.createEdge = [&](const ade::EdgeHandle& edge)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(node1, edge->srcNode());
        EXPECT_EQ(node2, edge->dstNode());
    };
    auto edge1 = gr.link(node1, node2);
    EXPECT_TRUE(called);

    listener.reset();
    called = false;
    listener.createEdge = [&](const ade::EdgeHandle& edge)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(node2, edge->srcNode());
        EXPECT_EQ(node3, edge->dstNode());
    };
    auto edge2 = gr.link(node2, node3);
    EXPECT_TRUE(called);

    listener.reset();
    called = false;
    listener.relinkEdge = [&](const EdgeHandle& edge,
                              const NodeHandle& newSrcNode,
                              const NodeHandle& newDstNode)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(edge1, edge);
        EXPECT_EQ(node1, newSrcNode);
        EXPECT_EQ(node3, newDstNode);
    };
    gr.link(edge1, node3);
    EXPECT_TRUE(called);

    listener.reset();
    called = false;
    listener.relinkEdge = [&](const EdgeHandle& edge,
                              const NodeHandle& newSrcNode,
                              const NodeHandle& newDstNode)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(edge2, edge);
        EXPECT_EQ(node1, newSrcNode);
        EXPECT_EQ(node3, newDstNode);
    };
    gr.link(node1, edge2);
    EXPECT_TRUE(called);


    listener.reset();
    called = false;
    listener.destroyEdge = [&](const EdgeHandle& edge)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(edge1, edge);
    };
    gr.erase(edge1);
    EXPECT_TRUE(called);

    listener.reset();
    called = false;
    listener.destroyEdge = [&](const EdgeHandle& edge)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(edge2, edge);
    };
    gr.erase(edge2);
    EXPECT_TRUE(called);


    listener.reset();
    called = false;
    listener.destroyNode = [&](const NodeHandle& node)
    {
        EXPECT_FALSE(called);
        called = true;
        EXPECT_EQ(node2, node);
    };
    gr.erase(node2);
    EXPECT_TRUE(called);
    gr.setListener(nullptr);
}
