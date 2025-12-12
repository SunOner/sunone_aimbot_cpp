// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/node.hpp>
#include <ade/edge.hpp>

using namespace ade;

TEST(Node, Link)
{
    Graph gr;
    auto node1 = gr.createNode();
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(0, node1->outEdges().size());
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();

    auto edge1 = gr.link(node1, node2);
    ASSERT_NE(nullptr, edge1);
    ASSERT_EQ(node1, edge1->srcNode());
    ASSERT_EQ(node2, edge1->dstNode());
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(1, node1->outEdges().size());
    ASSERT_EQ(1, node2->inEdges().size());
    ASSERT_EQ(0, node2->outEdges().size());

    ASSERT_EQ(0, node1->inNodes().size());
    ASSERT_EQ(1, node1->outNodes().size());
    ASSERT_EQ(1, node2->inNodes().size());
    ASSERT_EQ(0, node2->outNodes().size());

    auto edge2 = gr.link(node2, node3);
    ASSERT_NE(nullptr, edge2);
    ASSERT_EQ(1, node2->inEdges().size());
    ASSERT_EQ(1, node2->outEdges().size());
    ASSERT_EQ(1, node3->inEdges().size());
    ASSERT_EQ(0, node3->outEdges().size());

    ASSERT_EQ(1, node2->inNodes().size());
    ASSERT_EQ(1, node2->outNodes().size());
    ASSERT_EQ(1, node3->inNodes().size());
    ASSERT_EQ(0, node3->outNodes().size());

    gr.erase(node3);
    ASSERT_EQ(nullptr, edge2);
    ASSERT_EQ(1, node2->inEdges().size());
    ASSERT_EQ(0, node2->outEdges().size());

    ASSERT_EQ(1, node2->inNodes().size());
    ASSERT_EQ(0, node2->outNodes().size());

    gr.erase(edge1);
    ASSERT_EQ(nullptr, edge1);
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(0, node1->outEdges().size());
    ASSERT_EQ(0, node2->inEdges().size());
    ASSERT_EQ(0, node2->outEdges().size());

    ASSERT_EQ(0, node1->inNodes().size());
    ASSERT_EQ(0, node1->outNodes().size());
    ASSERT_EQ(0, node2->inNodes().size());
    ASSERT_EQ(0, node2->outNodes().size());
}

TEST(Node, ReLink)
{
    Graph gr;
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();

    auto edge = gr.link(node1, node2);
    ASSERT_NE(nullptr, edge);
    ASSERT_EQ(node1, edge->srcNode());
    ASSERT_EQ(node2, edge->dstNode());
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(1, node1->outEdges().size());
    ASSERT_EQ(1, node2->inEdges().size());
    ASSERT_EQ(0, node2->outEdges().size());
    ASSERT_EQ(0, node3->inEdges().size());
    ASSERT_EQ(0, node3->outEdges().size());

    edge = gr.link(edge, node3);
    ASSERT_NE(nullptr, edge);
    ASSERT_EQ(node1, edge->srcNode());
    ASSERT_EQ(node3, edge->dstNode());
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(1, node1->outEdges().size());
    ASSERT_EQ(0, node2->inEdges().size());
    ASSERT_EQ(0, node2->outEdges().size());
    ASSERT_EQ(1, node3->inEdges().size());
    ASSERT_EQ(0, node3->outEdges().size());

    edge = gr.link(node2, edge);
    ASSERT_NE(nullptr, edge);
    ASSERT_EQ(node2, edge->srcNode());
    ASSERT_EQ(node3, edge->dstNode());
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(0, node1->outEdges().size());
    ASSERT_EQ(0, node2->inEdges().size());
    ASSERT_EQ(1, node2->outEdges().size());
    ASSERT_EQ(1, node3->inEdges().size());
    ASSERT_EQ(0, node3->outEdges().size());

    gr.erase(edge);
    ASSERT_EQ(nullptr, edge);
    ASSERT_EQ(0, node1->inEdges().size());
    ASSERT_EQ(0, node1->outEdges().size());
    ASSERT_EQ(0, node2->inEdges().size());
    ASSERT_EQ(0, node2->outEdges().size());
    ASSERT_EQ(0, node3->inEdges().size());
    ASSERT_EQ(0, node3->outEdges().size());
}

TEST(Node, Edges)
{
    Graph gr;
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();
    auto node4 = gr.createNode();

    EdgeHandle edges[] = { gr.link(node1, node2),
                           gr.link(node1, node3),
                           gr.link(node1, node4) };
    ASSERT_EQ(3, node1->outEdges().size());
    for (EdgeHandle e: node1->outEdges())
    {
        ASSERT_NE(nullptr, e);
        auto it = std::find(std::begin(edges), std::end(edges), e);
        ASSERT_NE(std::end(edges), it);
    }

    ASSERT_EQ(1, node2->inEdges().size());
    ASSERT_EQ(edges[0], node2->inEdges().front());

    ASSERT_EQ(1, node2->inNodes().size());
    ASSERT_EQ(node1, node2->inNodes().front());

    ASSERT_EQ(1, node3->inEdges().size());
    ASSERT_EQ(edges[1], node3->inEdges().front());

    ASSERT_EQ(1, node3->inNodes().size());
    ASSERT_EQ(node1, node3->inNodes().front());

    ASSERT_EQ(1, node4->inEdges().size());
    ASSERT_EQ(edges[2], node4->inEdges().front());

    ASSERT_EQ(1, node4->inNodes().size());
    ASSERT_EQ(node1, node4->inNodes().front());
}
