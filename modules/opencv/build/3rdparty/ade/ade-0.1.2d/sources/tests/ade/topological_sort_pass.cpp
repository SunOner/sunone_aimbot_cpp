// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passes/topological_sort.hpp>

using namespace ade;

template<typename T>
static std::vector<NodeHandle> toVec(T range)
{
    return std::vector<ade::NodeHandle>(range.begin(), range.end());
}

TEST(TopologicalSortPass, Simple1)
{
    using namespace passes;
    Graph gr;
    TypedGraph<TopologicalSortData> tgr(gr);
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto edge1 = gr.link(node1, node2);
    TopologicalSort()(context);
    const auto sorted = toVec(tgr.metadata().get<TopologicalSortData>().nodes());
    ASSERT_EQ(2, sorted.size());
    ASSERT_EQ(node1, sorted[0]);
    ASSERT_EQ(node2, sorted[1]);
}

TEST(TopologicalSortPass, Simple2)
{
    using namespace passes;
    Graph gr;
    TypedGraph<TopologicalSortData> tgr(gr);
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto edge1 = gr.link(node2, node1);
    TopologicalSort()(context);
    const auto sorted = toVec(tgr.metadata().get<TopologicalSortData>().nodes());
    ASSERT_EQ(2, sorted.size());
    ASSERT_EQ(node2, sorted[0]);
    ASSERT_EQ(node1, sorted[1]);
}

static void checkOrder(const std::vector<NodeHandle>& nodes)
{
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        auto node = nodes[i];
        for (auto prev:
             util::map(node->inEdges(), [](const EdgeHandle& e) { return e->srcNode(); }))
        {
            auto it = std::find(nodes.begin(), nodes.begin() + i, prev);
            EXPECT_NE(nodes.begin() + i, it);
        }
    }
}

TEST(TopologicalSortPass, Complex1)
{
    using namespace passes;
    Graph gr;
    TypedGraph<TopologicalSortData> tgr(gr);
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();
    auto node4 = gr.createNode();
    auto node5 = gr.createNode();
    auto node6 = gr.createNode();
    auto edge1 = gr.link(node6, node5);
    auto edge2 = gr.link(node5, node4);
    auto edge3 = gr.link(node4, node2);
    auto edge4 = gr.link(node2, node3);
    auto edge5 = gr.link(node3, node1);
    TopologicalSort()(context);
    const auto sorted = toVec(tgr.metadata().get<TopologicalSortData>().nodes());
    checkOrder(sorted);
    ASSERT_EQ(6, sorted.size());
    ASSERT_EQ(node6, sorted[0]);
    ASSERT_EQ(node5, sorted[1]);
    ASSERT_EQ(node4, sorted[2]);
    ASSERT_EQ(node2, sorted[3]);
    ASSERT_EQ(node3, sorted[4]);
    ASSERT_EQ(node1, sorted[5]);
}

TEST(TopologicalSortPass, Complex2)
{
    using namespace passes;
    Graph gr;
    TypedGraph<TopologicalSortData> tgr(gr);
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();
    auto node4 = gr.createNode();
    auto node5 = gr.createNode();
    auto node6 = gr.createNode();
    auto edge1 = gr.link(node2, node3);
    auto edge2 = gr.link(node2, node5);
    auto edge3 = gr.link(node3, node4);
    auto edge4 = gr.link(node3, node5);
    auto edge5 = gr.link(node4, node1);
    auto edge6 = gr.link(node4, node6);
    TopologicalSort()(context);
    const auto sorted = toVec(tgr.metadata().get<TopologicalSortData>().nodes());
    ASSERT_EQ(6, sorted.size());
    checkOrder(sorted);
}

TEST(TopologicalSortPass, TestNodeRemoval)
{
    using namespace passes;
    Graph gr;
    TypedGraph<TopologicalSortData> tgr(gr);
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();
    auto edge1 = gr.link(node1, node2);
    auto edge2 = gr.link(node2, node3);
    TopologicalSort()(context);
    {
        const auto sorted = toVec(tgr.metadata().get<TopologicalSortData>().nodes());
        ASSERT_EQ((std::vector<NodeHandle>{node1,node2,node3}), sorted);
    }
    gr.erase(node2);
    {
        const auto sorted = toVec(tgr.metadata().get<TopologicalSortData>().nodes());
        ASSERT_EQ((std::vector<NodeHandle>{node1,node3}), sorted);
    }
}
