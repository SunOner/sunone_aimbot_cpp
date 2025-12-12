// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/passes/check_cycles.hpp>

using namespace ade;

TEST(CheckCyclesPass, Simple)
{
    using namespace passes;
    Graph gr;
    PassContext context{gr};
    ASSERT_NO_THROW(CheckCycles()(context)); // Empty graph
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
    ASSERT_NO_THROW(CheckCycles()(context));
}

TEST(CheckCyclesPass, Cycle1)
{
    using namespace passes;
    Graph gr;
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto edge1 = gr.link(node1, node2);
    auto edge2 = gr.link(node2, node1);
    ASSERT_THROW(CheckCycles()(context), CycleFound);
}

TEST(CheckCyclesPass, Cycle2)
{
    using namespace passes;
    Graph gr;
    PassContext context{gr};
    auto node1 = gr.createNode();
    auto node2 = gr.createNode();
    auto node3 = gr.createNode();
    auto node4 = gr.createNode();
    auto node5 = gr.createNode();
    auto node6 = gr.createNode();
    auto edge1 = gr.link(node1, node2);
    auto edge2 = gr.link(node2, node3);
    auto edge3 = gr.link(node3, node4);
    auto edge4 = gr.link(node4, node5);
    auto edge5 = gr.link(node5, node6);
    auto edge6 = gr.link(node6, node1);
    ASSERT_THROW(CheckCycles()(context), CycleFound);
}

TEST(CheckCyclesPass, Cycle3)
{
    using namespace passes;
    Graph gr;
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
    auto edge6 = gr.link(node1, node2);
    auto edge7 = gr.link(node6, node3);
    ASSERT_THROW(CheckCycles()(context), CycleFound);
}
