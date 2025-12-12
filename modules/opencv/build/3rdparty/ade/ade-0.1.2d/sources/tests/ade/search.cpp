// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

#include <ade/graph.hpp>

#include <ade/helpers/search.hpp>

TEST(Search, Dfs)
{
    //     0
    //    / \
    //   1   2
    //  / \ / \
    // 3   4   5

    ade::Graph gr;

    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };

    gr.link(nodes[0],nodes[1]);
    gr.link(nodes[0],nodes[2]);
    gr.link(nodes[1],nodes[3]);
    gr.link(nodes[1],nodes[4]);
    gr.link(nodes[2],nodes[4]);
    gr.link(nodes[2],nodes[5]);

    std::unordered_multiset<ade::NodeHandle, ade::HandleHasher<ade::Node>>
            visited;

    ade::dfs(nodes[0], [&](const ade::NodeHandle& node)
    {
        visited.insert(node);
        return true;
    });

    ASSERT_EQ(6, visited.size());
    EXPECT_EQ(1, visited.count(nodes[1]));
    EXPECT_EQ(1, visited.count(nodes[2]));
    EXPECT_EQ(1, visited.count(nodes[3]));
    EXPECT_EQ(2, visited.count(nodes[4]));
    EXPECT_EQ(1, visited.count(nodes[5]));

    visited.clear();

    ade::dfs(nodes[0], [&](const ade::NodeHandle& node)
    {
        visited.insert(node);
        return (node != nodes[2]);
    });

    ASSERT_EQ(4, visited.size());
    EXPECT_EQ(1, visited.count(nodes[1]));
    EXPECT_EQ(1, visited.count(nodes[2]));
    EXPECT_EQ(1, visited.count(nodes[3]));
    EXPECT_EQ(1, visited.count(nodes[4]));
}

namespace
{
struct DfsFunctor
{
    template<typename F>
    void operator()(const ade::Node& node, F&& func)
    {
        for (const auto& nextNode : node.outNodes())
        {
            func(nextNode);
        }
    }
};
}

TEST(Search, DfsFunctors)
{
    //     0
    //    / \
    //   1   2
    //  / \ / \
    // 3   4   5

    ade::Graph gr;

    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };

    gr.link(nodes[0],nodes[1]);
    gr.link(nodes[0],nodes[2]);
    gr.link(nodes[1],nodes[3]);
    gr.link(nodes[1],nodes[4]);
    gr.link(nodes[2],nodes[4]);
    gr.link(nodes[2],nodes[5]);

    std::unordered_multiset<ade::NodeHandle, ade::HandleHasher<ade::Node>>
            visited;

    ade::dfs(nodes[0], [&](const ade::NodeHandle& node)
    {
        visited.insert(node);
        return true;
    },
    [](const ade::Node& node, std::function<void (const ade::NodeHandle&)> func)
    {
        for (auto&& nextNode : node.outNodes())
        {
            func(nextNode);
        }
    });

    ASSERT_EQ(6, visited.size());
    EXPECT_EQ(1, visited.count(nodes[1]));
    EXPECT_EQ(1, visited.count(nodes[2]));
    EXPECT_EQ(1, visited.count(nodes[3]));
    EXPECT_EQ(2, visited.count(nodes[4]));
    EXPECT_EQ(1, visited.count(nodes[5]));

    visited.clear();

    DfsFunctor func;

    ade::dfs(nodes[0], [&](const ade::NodeHandle& node)
    {
        visited.insert(node);
        return true;
    },
    func);

    ASSERT_EQ(6, visited.size());
    EXPECT_EQ(1, visited.count(nodes[1]));
    EXPECT_EQ(1, visited.count(nodes[2]));
    EXPECT_EQ(1, visited.count(nodes[3]));
    EXPECT_EQ(2, visited.count(nodes[4]));
    EXPECT_EQ(1, visited.count(nodes[5]));
}

TEST(Search, TransitiveClosureOut1)
{
    //     0
    //    / \
    //   1   2
    //  / \ / \
    // 3   4   5

    ade::Graph gr;

    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };

    gr.link(nodes[0],nodes[1]);
    gr.link(nodes[0],nodes[2]);
    gr.link(nodes[1],nodes[3]);
    gr.link(nodes[1],nodes[4]);
    gr.link(nodes[2],nodes[4]);
    gr.link(nodes[2],nodes[5]);

    std::unordered_map<ade::NodeHandle,
            std::unordered_multiset<ade::NodeHandle, ade::HandleHasher<ade::Node>>,
            ade::HandleHasher<ade::Node>>
            visited;

    ade::transitiveClosure(gr.nodes(), [&](const ade::NodeHandle& node1,
                                           const ade::NodeHandle& node2)
    {
        ADE_ASSERT(nullptr != node1);
        ADE_ASSERT(nullptr != node2);
        visited[node1].insert(node2);
    });

    EXPECT_EQ(3, visited.size());

    EXPECT_EQ(5, visited[nodes[0]].size());
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[1]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[2]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[3]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[4]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[5]));

    EXPECT_EQ(2, visited[nodes[1]].size());
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[3]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[4]));

    EXPECT_EQ(2, visited[nodes[2]].size());
    EXPECT_EQ(1, visited[nodes[2]].count(nodes[4]));
    EXPECT_EQ(1, visited[nodes[2]].count(nodes[5]));
}

TEST(Search, TransitiveClosureOut2)
{
    //     0   1
    //     |   |
    //     2   3
    //     |   |
    //     4   5

    ade::Graph gr;

    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };

    gr.link(nodes[0],nodes[2]);
    gr.link(nodes[2],nodes[4]);
    gr.link(nodes[1],nodes[3]);
    gr.link(nodes[3],nodes[5]);

    std::unordered_map<ade::NodeHandle,
            std::unordered_multiset<ade::NodeHandle, ade::HandleHasher<ade::Node>>,
            ade::HandleHasher<ade::Node>>
            visited;

    ade::transitiveClosure(gr.nodes(), [&](const ade::NodeHandle& node1,
                                           const ade::NodeHandle& node2)
    {
        ADE_ASSERT(nullptr != node1);
        ADE_ASSERT(nullptr != node2);
        visited[node1].insert(node2);
    });

    EXPECT_EQ(4, visited.size());

    EXPECT_EQ(2, visited[nodes[0]].size());
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[2]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[4]));

    EXPECT_EQ(1, visited[nodes[2]].size());
    EXPECT_EQ(1, visited[nodes[2]].count(nodes[4]));

    EXPECT_EQ(2, visited[nodes[1]].size());
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[3]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[5]));

    EXPECT_EQ(1, visited[nodes[3]].size());
    EXPECT_EQ(1, visited[nodes[3]].count(nodes[5]));
}

TEST(Search, TransitiveClosureOut3)
{
    //       0
    //       |
    //     --1--
    //   /  / \  \
    //  2  3   4  5
    //   \  \ /  /
    //     --6--
    //       |
    //       7

    ade::Graph gr;

    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[1], nodes[4]);
    gr.link(nodes[1], nodes[5]);
    gr.link(nodes[2], nodes[6]);
    gr.link(nodes[3], nodes[6]);
    gr.link(nodes[4], nodes[6]);
    gr.link(nodes[5], nodes[6]);
    gr.link(nodes[6], nodes[7]);

    std::unordered_map<ade::NodeHandle,
            std::unordered_multiset<ade::NodeHandle, ade::HandleHasher<ade::Node>>,
            ade::HandleHasher<ade::Node>>
            visited;

    ade::transitiveClosure(gr.nodes(), [&](const ade::NodeHandle& node1,
                                           const ade::NodeHandle& node2)
    {
        ADE_ASSERT(nullptr != node1);
        ADE_ASSERT(nullptr != node2);
        visited[node1].insert(node2);
    });

    EXPECT_EQ(7, visited.size());

    EXPECT_EQ(7, visited[nodes[0]].size());
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[1]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[2]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[3]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[4]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[5]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[6]));
    EXPECT_EQ(1, visited[nodes[0]].count(nodes[7]));

    EXPECT_EQ(6, visited[nodes[1]].size());
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[2]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[3]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[4]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[5]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[6]));
    EXPECT_EQ(1, visited[nodes[1]].count(nodes[7]));

    EXPECT_EQ(2, visited[nodes[2]].size());
    EXPECT_EQ(1, visited[nodes[2]].count(nodes[6]));
    EXPECT_EQ(1, visited[nodes[2]].count(nodes[7]));

    EXPECT_EQ(2, visited[nodes[3]].size());
    EXPECT_EQ(1, visited[nodes[3]].count(nodes[6]));
    EXPECT_EQ(1, visited[nodes[3]].count(nodes[7]));

    EXPECT_EQ(2, visited[nodes[4]].size());
    EXPECT_EQ(1, visited[nodes[4]].count(nodes[6]));
    EXPECT_EQ(1, visited[nodes[4]].count(nodes[7]));

    EXPECT_EQ(2, visited[nodes[5]].size());
    EXPECT_EQ(1, visited[nodes[5]].count(nodes[6]));
    EXPECT_EQ(1, visited[nodes[5]].count(nodes[7]));

    EXPECT_EQ(1, visited[nodes[6]].size());
    EXPECT_EQ(1, visited[nodes[6]].count(nodes[7]));
}
