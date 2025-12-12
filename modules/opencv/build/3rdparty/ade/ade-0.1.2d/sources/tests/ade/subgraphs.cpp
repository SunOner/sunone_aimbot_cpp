// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <unordered_set>
#include <vector>
#include <iostream>

#include <ade/graph.hpp>
#include <ade/helpers/subgraphs.hpp>

#include <ade/util/iota_range.hpp>

namespace
{
bool hasNode(const ade::NodeHandle& node,
             const std::vector<ade::NodeHandle>& subgraph)
{
    ADE_ASSERT(nullptr != node);
    return subgraph.end() != std::find(subgraph.begin(), subgraph.end(), node);
}

bool isSame(const std::vector<ade::NodeHandle>& subgraph1,
            const std::vector<ade::NodeHandle>& subgraph2)
{
    if (subgraph1.size() != subgraph2.size())
    {
        return false;
    }

    for (auto&& node: subgraph1)
    {
        if (!hasNode(node, subgraph2))
        {
            return false;
        }
    }
    for (auto&& node: subgraph2)
    {
        if (!hasNode(node, subgraph1))
        {
            return false;
        }
    }
    return true;
}
}

TEST(Helpers, Subgraphs)
{
    // Graph:
    //
    // 0 1   2
    // |  \ /
    // 3   4
    // |   |
    // S   S
    //  \ / \
    //   5   6
    //   |   |
    //   7   S
    //   |   |
    //   S   |
    //    \ /
    //     8
    //     |
    //     9
    //
    // Roots: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // S - plitter nodes
    //
    // Resulting subgraphs:
    // 0, 3
    // 1, 2, 4
    // 5, 7
    // 6
    // 8, 9
    //
    // Splitter nodes not accessible from roots

    ade::Graph gr;

    std::vector<ade::NodeHandle> nodes{
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };

    std::vector<ade::NodeHandle> splitterNodes{
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()
    };

    gr.link(nodes[0],nodes[3]);
    gr.link(nodes[3],splitterNodes[0]);

    gr.link(nodes[1],nodes[4]);
    gr.link(nodes[2],nodes[4]);
    gr.link(nodes[4],splitterNodes[1]);

    gr.link(splitterNodes[0],nodes[5]);
    gr.link(splitterNodes[1],nodes[5]);
    gr.link(nodes[5],nodes[7]);
    gr.link(nodes[7],splitterNodes[2]);

    gr.link(splitterNodes[1],nodes[6]);
    gr.link(nodes[6],splitterNodes[3]);

    gr.link(splitterNodes[2],nodes[8]);
    gr.link(splitterNodes[3],nodes[8]);
    gr.link(nodes[8],nodes[9]);

    const auto subgraphs = ade::splitSubgraphs(nodes,
                                               [&](const ade::EdgeHandle& edge)
    {
        return (splitterNodes.end() != std::find(splitterNodes.begin(), splitterNodes.end(), edge->srcNode())) ||
               (splitterNodes.end() != std::find(splitterNodes.begin(), splitterNodes.end(), edge->dstNode()));
    });

    std::vector<std::vector<ade::NodeHandle>> expected{
        {nodes[0],nodes[3]},
        {nodes[1],nodes[2],nodes[4]},
        {nodes[5],nodes[7]},
        {nodes[6]},
        {nodes[8],nodes[9]}
    };

    ASSERT_EQ(expected.size(), subgraphs.size());

    for (auto&& expected_subgraph: expected)
    {
        bool found = false;
        for (auto&& subgraph: subgraphs)
        {
            if (isSame(expected_subgraph, subgraph))
            {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

namespace
{
using node_set = std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>>;

template<typename C>
void sort_subgraphs(C& c)
{
    std::sort(c.begin(), c.end(),
              [](const decltype(c.front())& val1,
                 const decltype(c.front())& val2)
    {
        return val1.front().get() < val2.front().get();
    });
}

template<bool Value>
struct Always
{
    template<typename... T>
    bool operator()(T&&...) const
    {
        return Value;
    }
};
}

TEST(Helpers, AssembleSubgraphSimple)
{
    //     0
    //    /|
    //   1 |
    //  /|/
    // 2 3
    //  \|
    //   4
    //   |
    //   5

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[0], nodes[3]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[4]);
    gr.link(nodes[3], nodes[4]);
    gr.link(nodes[4], nodes[5]);
    for (auto&& node : nodes)
    {
        auto subgraph = ade::assembleSubgraph(node, Always<true>{},Always<true>{});

        EXPECT_EQ((node_set(std::begin(nodes), std::end(nodes))), node_set(subgraph.begin(),subgraph.end()));
    }
}

namespace
{
bool hasCycle(const ade::subgraphs::NodesSet& acceptedNodes,
              const ade::subgraphs::NodesSet& rejectedNodes)
{
    for (auto&& srcNode : acceptedNodes)
    {
        ADE_ASSERT(nullptr != srcNode);
        if (srcNode->outNodes().size() > 1)
        {
            for (auto&& dstNode : acceptedNodes)
            {
                ADE_ASSERT(nullptr != dstNode);
                if (srcNode != dstNode &&
                    dstNode->inNodes().size() > 1)
                {
                    bool invalidPath = false;
                    ade::findPaths(srcNode, dstNode,
                                   [&](const std::vector<ade::NodeHandle>& path)
                    {
                        for (auto&& node : path)
                        {
                            ADE_ASSERT(nullptr != node);
                            if (ade::util::contains(rejectedNodes, node))
                            {
                                invalidPath = true;
                                return true;
                            }
                        }
                        return false;
                    });
                    if (invalidPath)
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
}

TEST(Helpers, AssembleSubgraphCycles1)
{
    //    0
    //    |
    //    1
    //   / \
    //  2   3
    //  |   |
    //  4   5
    //   \ /
    //    6
    //    |
    //    7

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[4]);
    gr.link(nodes[3], nodes[5]);
    gr.link(nodes[4], nodes[6]);
    gr.link(nodes[5], nodes[6]);
    gr.link(nodes[6], nodes[7]);

    auto mergeChecker = [&](
                        const ade::EdgeHandle& edge,
                        ade::SubgraphMergeDirection direction)
    {
        auto dstNode = ade::getDstMergeNode(edge, direction);

        if (dstNode == nodes[3] || dstNode == nodes[5])
        {
            return false;
        }

        return true;
    };
    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    auto topoChecker = [&](
                       const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
    {
        EXPECT_EQ(hasCycle(acceptedNodes, rejectedNodes),
                  cycleChecker(acceptedNodes, rejectedNodes));
        return !cycleChecker(acceptedNodes, rejectedNodes);
    };
    ade::NodeHandle roots[] = {
        nodes[0],
        nodes[1],
        nodes[2],
        nodes[4],
        nodes[6],
        nodes[7]
    };
    auto subgraphs = ade::selectSubgraphs(roots, mergeChecker, topoChecker);
    ASSERT_EQ(2, subgraphs.size());
    auto res0 = node_set(subgraphs[0].begin(), subgraphs[0].end());
    auto res1 = node_set(subgraphs[1].begin(), subgraphs[1].end());
    auto exp0 = node_set{nodes[0],nodes[1],nodes[2],nodes[4]};
    auto exp1 = node_set{nodes[2],nodes[4],nodes[6],nodes[7]};
    EXPECT_TRUE((exp0 == res0 && exp1 == res1) ||
                (exp0 == res1 && exp1 == res0));
}

TEST(Helpers, AssembleSubgraphCycles2)
{
    //    0
    //    |
    //    1
    //   / \
    //  2   3
    //  |   |
    //  4   5
    //  |   |
    //  6   7
    //   \ /
    //    8
    //    |
    //    9

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[4]);
    gr.link(nodes[3], nodes[5]);
    gr.link(nodes[4], nodes[6]);
    gr.link(nodes[5], nodes[7]);
    gr.link(nodes[6], nodes[8]);
    gr.link(nodes[7], nodes[8]);
    gr.link(nodes[8], nodes[9]);

    auto mergeChecker = [&](
                        const ade::EdgeHandle& edge,
                        ade::SubgraphMergeDirection direction)
    {
        auto dstNode = ade::getDstMergeNode(edge, direction);

        if (dstNode == nodes[5])
        {
            return false;
        }

        return true;
    };
    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    auto topoChecker = [&](
                       const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
    {
        EXPECT_EQ(hasCycle(acceptedNodes, rejectedNodes),
                  cycleChecker(acceptedNodes, rejectedNodes));
        return !cycleChecker(acceptedNodes, rejectedNodes);
    };
    ade::NodeHandle roots[] = {
        nodes[0],
        nodes[1],
        nodes[2],
        nodes[3],
        nodes[4],
        nodes[6],
        nodes[7],
        nodes[8],
        nodes[9]
    };
    auto subgraphs = ade::selectSubgraphs(roots, mergeChecker, topoChecker);
    ASSERT_EQ(2, subgraphs.size());
    auto res0 = node_set(subgraphs[0].begin(), subgraphs[0].end());
    auto res1 = node_set(subgraphs[1].begin(), subgraphs[1].end());
    auto exp0 = node_set{nodes[0],nodes[1],nodes[2],nodes[3],nodes[4],nodes[6]};
    auto exp1 = node_set{nodes[2],nodes[4],nodes[6],nodes[7],nodes[8],nodes[9]};
    EXPECT_TRUE((exp0 == res0 && exp1 == res1) ||
                (exp0 == res1 && exp1 == res0));
}

TEST(Helpers, AssembleSubgraphCyclesStress1)
{
    //       0
    //       |
    //     --1--
    //   /  / \  \
    //  2  3   4  5
    //   \  \ /  /
    //     --6--
    //       |
    //      ...
    //   repeat 50 times


    ade::Graph gr;
    std::vector<ade::NodeHandle> nodes;
    ade::NodeHandle prevNode;

    for (auto i : ade::util::iota(50))
    {
        (void)i;
        ade::NodeHandle tempNodes[] = {
            gr.createNode(),
            gr.createNode(),
            gr.createNode(),
            gr.createNode(),
            gr.createNode(),
            gr.createNode(),
            gr.createNode()
        };
        if (nullptr != prevNode)
        {
            gr.link(prevNode, tempNodes[0]);
        }
        gr.link(tempNodes[0], tempNodes[1]);
        gr.link(tempNodes[1], tempNodes[2]);
        gr.link(tempNodes[1], tempNodes[3]);
        gr.link(tempNodes[1], tempNodes[4]);
        gr.link(tempNodes[1], tempNodes[5]);
        gr.link(tempNodes[2], tempNodes[6]);
        gr.link(tempNodes[3], tempNodes[6]);
        gr.link(tempNodes[4], tempNodes[6]);
        gr.link(tempNodes[5], tempNodes[6]);
        prevNode = tempNodes[6];
        ade::util::copy(tempNodes, std::back_inserter(nodes));
    }

    auto mergeChecker = [&](
                        const ade::EdgeHandle& /*edge*/,
                        ade::SubgraphMergeDirection /*direction*/)
    {
        return true;
    };
    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    auto topoChecker = [&](
                       const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
    {
        return !cycleChecker(acceptedNodes, rejectedNodes);
    };
    auto subgraphs = ade::selectSubgraphs(nodes, mergeChecker, topoChecker);
    ASSERT_EQ(1, subgraphs.size());
    ASSERT_EQ(nodes.size(), subgraphs.front().size());
}

TEST(Helpers, AssembleSubgraphCyclesStress2)
{
    //      ...
    //      / \
    //     |   0*
    //     |   |
    //     |   1
    //     |   |
    //     |   2
    //     |   |
    //     |   3
    //      \ /
    //       4*
    //      / \
    //      ...
    //   repeat N times


    ade::Graph gr;
    std::vector<ade::NodeHandle> nodes;
    ade::subgraphs::NodesSet rejectedNodes;
    ade::NodeHandle prevNode = gr.createNode();

    int N = 20;

    for (auto i : ade::util::iota(N))
    {
        (void)i;
        ade::NodeHandle tempNodes[] = {
            gr.createNode(),
            gr.createNode(),
            gr.createNode(),
            gr.createNode(),
            gr.createNode()
        };
        if (nullptr != prevNode)
        {
            gr.link(prevNode, tempNodes[0]);
            gr.link(prevNode, tempNodes[4]);
        }

        rejectedNodes.insert(tempNodes[0]);
        rejectedNodes.insert(tempNodes[4]);

        gr.link(tempNodes[0], tempNodes[1]);
        gr.link(tempNodes[1], tempNodes[2]);
        gr.link(tempNodes[2], tempNodes[3]);
        gr.link(tempNodes[3], tempNodes[4]);
        prevNode = tempNodes[4];
        ade::util::copy(tempNodes, std::back_inserter(nodes));
    }

    auto mergeChecker = [&](
                        const ade::EdgeHandle& edge,
                        ade::SubgraphMergeDirection /*direction*/)
    {
        return ade::util::contains(rejectedNodes, edge->srcNode()) ==
               ade::util::contains(rejectedNodes, edge->dstNode());
    };
    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    auto topoChecker = [&](
                       const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
    {
        return !cycleChecker(acceptedNodes, rejectedNodes);
    };
    auto subgraphs = ade::selectSubgraphs(nodes, mergeChecker, topoChecker);
    ASSERT_EQ(N * 2 + 1, subgraphs.size());
}

namespace
{
bool hasMultipleInputs(const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
{
    int numInputs = 0;
    for (auto&& node : acceptedNodes)
    {
        if (node->inEdges().empty())
        {
            ++numInputs;
            if (numInputs > 1)
            {
                return true;
            }
        }
        else
        {
            for (auto&& node1 : node->inNodes())
            {
                if (ade::util::contains(rejectedNodes, node1))
                {
                    ++numInputs;
                    if (numInputs > 1)
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
}

TEST(Helpers, AssembleSubgraphMultipleInputs1)
{
    // Forbid graphs with multiple inputs
    //  0
    //  |
    //  1   2
    //   \ /
    //    3
    //    |
    //    4

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[3]);
    gr.link(nodes[3], nodes[4]);

    Always<true> mergeChecker;
    auto topoChecker = [](
                       const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
    {
        return !hasMultipleInputs(acceptedNodes,rejectedNodes);
    };

    auto subgraphs = ade::selectSubgraphs(nodes, mergeChecker, topoChecker);
    ASSERT_EQ(3, subgraphs.size());
    sort_subgraphs(subgraphs);

    std::vector<std::vector<ade::NodeHandle>> resSubgraphs = {
        {nodes[0],nodes[1]},
        {nodes[2]},
        {nodes[4]}};
    sort_subgraphs(resSubgraphs);

    EXPECT_EQ(resSubgraphs, subgraphs);
}

TEST(Helpers, AssembleSubgraphMultipleInputs2)
{
    // Forbid graphs with multiple inputs
    //    0
    //   / \
    //  1   2
    //   \ /
    //    3
    //    |
    //    4

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[0], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[3]);
    gr.link(nodes[3], nodes[4]);

    Always<true> mergeChecker;
    auto topoChecker = [](
                       const ade::subgraphs::NodesSet& acceptedNodes,
                       const ade::subgraphs::NodesSet& rejectedNodes)
    {
        return !hasMultipleInputs(acceptedNodes,rejectedNodes);
    };

    auto subgraphs = ade::selectSubgraphs(std::vector<ade::NodeHandle>{nodes[4]},
                                          mergeChecker, topoChecker);
    ASSERT_EQ(1, subgraphs.size());

    EXPECT_EQ((node_set{nodes[0],nodes[1],nodes[2],nodes[3],nodes[4]}),
               node_set(subgraphs[0].begin(), subgraphs[0].end()));
}

namespace
{
std::size_t getLongestSubgraph(
        const std::vector<std::vector<ade::NodeHandle>>& subgraphs)
{
    std::size_t ret = ade::SubgraphSelectResult::NoSubgraph;
    std::size_t maxlen = 0;
    for (auto i : ade::util::iota(subgraphs.size()))
    {
        const auto& s = subgraphs[i];
        if (s.size() > maxlen)
        {
            maxlen = s.size();
            ret = i;
        }
    }
    return ret;
}
}

TEST(Helpers, SelectAllSubgraphs1)
{
    //     0
    //     | \
    //     1  |
    //   / |  |
    //  2  3  4
    //  |  |  |
    //  5  6  7
    //   \ | /
    //     8
    //     |
    //     9

    ade::Graph gr;
    std::vector<ade::NodeHandle> nodes;
    for (auto i : ade::util::iota(10))
    {
        (void)i;
        nodes.push_back(gr.createNode());
    }
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[0], nodes[4]);
    gr.link(nodes[2], nodes[5]);
    gr.link(nodes[3], nodes[6]);
    gr.link(nodes[4], nodes[7]);
    gr.link(nodes[5], nodes[8]);
    gr.link(nodes[6], nodes[8]);
    gr.link(nodes[7], nodes[8]);
    gr.link(nodes[8], nodes[9]);

    ade::subgraphs::NodesSet nodes1;
    ade::subgraphs::NodesSet nodes2;
    for (auto i : ade::util::iota(10))
    {
        if (0 == i || 4 == i || 7 == i || 8 == i || 9 == i)
        {
            nodes2.insert(nodes[i]);
        }
        else
        {
            nodes1.insert(nodes[i]);
        }
    }

    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    auto subgraphs1 = ade::selectAllSubgraphs(
                          nodes1,
    [&](
    const ade::EdgeHandle& edge,
    const ade::SubgraphMergeDirection dir)
    {
        auto dstNode = ade::getDstMergeNode(edge, dir);
        ADE_ASSERT(nullptr != dstNode);
        return ade::util::contains(nodes1, dstNode);
    },
    [&](
    const ade::subgraphs::NodesSet& acceptedNodes,
    const ade::subgraphs::NodesSet& rejectedNodes)
    {
        EXPECT_EQ(hasCycle(acceptedNodes, rejectedNodes),
                  cycleChecker(acceptedNodes, rejectedNodes));
        return !cycleChecker(acceptedNodes, rejectedNodes);
    },
    [](const std::vector<std::vector<ade::NodeHandle>>& subgraphs)
    {
        ade::SubgraphSelectResult ret;
        ret.index = getLongestSubgraph(subgraphs);
        ret.continueSelect = true;
        return ret;
    });
    auto subgraphs2 = ade::selectAllSubgraphs(
                          nodes2,
    [&](
    const ade::EdgeHandle& edge,
    const ade::SubgraphMergeDirection dir)
    {
        auto dstNode = ade::getDstMergeNode(edge, dir);
        ADE_ASSERT(nullptr != dstNode);
        return ade::util::contains(nodes2, dstNode);
    },
    [&](
    const ade::subgraphs::NodesSet& acceptedNodes,
    const ade::subgraphs::NodesSet& rejectedNodes)
    {
        EXPECT_EQ(hasCycle(acceptedNodes, rejectedNodes),
                  cycleChecker(acceptedNodes, rejectedNodes));
        return !cycleChecker(acceptedNodes, rejectedNodes);
    },
    [](const std::vector<std::vector<ade::NodeHandle>>& subgraphs)
    {
        ade::SubgraphSelectResult ret;
        ret.index = getLongestSubgraph(subgraphs);
        ret.continueSelect = true;
        return ret;
    });
    ASSERT_EQ(1, subgraphs1.size());
    ASSERT_EQ(2, subgraphs2.size());

    EXPECT_EQ((node_set{nodes[1],nodes[2],nodes[3],nodes[5],nodes[6]}),
               node_set(subgraphs1[0].begin(), subgraphs1[0].end()));

    EXPECT_EQ((node_set{nodes[4],nodes[7],nodes[8],nodes[9]}),
               node_set(subgraphs2[0].begin(), subgraphs2[0].end()));
    EXPECT_EQ((node_set{nodes[0]}),
               node_set(subgraphs2[1].begin(), subgraphs2[1].end()));
}

TEST(Helpers, SelectAllSubgraphs2)
{
    //  0   1
    //  |   |
    //  2   3
    //   \ / \
    //    4   5
    //    |   |
    //    6   7
    //   / \ /
    //  8   9
    //  |   |
    //  10  11
    //   \ /
    //    12

    ade::Graph gr;
    std::vector<ade::NodeHandle> nodes;
    for (auto i : ade::util::iota(13))
    {
        (void)i;
        nodes.push_back(gr.createNode());
    }
    gr.link(nodes[0], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[4]);
    gr.link(nodes[3], nodes[4]);
    gr.link(nodes[3], nodes[5]);
    gr.link(nodes[4], nodes[6]);
    gr.link(nodes[5], nodes[7]);
    gr.link(nodes[6], nodes[8]);
    gr.link(nodes[6], nodes[9]);
    gr.link(nodes[7], nodes[9]);
    gr.link(nodes[8], nodes[10]);
    gr.link(nodes[9], nodes[11]);
    gr.link(nodes[10], nodes[12]);
    gr.link(nodes[11], nodes[12]);

    ade::subgraphs::NodesSet nodes1;
    ade::subgraphs::NodesSet nodes2;
    for (auto i : ade::util::iota(std::size_t(13)))
    {
        if (5 == i || 7 == i || 8 == i || 10 == i)
        {
            nodes2.insert(nodes[i]);
        }
        else
        {
            nodes1.insert(nodes[i]);
        }
    }

    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    auto subgraphs1 = ade::selectAllSubgraphs(
                          nodes1,
    [&](
    const ade::EdgeHandle& edge,
    const ade::SubgraphMergeDirection dir)
    {
        auto dstNode = ade::getDstMergeNode(edge, dir);
        ADE_ASSERT(nullptr != dstNode);
        return ade::util::contains(nodes1, dstNode);
    },
    [&](
    const ade::subgraphs::NodesSet& acceptedNodes,
    const ade::subgraphs::NodesSet& rejectedNodes)
    {
        EXPECT_EQ(hasCycle(acceptedNodes, rejectedNodes),
                  cycleChecker(acceptedNodes, rejectedNodes));
        return !cycleChecker(acceptedNodes, rejectedNodes);
    },
    [](const std::vector<std::vector<ade::NodeHandle>>& subgraphs)
    {
        ade::SubgraphSelectResult ret;
        ret.index = getLongestSubgraph(subgraphs);
        ret.continueSelect = false;
        return ret;
    });
    auto subgraphs2 = ade::selectAllSubgraphs(
                          nodes2,
    [&](
    const ade::EdgeHandle& /*edge*/,
    const ade::SubgraphMergeDirection /*dir*/)
    {
        return true;
    },
    [&](
    const ade::subgraphs::NodesSet& acceptedNodes,
    const ade::subgraphs::NodesSet& rejectedNodes)
    {
        EXPECT_EQ(hasCycle(acceptedNodes, rejectedNodes),
                  cycleChecker(acceptedNodes, rejectedNodes));
        return !cycleChecker(acceptedNodes, rejectedNodes);
    },
    [](const std::vector<std::vector<ade::NodeHandle>>& subgraphs)
    {
        ade::SubgraphSelectResult ret;
        ret.index = getLongestSubgraph(subgraphs);
        ret.continueSelect = true;
        return ret;
    });
    ASSERT_EQ(1, subgraphs1.size());
    ASSERT_EQ(2, subgraphs2.size());

    EXPECT_EQ((node_set{nodes[0],nodes[2],nodes[4],nodes[3],nodes[1],nodes[6]}),
               node_set(subgraphs1[0].begin(), subgraphs1[0].end()));

//    EXPECT_EQ((node_set{nodes[5],nodes[7],nodes[9],nodes[11]}),
//               node_set(subgraphs2[0].begin(), subgraphs2[0].end()));
//    EXPECT_EQ((node_set{nodes[8],nodes[10],nodes[12]}),
//               node_set(subgraphs2[1].begin(), subgraphs2[1].end()));
}

TEST(Helpers, FindPaths1)
{
    //    0
    //    |
    //    1
    //   / \
    //  2   3
    //  |   |
    //  4   5
    //   \ /
    //    6
    //    |
    //    7

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[4]);
    gr.link(nodes[3], nodes[5]);
    gr.link(nodes[4], nodes[6]);
    gr.link(nodes[5], nodes[6]);
    gr.link(nodes[6], nodes[7]);

    std::vector<std::vector<ade::NodeHandle>> resPaths = {
        {nodes[0],nodes[1],nodes[2],nodes[4],nodes[6],nodes[7]},
        {nodes[0],nodes[1],nodes[3],nodes[5],nodes[6],nodes[7]}};
    sort_subgraphs(resPaths);

    {
        std::vector<std::vector<ade::NodeHandle>> paths;
        ade::findPaths(nodes[0], nodes[7], [&](const std::vector<ade::NodeHandle>& path)
        {
            paths.push_back(path);
            return false;
        });
        ASSERT_EQ(2, paths.size());
        sort_subgraphs(paths);

        EXPECT_EQ(resPaths, paths);
    }

    {
        std::vector<std::vector<ade::NodeHandle>> paths;
        ade::findPaths(nodes[0], nodes[7], [&](const std::vector<ade::NodeHandle>& path)
        {
            paths.push_back(path);
            return true;
        });
        ASSERT_EQ(1, paths.size());

        EXPECT_TRUE((paths[0] == resPaths[0]) ||
                    (paths[0] == resPaths[1]));
    }

    {
        std::vector<std::vector<ade::NodeHandle>> paths;
        ade::findBiPaths(nodes[7], nodes[0], [&](const std::vector<ade::NodeHandle>& path)
        {
            paths.push_back(path);
            return false;
        });
        ASSERT_EQ(2, paths.size());
        sort_subgraphs(paths);

        EXPECT_EQ(resPaths, paths);
    }
}

TEST(Helpers, FindPaths2)
{
    //     0
    //     |
    //     1
    //   /   \
    //  2     3
    //  |    / \
    //  4   5   6
    //   \  |  /
    //      7
    //      |
    //      8

    ade::Graph gr;
    ade::NodeHandle nodes[] = {
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode(),
        gr.createNode()};
    gr.link(nodes[0], nodes[1]);
    gr.link(nodes[1], nodes[2]);
    gr.link(nodes[1], nodes[3]);
    gr.link(nodes[2], nodes[4]);
    gr.link(nodes[3], nodes[5]);
    gr.link(nodes[3], nodes[6]);
    gr.link(nodes[4], nodes[7]);
    gr.link(nodes[5], nodes[7]);
    gr.link(nodes[6], nodes[7]);
    gr.link(nodes[7], nodes[8]);

    std::vector<std::vector<ade::NodeHandle>> resPaths = {
        {nodes[0],nodes[1],nodes[2],nodes[4],nodes[7],nodes[8]},
        {nodes[0],nodes[1],nodes[3],nodes[5],nodes[7],nodes[8]},
        {nodes[0],nodes[1],nodes[3],nodes[6],nodes[7],nodes[8]}};
    sort_subgraphs(resPaths);

    {
        std::vector<std::vector<ade::NodeHandle>> paths;
        ade::findPaths(nodes[0], nodes[8], [&](const std::vector<ade::NodeHandle>& path)
        {
            paths.push_back(path);
            return false;
        });
        ASSERT_EQ(3, paths.size());
        sort_subgraphs(paths);

        EXPECT_EQ(resPaths, paths);
    }

    {
        std::vector<std::vector<ade::NodeHandle>> paths;
        ade::findPaths(nodes[0], nodes[8], [&](const std::vector<ade::NodeHandle>& path)
        {
            paths.push_back(path);
            return true;
        });
        ASSERT_EQ(1, paths.size());

        EXPECT_TRUE((paths[0] == resPaths[0]) ||
                    (paths[0] == resPaths[1]) ||
                    (paths[0] == resPaths[2]));
    }

    {
        std::vector<std::vector<ade::NodeHandle>> paths;
        ade::findBiPaths(nodes[8], nodes[0], [&](const std::vector<ade::NodeHandle>& path)
        {
            paths.push_back(path);
            return false;
        });
        ASSERT_EQ(3, paths.size());
        sort_subgraphs(paths);

        EXPECT_EQ(resPaths, paths);
    }
}
