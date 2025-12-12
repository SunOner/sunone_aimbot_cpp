// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passmanager.hpp>

struct Node
{
    std::string op;

    static const char *name() {return "Node";}
};

struct State
{
    enum {
        DIRTY, COMPLETE
    } st;

    static const char *name() {return "State";}
};

int main(int argc, char *argv[])
{
    // Build the following graph:
    //
    //    A ---> B ---> C ---> D      L
    //           :             :      ^
    //           V             V      :
    //    F <--- E ---> G      H ---> I
    //    :      :             :      :
    //    :      `----> J <----'      :
    //    :             :             V
    //    '----> M <----'             K
    // FM and JM shouldn't be fused!

    ade::Graph graph;

    using TGraph = ade::TypedGraph<Node, State>;
    TGraph tgraph(graph);

    std::unordered_map<char, ade::NodeHandle> nodes;
    for (char c : std::string("ABCDEFGHIJKLM"))
    {
        nodes[c] = tgraph.createNode();
        const char s[] = {c};
        tgraph.metadata(nodes[c]).set(Node{s});
    }

#define L(c1,c2) graph.link(nodes[c1], nodes[c2])
    L('A', 'B');
    L('B', 'C');
    L('C', 'D');
    L('B', 'E');
    L('E', 'F');
    L('E', 'G');
    L('E', 'J');
    L('D', 'H');
    L('H', 'I');
    L('I', 'L');
    L('I', 'K');
    L('H', 'J');
    L('F', 'M');
    L('J', 'M');
#undef  L

    tgraph.metadata().set(State{State::DIRTY});

    // Define passes. There are two:
    // 1. Dump pass - dumps graph in Graphviz format
    auto dump_pass = [](TGraph &gr) {
        for (const auto src : gr.nodes())
        {
            for (const auto out : src->outEdges())
            {
                auto dst = out->dstNode();
                std::cout << gr.metadata(src).get<Node>().op
                          << " -> "
                          << gr.metadata(dst).get<Node>().op
                          << "\n";
            }
        }
    };

    // 2. Squash pass - Replaces a series of Nodes with a squashed
    // one if there's no other links in-between them
    // If there was nothing to squash, Graph status is set to COMPLETE
    auto squash_pass = [](TGraph &gr) {
        for (auto src : gr.nodes())
        {
            if (src->outEdges().size() == 1)
            {
                auto dst = src->outNodes().front();
                if (dst->inEdges().size() == 1)
                {
                    // Node "X" node has a single consumer ("Y"),
                    // so we can replace "X->Y" with "XY".
                    ade::NodeHandle   fused    = gr.createNode();
                    const std::string fused_op =
                        gr.metadata(src).get<Node>().op
                        + gr.metadata(dst).get<Node>().op;
                    gr.metadata(fused).set<Node>(Node{fused_op});

                    for (auto inNode : src->inNodes())
                    {
                        gr.link(inNode, fused);
                    }
                    for (auto outNode : dst->outNodes())
                    {
                        gr.link(fused, outNode);
                    }

                    gr.erase(src);
                    gr.erase(dst);
                    gr.metadata().set(State{State::DIRTY});
                    return;
                }
            }
        }
        gr.metadata().set(State{State::COMPLETE});
    };

    // Now define a pass list and to the squashing whenever possible!
    std::cout << "Initial graph" << std::endl;
    dump_pass(tgraph);

    ade::PassList<TGraph> list;
    list.addPass(squash_pass);
    list.addPass(dump_pass);

    int n = 0, origNumNodes = tgraph.nodes().size();
    while (   n < origNumNodes
           && tgraph.metadata().get<State>().st != State::COMPLETE)
    {
        std::cout << "====================\n";
        std::cout << "Iteration #" << n << std::endl;
        list.run(tgraph);
        n++;
        std::cout << tgraph.nodes().size() << " nodes in graph" << std::endl;
    }

    std::cout << "====================\n"
              << "End!" << std::endl;
    return 0;
}
