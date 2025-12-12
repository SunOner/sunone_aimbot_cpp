// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <ade/graph.hpp>

int main(int argc, char *argv[])
{
    // Define node and edge handles before we start.
    // These objects act as smart references to ADE entities.
    ade::NodeHandle n1, n2;
    ade::EdgeHandle e;

    auto p_edges = [](const char *name, ade::NodeHandle n) {
        std::cout << name << " edges: in=" << n->inEdges().size() << ", out=" << n->outEdges().size() << std::endl;
    };

    auto p_conns = [](const char *name, ade::Node::EdgeSetRange edges) {
        std::cout << name << " connections: { ";
        for (const auto &h : edges) { std::cout << h << ":[" << h->srcNode() << "->" << h->dstNode() << "]"; }
        std::cout << " }" << std::endl;
    };

    {
        ade::Graph graph;

        // Use Graph::createNode() to add a new node to graph.
        n1 = graph.createNode();
        std::cout << "After 1st .createNode() [n1=" << n1 << "], there\'re " << graph.nodes().size() << " node(s)" << std::endl;

        n2 = graph.createNode();
        std::cout << "After 2nd .createNode() [n2=" << n2 << "], there\'re " << graph.nodes().size() << " node(s)" << std::endl;

        // Linking nodes is easy - use Graph::link()
        std::cout << std::endl;

        e = graph.link(n1, n2);
        std::cout << "Edge handle is " << e << std::endl;
        p_edges("n1", n1);
        p_edges("n2", n2);

        // Walk through connections
        p_conns("n1 (out)", n1->outEdges());
        p_conns("n2 (in) ", n2->inEdges());

        // Unlinking is easy as well
        std::cout << std::endl;

        graph.erase(e);
        std::cout << "After edge erase: edge handle is " << e << std::endl;
        p_edges("n1", n1);
        p_edges("n2", n2);

        p_conns("n1 (out)", n1->outEdges());
        p_conns("n2 (in) ", n2->inEdges());

        // Link again, now in another order
        std::cout << std::endl;

        e = graph.link(n2, n1);

        p_edges("n1", n1);
        p_edges("n2", n2);

        p_conns("n1 (in) " , n1->inEdges());
        p_conns("n2 (out)", n2->outEdges());

        // A node can also be removed with Graph::erase()
        std::cout << std::endl;

        graph.erase(n1);
        std::cout << "After erase(), there\'re " << graph.nodes().size() << " node(s)" << std::endl;
        std::cout << "n1 is now " << n1 << std::endl;

        // Removing a node also removes connections
        p_edges("n2", n2);
        p_conns("n2 (out)", n2->outEdges());
    }

    // Resource management is smart as we^well
    std::cout << std::endl;
    std::cout << "After graph destruction: n1=" << n1 << ", n2=" << n2 << ", e=" << e << std::endl;

    return 0;
}
