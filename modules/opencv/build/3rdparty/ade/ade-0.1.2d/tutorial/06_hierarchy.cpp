// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <iostream>
#include <string>
#include <map>

#include <ade/util/assert.hpp>
#include <ade/util/algorithm.hpp>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passes/topological_sort.hpp>

struct Type
{
    enum { OP, WRAP} t;
    static const char *name() { return "Type"; }
};

struct Operation
{
    std::string op;
    static const char *name() { return "Operation"; }
};

struct Wrap
{
    std::shared_ptr<ade::Graph> wrapped;
    std::string op;
    static const char *name() { return "Wrap"; }
};

using TGraph = ade::TypedGraph<Type, Operation, Wrap, ade::passes::TopologicalSortData>;

namespace m
{

// Returns a node handle of created operation
ade::NodeHandle op(TGraph &graph, const std::string &name)
{
    ade::NodeHandle h = graph.createNode();
    graph.metadata(h).set(Type{ Type::OP });
    graph.metadata(h).set(Operation{ name });
    return h;
}

// Wraps a graph into node
ade::NodeHandle wrap(TGraph &graph,
                     std::shared_ptr<ade::Graph> g,
                     const std::string &op)
{
    // All checks are ok, now create a new super node and reconnect it in graph
    ade::NodeHandle super = graph.createNode();
    graph.metadata(super).set(Type{ Type::WRAP });
    graph.metadata(super).set(Wrap{ g, op });
    return super;
}

} // namespace m

void run(ade::Graph &gsrc, int depth = 0)
{
    TGraph g(gsrc);

    ade::passes::PassContext context{gsrc};
    ade::passes::TopologicalSort()(context);

    auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
    for (auto n : sorted)
    {
        Type type = g.metadata(n).get<Type>();
        if (type.t == Type::OP)
        {
           for (int i = 0; i < depth*4; i++) { std::cout << ' '; }
           std::cout << g.metadata(n).get<Operation>().op << std::endl;
        }
        else if (type.t == Type::WRAP)
        {
            Wrap w = g.metadata(n).get<Wrap>();
            for (int i = 0; i < depth*4; i++) { std::cout << ' '; }
            std::cout << w.op << std::endl;

            run(*w.wrapped, depth+1);
        }
        else ADE_ASSERT(false);
    }
}

int main(int argc, char *argv[])
{
    // First, define a graph we plan to nest
    std::shared_ptr<ade::Graph> rms(new ade::Graph);
    TGraph trms(*rms);

    ade::NodeHandle
        foo = m::op(trms, "foo"),
        bar = m::op(trms, "bar"),
        baz = m::op(trms, "baz");

    rms->link(foo, baz);
    rms->link(bar, baz);

    // No build a supergraph using our graph
    ade::Graph super;
    TGraph tsuper(super);

    ade::NodeHandle
        pre  = m::op  (tsuper,      "pre"),
        fbb  = m::wrap(tsuper, rms, "fbb"),
        post = m::op  (tsuper,      "post");

    run(super);

    return 0;
}
