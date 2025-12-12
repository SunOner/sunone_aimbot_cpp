// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <algorithm>

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/node.hpp>
#include <ade/edge.hpp>
#include <ade/typed_graph.hpp>

using namespace ade;

namespace
{
// Put all types in custom namespace to test ADL for metadata name
namespace Tst
{
struct Foo
{
    int val;
    static const char* name() { return "Foo"; }
};
struct Bar
{
    int val;
};
struct Baz
{
    int val;
    static std::string name() { return "Baz"; }
};
struct Qux
{
    int val;
    static std::string name() { return "Baz"; } // Qux and Baz have the same name
};

const char* getMetadataName(ade::MetadataNameTag<Bar>)
{
    return "Bar";
}
}
}

TEST(TypedGraph, NotUniqueName)
{
    Graph srcGr;
    using TG = TypedGraph<Tst::Foo, Tst::Baz, Tst::Qux>;
    EXPECT_THROW(TG gr(srcGr), std::logic_error);  // Qux and Baz cannot be in one graph, because have the same name
}

TEST(TypedGraph, Simple)
{
    Graph srcGr;
    TypedGraph<Tst::Foo> gr(srcGr);

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

TEST(TypedGraph, EraseMiddleNode)
{
    Graph srcGr;
    TypedGraph<Tst::Foo> gr(srcGr);

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
template<typename T>
void checkMeta3(T& gr)
{
    auto meta = gr.metadata();
    EXPECT_EQ(42,  meta.template get<Tst::Foo>().val);
    EXPECT_EQ(77,  meta.template get<Tst::Bar>().val);
    EXPECT_EQ(123, meta.template get<Tst::Baz>().val);
}

template<typename T>
void checkMeta2(T& gr)
{
    auto meta = gr.metadata();
    EXPECT_EQ(42,  meta.template get<Tst::Foo>().val);
    EXPECT_EQ(77,  meta.template get<Tst::Bar>().val);
}
}

TEST(TypedGraph, Constructors)
{
    Graph srcGr;

    ConstTypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr1(srcGr);
    TypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr2(srcGr);
    gr2.metadata().set(Tst::Foo{42});
    gr2.metadata().set(Tst::Bar{77});
    gr2.metadata().set(Tst::Baz{123});

    {
        ConstTypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr3(gr1);
        checkMeta3(gr3);
    }

    {
        ConstTypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr3(gr2);
        TypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr4(gr2);
        checkMeta3(gr3);
        checkMeta3(gr4);
    }

    {
        ConstTypedGraph<Tst::Foo, Tst::Bar> gr3(gr1);
        checkMeta2(gr3);
    }

    {
        ConstTypedGraph<Tst::Foo, Tst::Bar> gr3(gr2);
        TypedGraph<Tst::Foo, Tst::Bar> gr4(gr2);
        checkMeta2(gr3);
        checkMeta2(gr4);
    }
}

TEST(TypedGraph, TypedMetadata)
{
    Graph srcGr;
    TypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr1(srcGr);
    ConstTypedGraph<Tst::Foo, Tst::Bar, Tst::Baz> gr2(gr1);

    ASSERT_FALSE(gr1.metadata().contains<Tst::Foo>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Baz>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Foo>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Baz>());


    gr1.metadata().set(Tst::Foo{42});
    ASSERT_TRUE( gr1.metadata().contains<Tst::Foo>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Baz>());
    ASSERT_TRUE( gr2.metadata().contains<Tst::Foo>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Baz>());

    ASSERT_EQ(42, gr1.metadata().get<Tst::Foo>().val);
    ASSERT_EQ(42, gr2.metadata().get<Tst::Foo>().val);


    gr1.metadata().set(Tst::Bar{77});
    ASSERT_TRUE( gr1.metadata().contains<Tst::Foo>());
    ASSERT_TRUE( gr1.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Baz>());
    ASSERT_TRUE( gr2.metadata().contains<Tst::Foo>());
    ASSERT_TRUE( gr2.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Baz>());

    ASSERT_EQ(42,  gr1.metadata().get<Tst::Foo>().val);
    ASSERT_EQ(77,  gr1.metadata().get<Tst::Bar>().val);
    ASSERT_EQ(123, gr1.metadata().get(Tst::Baz{123}).val);
    ASSERT_EQ(42,  gr2.metadata().get<Tst::Foo>().val);
    ASSERT_EQ(77,  gr2.metadata().get<Tst::Bar>().val);
    ASSERT_EQ(123, gr2.metadata().get(Tst::Baz{123}).val);


    gr1.metadata().erase<Tst::Bar>();
    ASSERT_TRUE( gr1.metadata().contains<Tst::Foo>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr1.metadata().contains<Tst::Baz>());
    ASSERT_TRUE( gr2.metadata().contains<Tst::Foo>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Bar>());
    ASSERT_FALSE(gr2.metadata().contains<Tst::Baz>());
}
