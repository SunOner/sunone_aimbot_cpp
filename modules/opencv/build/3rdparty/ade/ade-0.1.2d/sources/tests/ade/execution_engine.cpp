// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/node.hpp>
#include <ade/edge.hpp>
#include <ade/passes/pass_base.hpp>
#include <ade/execution_engine/execution_engine.hpp>

#include <ade/util/zip_range.hpp>

using namespace ade;

TEST(ExecutionEngine, PassOrder)
{
    ExecutionEngine engine;
    Graph gr;

    engine.addPassStage("stage1");
    engine.addPassStage("stage3");
    engine.addPassStage("stage2", "stage1");

    EXPECT_FALSE(engine.passStages().empty());
    for (auto it: indexed(engine.passStages()))
    {
        auto i = util::index(it);
        auto& stage = util::value(it);
        if(0 == i)
        {
            EXPECT_EQ("stage1", stage);
        }
        else if(1 == i)
        {
            EXPECT_EQ("stage2", stage);
        }
        else if(2 == i)
        {
            EXPECT_EQ("stage3", stage);
        }
        else
        {
            FAIL();
        }
    }

    bool passCalled1 = false;
    bool passCalled2 = false;
    bool passCalled3 = false;
    bool passCalled4 = false;

    engine.addPass("stage1", "pass1",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_FALSE(passCalled1);
        EXPECT_FALSE(passCalled2);
        EXPECT_FALSE(passCalled3);
        EXPECT_FALSE(passCalled4);
        passCalled1 = true;
    });

    engine.addPass("stage2", "pass2",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_TRUE(passCalled1);
        EXPECT_FALSE(passCalled2);
        EXPECT_FALSE(passCalled3);
        EXPECT_FALSE(passCalled4);
        passCalled2 = true;
    });

    engine.addPass("stage2", "pass3",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_TRUE(passCalled1);
        EXPECT_TRUE(passCalled2);
        EXPECT_FALSE(passCalled3);
        EXPECT_FALSE(passCalled4);
        passCalled3 = true;
    });

    engine.addPass("stage3", "pass4",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_TRUE(passCalled1);
        EXPECT_TRUE(passCalled2);
        EXPECT_TRUE(passCalled3);
        EXPECT_FALSE(passCalled4);
        passCalled4 = true;
    });

    engine.runPasses(gr);

    EXPECT_TRUE(passCalled1);
    EXPECT_TRUE(passCalled2);
    EXPECT_TRUE(passCalled3);
    EXPECT_TRUE(passCalled4);
}

TEST(ExecutionEngine, PassCallbacks)
{
    ExecutionEngine engine;
    Graph gr;

    engine.addPassStage("stage1");
    engine.addPassStage("stage2");

    bool passCalled1 = false;
    bool passCalled2 = false;
    bool passCalled3 = false;
    int prePassCallCount = 0;
    int postPassCallCount = 0;
    int prePassCallCount2 = 0;
    int postPassCallCount2 = 0;

    engine.addPrePassCallback([&](const ExecutionEngine::PassDesc& desc, const passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_EQ(prePassCallCount, postPassCallCount);
        if (0 == prePassCallCount)
        {
            EXPECT_EQ("stage1", desc.stage);
            EXPECT_EQ("pass1", desc.pass);
        }
        if (1 == prePassCallCount)
        {
            EXPECT_EQ("stage1", desc.stage);
            EXPECT_EQ("pass2", desc.pass);
        }
        if (2 == prePassCallCount)
        {
            EXPECT_EQ("stage2", desc.stage);
            EXPECT_EQ("pass3", desc.pass);
        }
        ++prePassCallCount;
    });
    engine.addPrePassCallback([&](const ExecutionEngine::PassDesc& /*desc*/, const passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_EQ(prePassCallCount - 1, prePassCallCount2);
        ++prePassCallCount2;
    });

    engine.addPostPassCallback([&](const ExecutionEngine::PassDesc& desc, const passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_EQ(prePassCallCount - 1, postPassCallCount);
        if (0 == postPassCallCount)
        {
            EXPECT_EQ("stage1", desc.stage);
            EXPECT_EQ("pass1", desc.pass);
        }
        if (1 == postPassCallCount)
        {
            EXPECT_EQ("stage1", desc.stage);
            EXPECT_EQ("pass2", desc.pass);
        }
        if (2 == postPassCallCount)
        {
            EXPECT_EQ("stage2", desc.stage);
            EXPECT_EQ("pass3", desc.pass);
        }
        ++postPassCallCount;
    });
    engine.addPostPassCallback([&](const ExecutionEngine::PassDesc& /*desc*/, const passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_EQ(postPassCallCount - 1, postPassCallCount2);
        ++postPassCallCount2;
    });

    engine.addPass("stage1", "pass1",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_FALSE(passCalled1);
        EXPECT_FALSE(passCalled2);
        EXPECT_FALSE(passCalled3);
        passCalled1 = true;
    });

    engine.addPass("stage1", "pass2",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_TRUE(passCalled1);
        EXPECT_FALSE(passCalled2);
        EXPECT_FALSE(passCalled3);
        passCalled2 = true;
    });

    engine.addPass("stage2", "pass3",[&](passes::PassContext& context)
    {
        EXPECT_EQ(&gr, &context.graph);
        EXPECT_TRUE(passCalled1);
        EXPECT_TRUE(passCalled2);
        EXPECT_FALSE(passCalled3);
        passCalled3 = true;
    });

    engine.runPasses(gr);

    EXPECT_TRUE(passCalled1);
    EXPECT_TRUE(passCalled2);
    EXPECT_TRUE(passCalled3);
    EXPECT_EQ(3, prePassCallCount);
    EXPECT_EQ(3, postPassCallCount);
    EXPECT_EQ(3, prePassCallCount2);
    EXPECT_EQ(3, postPassCallCount2);
}

namespace
{
struct TestPassState
{
    int testPassCalled = 0;
    int nodeCreatedCalled = 0;
    int nodeAboutToBeDestroyedCalled = 0;
    int edgeCreatedCalled = 0;
    int edgeAboutToBeDestroyedCalled = 0;
    int edgeAboutToBeRelinkedCalled = 0;
    bool testPassCheckerRet = true;
};

struct TestPass final
{
    TestPassState* state;

    void operator()(passes::PassContext&)
    {
        ++state->testPassCalled;
    }
};

struct TestPassChecker final
{
    TestPassState* state;

    bool nodeCreated(const Graph& /*graph*/, const NodeHandle& /*node*/)
    {
        ++state->nodeCreatedCalled;
        return state->testPassCheckerRet;
    }
    bool nodeAboutToBeDestroyed(const Graph& /*graph*/, const NodeHandle& /*node*/)
    {
        ++state->nodeAboutToBeDestroyedCalled;
        return state->testPassCheckerRet;
    }
    bool edgeCreated(const Graph& /*graph*/, const EdgeHandle& /*edge*/)
    {
        ++state->edgeCreatedCalled;
        return state->testPassCheckerRet;
    }
    bool edgeAboutToBeDestroyed(const Graph& /*graph*/, const EdgeHandle& /*edge*/)
    {
        ++state->edgeAboutToBeDestroyedCalled;
        return state->testPassCheckerRet;
    }
    bool edgeAboutToBeRelinked(const Graph& /*graph*/,
                               const EdgeHandle& /*edge*/,
                               const NodeHandle& /*newSrcNode*/,
                               const NodeHandle& /*newDstNode*/)
    {
        ++state->edgeAboutToBeRelinkedCalled;
        return state->testPassCheckerRet;
    }
};
}

TEST(ExecutionEngine, LazyPass)
{
    TestPassState state;
    std::string passName = "lazy_test_pass";
    auto reset = [&]()
    {
        state = TestPassState{};
    };

    {
        ExecutionEngine engine;
        Graph gr;

        engine.addLazyPass(passName, TestPass{&state}, TestPassChecker{&state});
        engine.addPassStage("stage");
        engine.addPass("stage", "pass1", [](passes::PassContext&){}, {passName});
        engine.addPass("stage", "pass2", [](passes::PassContext&){}, std::array<std::string, 1>{{passName}});

        reset();
        engine.runPasses(gr);
        EXPECT_EQ(1, state.testPassCalled);
        state.testPassCalled = 0;
        engine.runPasses(gr);
        EXPECT_EQ(1, state.testPassCalled);
    }
    {
        ExecutionEngine engine;
        Graph gr;

        NodeHandle node1;
        NodeHandle node2;
        NodeHandle node3;
        EdgeHandle edge;

        reset();
        state.testPassCheckerRet = false;
        engine.addLazyPass(passName, TestPass{&state}, TestPassChecker{&state});
        engine.addPassStage("stage");
        engine.addPass("stage", "pass1", [&](passes::PassContext&)
        {
            // Check dependent pass was called for the first pass
            EXPECT_EQ(1, state.testPassCalled);
            state.testPassCalled = 0;
        }, {passName});
        engine.addPass("stage", "pass2", [&](passes::PassContext& ctx)
        {
            // Graph wasn't changed in previous pass so dependent pass shouldn't be called
            EXPECT_EQ(0, state.testPassCalled);
            node1 = ctx.graph.createNode();
            node2 = ctx.graph.createNode();
            node3 = ctx.graph.createNode();
        }, {passName});
        engine.addPass("stage", "pass3", [&](passes::PassContext& ctx)
        {
            EXPECT_EQ(1, state.testPassCalled);
            state.testPassCalled = 0;
            edge = ctx.graph.link(node1, node2);
        }, {passName});
        engine.addPass("stage", "pass4", [&](passes::PassContext& ctx)
        {
            EXPECT_EQ(1, state.testPassCalled);
            state.testPassCalled = 0;
            ctx.graph.link(node3, edge);
        }, {passName});
        engine.addPass("stage", "pass5", [&](passes::PassContext& ctx)
        {
            EXPECT_EQ(1, state.testPassCalled);
            state.testPassCalled = 0;
            ctx.graph.erase(edge);
        }, {passName});
        engine.addPass("stage", "pass6", [&](passes::PassContext& ctx)
        {
            EXPECT_EQ(1, state.testPassCalled);
            state.testPassCalled = 0;
            ctx.graph.erase(node2);
        }, {passName});

        engine.runPasses(gr);

        // We dont call listeners if pass was already invalidated
        // so it will be called only once on the first node creation
        EXPECT_EQ(1, state.nodeCreatedCalled);
        EXPECT_EQ(1, state.nodeAboutToBeDestroyedCalled);
        EXPECT_EQ(1, state.edgeCreatedCalled);
        EXPECT_EQ(1, state.edgeAboutToBeDestroyedCalled);
        EXPECT_EQ(1, state.edgeAboutToBeRelinkedCalled);
    }
    {
        ExecutionEngine engine;
        Graph gr;

        engine.addLazyPass(passName, TestPass{&state}, TestPassChecker{&state});

        reset();
        engine.runPasses(gr);
        EXPECT_EQ(0, state.testPassCalled);
    }
    {
        ExecutionEngine engine;
        Graph gr;

        engine.addLazyPass(passName, TestPass{&state}, TestPassChecker{&state});
        engine.addExecutableDependency(passName);

        reset();
        engine.runPasses(gr);
        EXPECT_EQ(1, state.testPassCalled);
    }
}
