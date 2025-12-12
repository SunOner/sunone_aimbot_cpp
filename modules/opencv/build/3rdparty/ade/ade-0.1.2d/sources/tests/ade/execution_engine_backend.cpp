// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/node.hpp>
#include <ade/edge.hpp>
#include <ade/passes/pass_base.hpp>
#include <ade/execution_engine/execution_engine.hpp>
#include <ade/execution_engine/executable.hpp>
#include <ade/execution_engine/backend.hpp>

#include <ade/util/assert.hpp>

using namespace ade;

namespace
{
class TestBackend;

class TestBackendExec : public BackendExecutable
{
public:
    bool runCalled          = false;
    bool runAsyncCalled     = false;
    bool waitCalled         = false;
    bool cancelCalled       = false;
    bool runOpaqCalled      = false;
    bool runAsyncOpaqCalled = false;
    int lastVal             = 0;

    bool runThroughException = false;
    bool runAsyncThroughException = false;
    bool waitThroughException = false;

    TestBackend* parent = nullptr;

    TestBackendExec(TestBackend* e);

    ~TestBackendExec();

    virtual void run() override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_FALSE(runAsyncCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        runCalled = true;
        if (runThroughException)
            throw std::exception();
    }

    virtual void runAsync() override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_FALSE(runAsyncCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        runAsyncCalled = true;
        if (runThroughException)
            throw std::exception();
    }

    virtual void wait() override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_TRUE(runAsyncCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        waitCalled = true;
        if (waitThroughException)
            throw std::exception();
    }

    virtual void cancel() override
    {
        EXPECT_TRUE(runCalled || runAsyncCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        cancelCalled = true;
    }
};

class TestBackendExecOpaq final : public TestBackendExec
{
public:
    using TestBackendExec::TestBackendExec;

    virtual void run() override
    {
        FAIL() << "TestBackendExecOpaq::run() shouldn't be called";
    }

    virtual void runAsync() override
    {
        FAIL() << "TestBackendExecOpaq::runAsync() shouldn't be called";
    }

    virtual void run(void *opaque) override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_FALSE(runOpaqCalled);
        EXPECT_FALSE(runAsyncCalled);
        EXPECT_FALSE(runAsyncOpaqCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        runOpaqCalled = true;
        lastVal = *static_cast<int*>(opaque);
    }

    virtual void runAsync(void* opaque) override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_FALSE(runOpaqCalled);
        EXPECT_FALSE(runAsyncCalled);
        EXPECT_FALSE(runAsyncOpaqCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        runAsyncOpaqCalled = true;
        lastVal = *static_cast<int*>(opaque);
    }

    virtual void wait() override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_FALSE(runOpaqCalled);
        EXPECT_FALSE(runAsyncCalled);
        EXPECT_TRUE (runAsyncOpaqCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        waitCalled = true;
    }

    virtual void cancel() override
    {
        EXPECT_FALSE(runCalled);
        EXPECT_FALSE(runAsyncCalled);
        EXPECT_TRUE (runOpaqCalled || runAsyncOpaqCalled);
        EXPECT_FALSE(waitCalled);
        EXPECT_FALSE(cancelCalled);
        cancelCalled = true;
    }
};

class TestBackend final : public ExecutionBackend
{
public:
    bool setupCalled = false;
    bool emptyExec = false;
    bool createOpaq = false;
    TestBackendExec* exec = nullptr;

    int lastVal = 0;

    virtual void setupExecutionEngine(ExecutionEngineSetupContext& /*context*/) override
    {
        EXPECT_FALSE(setupCalled);
        setupCalled = true;
    }
    virtual std::unique_ptr<BackendExecutable> createExecutable(const Graph& /*graph*/) override
    {
        EXPECT_TRUE(setupCalled);
        EXPECT_EQ(nullptr, exec);
        std::unique_ptr<BackendExecutable> retval;
        if (!emptyExec)
        {
            retval.reset(createOpaq ? new TestBackendExecOpaq(this) : new TestBackendExec(this));
        }
        return retval;
    }
};

TestBackendExec::TestBackendExec(TestBackend* e):
    parent(e)
{
    ADE_ASSERT(nullptr != parent);
    ADE_ASSERT(nullptr == parent->exec);
    parent->exec = this;
}

TestBackendExec::~TestBackendExec()
{
    ADE_ASSERT(nullptr != parent);
    parent->exec = nullptr;
}

template<typename T>
inline std::unique_ptr<T> wrap_unique(T* ptr)
{
    return std::unique_ptr<T>(ptr);
}

} // namespace

TEST(ExecutionEngine, BackendsSetup)
{
    ExecutionEngine engine;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));

    EXPECT_FALSE(backend1->setupCalled);
    EXPECT_FALSE(backend2->setupCalled);

    engine.setupBackends();

    EXPECT_TRUE(backend1->setupCalled);
    EXPECT_TRUE(backend2->setupCalled);
}

TEST(ExecutionEngine, BackendsRun)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);

    exec->run();

    EXPECT_TRUE(backend1->exec->runCalled);
    EXPECT_FALSE(backend1->exec->runAsyncCalled);
    EXPECT_FALSE(backend1->exec->waitCalled);

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE(backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->waitCalled);
}

TEST(ExecutionEngine, BackendsRunOpaq)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend; backend1->createOpaq = true;
    auto backend2 = new TestBackend; backend2->createOpaq = true;
    auto backend3 = new TestBackend; backend3->createOpaq = false;

    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);

    int value = 42;
    exec->run(&value);

    EXPECT_FALSE(backend1->exec->runCalled);
    EXPECT_TRUE (backend1->exec->runOpaqCalled);
    EXPECT_FALSE(backend1->exec->runAsyncCalled);
    EXPECT_FALSE(backend1->exec->runAsyncOpaqCalled);
    EXPECT_FALSE(backend1->exec->waitCalled);

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_FALSE(backend2->exec->runOpaqCalled);
    EXPECT_FALSE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE (backend2->exec->runAsyncOpaqCalled);
    EXPECT_TRUE (backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_FALSE(backend3->exec->runOpaqCalled);
    EXPECT_TRUE (backend3->exec->runAsyncCalled);
    EXPECT_FALSE(backend3->exec->runAsyncOpaqCalled);
    EXPECT_TRUE (backend3->exec->waitCalled);

    const int expected = value;
    EXPECT_EQ(expected, backend1->exec->lastVal);
    EXPECT_EQ(expected, backend2->exec->lastVal);
}

TEST(ExecutionEngine, BackendsRunAsync)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);

    exec->runAsync();

    EXPECT_FALSE(backend1->exec->runCalled);
    EXPECT_TRUE(backend1->exec->runAsyncCalled);
    EXPECT_FALSE(backend1->exec->waitCalled);

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_FALSE(backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_FALSE(backend3->exec->waitCalled);

    exec->wait();

    EXPECT_FALSE(backend1->exec->runCalled);
    EXPECT_TRUE(backend1->exec->runAsyncCalled);
    EXPECT_TRUE(backend1->exec->waitCalled);

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE(backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->waitCalled);
}

TEST(ExecutionEngine, BackendsRunAsyncOpaq)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend; backend1->createOpaq = true;
    auto backend2 = new TestBackend; backend2->createOpaq = true;
    auto backend3 = new TestBackend; backend3->createOpaq = false;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);

    int value = 42;
    exec->runAsync(&value);

    EXPECT_FALSE(backend1->exec->runCalled);
    EXPECT_FALSE(backend1->exec->runOpaqCalled);
    EXPECT_FALSE(backend1->exec->runAsyncCalled);
    EXPECT_TRUE (backend1->exec->runAsyncOpaqCalled);
    EXPECT_FALSE(backend1->exec->waitCalled);

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_FALSE(backend2->exec->runOpaqCalled);
    EXPECT_FALSE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE (backend2->exec->runAsyncOpaqCalled);
    EXPECT_FALSE(backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_FALSE(backend3->exec->runOpaqCalled);
    EXPECT_TRUE (backend3->exec->runAsyncCalled);
    EXPECT_FALSE(backend3->exec->runAsyncOpaqCalled);
    EXPECT_FALSE(backend3->exec->waitCalled);

    exec->wait();

    EXPECT_FALSE(backend1->exec->runCalled);
    EXPECT_FALSE(backend1->exec->runOpaqCalled);
    EXPECT_FALSE(backend1->exec->runAsyncCalled);
    EXPECT_TRUE (backend1->exec->runAsyncOpaqCalled);
    EXPECT_TRUE (backend1->exec->waitCalled);

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_FALSE(backend2->exec->runOpaqCalled);
    EXPECT_FALSE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE (backend2->exec->runAsyncOpaqCalled);
    EXPECT_TRUE (backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_FALSE(backend3->exec->runOpaqCalled);
    EXPECT_TRUE (backend3->exec->runAsyncCalled);
    EXPECT_FALSE(backend3->exec->runAsyncOpaqCalled);
    EXPECT_TRUE (backend3->exec->waitCalled);

    const int expected = value;
    EXPECT_EQ(expected, backend1->exec->lastVal);
    EXPECT_EQ(expected, backend2->exec->lastVal);
}

TEST(ExecutionEngine, BackendsRunSome)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    backend2->emptyExec = true;

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_EQ(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);

    exec->run();

    EXPECT_TRUE(backend1->exec->runCalled);
    EXPECT_FALSE(backend1->exec->runAsyncCalled);
    EXPECT_FALSE(backend1->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->waitCalled);
}

TEST(ExecutionEngine, BackendsRunAsyncSome)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    backend1->emptyExec = true;

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_EQ(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);

    exec->runAsync();

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_FALSE(backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_FALSE(backend3->exec->waitCalled);

    exec->wait();

    EXPECT_FALSE(backend2->exec->runCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE(backend2->exec->waitCalled);

    EXPECT_FALSE(backend3->exec->runCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->waitCalled);
}

TEST(ExecutionEngine, BackendsRunNone)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));

    backend1->emptyExec = true;
    backend2->emptyExec = true;
    backend3->emptyExec = true;

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_EQ(nullptr, exec);
    ASSERT_EQ(nullptr, backend1->exec);
    ASSERT_EQ(nullptr, backend2->exec);
    ASSERT_EQ(nullptr, backend3->exec);
}

//----------------------------------------------CANCEL------------------------------------

TEST(ExecutionEngine, BackendsCancel_MainRunThrows)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;

    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    backend1->exec->runThroughException = true;

    ASSERT_THROW(exec->run(), std::exception);

    EXPECT_TRUE(backend1->exec->runCalled);
    EXPECT_FALSE(backend1->exec->cancelCalled);

    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE(backend2->exec->cancelCalled);
}

TEST(ExecutionEngine, BackendsCancel_MiddleRunThrows)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    auto backend4 = new TestBackend;

    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));
    engine.addBackend(wrap_unique(backend4));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);
    ASSERT_NE(nullptr, backend4->exec);
    backend3->exec->runThroughException = true;

    ASSERT_THROW(exec->run(), std::exception);

    EXPECT_TRUE(backend4->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_FALSE(backend2->exec->runAsyncCalled);
    EXPECT_FALSE(backend1->exec->runCalled);

    EXPECT_TRUE(backend4->exec->cancelCalled);
    EXPECT_FALSE(backend3->exec->cancelCalled);
    EXPECT_FALSE(backend2->exec->cancelCalled);
    EXPECT_FALSE(backend1->exec->cancelCalled);
}

TEST(ExecutionEngine, BackendsCancel_RunWaitThrows)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    auto backend4 = new TestBackend;

    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));
    engine.addBackend(wrap_unique(backend4));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);
    ASSERT_NE(nullptr, backend4->exec);
    backend3->exec->waitThroughException = true;

    ASSERT_THROW(exec->run(), std::exception);

    EXPECT_TRUE(backend4->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE(backend1->exec->runCalled);

    EXPECT_FALSE(backend4->exec->waitCalled);
    EXPECT_TRUE(backend3->exec->waitCalled);
    EXPECT_TRUE(backend2->exec->waitCalled);
    EXPECT_FALSE(backend1->exec->waitCalled);

    EXPECT_TRUE(backend4->exec->cancelCalled);
    EXPECT_FALSE(backend3->exec->cancelCalled);
    EXPECT_FALSE(backend2->exec->cancelCalled);
    EXPECT_FALSE(backend1->exec->cancelCalled);
}

TEST(ExecutionEngine, BackendsCancel_WaitThrows)
{
    ExecutionEngine engine;
    Graph gr;

    auto backend1 = new TestBackend;
    auto backend2 = new TestBackend;
    auto backend3 = new TestBackend;
    auto backend4 = new TestBackend;

    engine.addBackend(wrap_unique(backend1));
    engine.addBackend(wrap_unique(backend2));
    engine.addBackend(wrap_unique(backend3));
    engine.addBackend(wrap_unique(backend4));

    engine.setupBackends();

    auto exec = engine.createExecutable(gr);
    ASSERT_NE(nullptr, exec);
    ASSERT_NE(nullptr, backend1->exec);
    ASSERT_NE(nullptr, backend2->exec);
    ASSERT_NE(nullptr, backend3->exec);
    ASSERT_NE(nullptr, backend4->exec);
    backend3->exec->waitThroughException = true;

    exec->runAsync();

    EXPECT_TRUE(backend4->exec->runAsyncCalled);
    EXPECT_TRUE(backend3->exec->runAsyncCalled);
    EXPECT_TRUE(backend2->exec->runAsyncCalled);
    EXPECT_TRUE(backend1->exec->runAsyncCalled);

    ASSERT_THROW(exec->wait(), std::exception);

    EXPECT_TRUE(backend1->exec->waitCalled);
    EXPECT_TRUE(backend2->exec->waitCalled);
    EXPECT_TRUE(backend3->exec->waitCalled);
    EXPECT_FALSE(backend4->exec->waitCalled);

    EXPECT_FALSE(backend1->exec->cancelCalled);
    EXPECT_FALSE(backend2->exec->cancelCalled);
    EXPECT_FALSE(backend3->exec->cancelCalled);
    EXPECT_TRUE(backend4->exec->cancelCalled);
}
