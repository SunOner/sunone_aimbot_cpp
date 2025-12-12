// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file backend.hpp

#ifndef ADE_BACKEND_HPP
#define ADE_BACKEND_HPP

#include <memory>

namespace ade
{

class Graph;
class Executable;
class ExecutionEngineSetupContext;

class BackendExecutable;

class ExecutionBackend
{
public:
    virtual ~ExecutionBackend() = default;

    virtual void setupExecutionEngine(ExecutionEngineSetupContext& engine) = 0;
    virtual std::unique_ptr<BackendExecutable> createExecutable(const Graph& graph) = 0;
};

class BackendExecutable
{
protected:
    // Backward-compatibility stubs
    virtual void run()      {};                    // called by run(void*)
    virtual void runAsync() {};                    // called by runAsync(void*)

public:
    virtual ~BackendExecutable() = default;

    virtual void run(void *opaque);           // Triggered by ADE engine
    virtual void runAsync(void *opaque);      // Triggered by ADE engine

    virtual void wait() = 0;
    virtual void cancel() = 0;
};
}

#endif // ADE_BACKEND_HPP
