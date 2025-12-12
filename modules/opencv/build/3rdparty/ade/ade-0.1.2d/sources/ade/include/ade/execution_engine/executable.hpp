// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file executable.hpp

#ifndef ADE_EXECUTABLE_HPP
#define ADE_EXECUTABLE_HPP

namespace ade
{
class Executable
{
public:
    virtual ~Executable() = default;
    virtual void run() = 0;
    virtual void run(void *opaque) = 0;      // WARNING: opaque may be accessed from various threads.

    virtual void runAsync() = 0;
    virtual void runAsync(void *opaque) = 0; // WARNING: opaque may be accessed from various threads.

    virtual void wait() = 0;
};
}

#endif // ADE_EXECUTABLE_HPP
