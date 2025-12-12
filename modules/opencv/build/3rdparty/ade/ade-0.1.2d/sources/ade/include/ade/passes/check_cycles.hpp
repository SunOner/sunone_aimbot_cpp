// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file check_cycles.hpp

#ifndef ADE_CHECK_CYCLES_HPP
#define ADE_CHECK_CYCLES_HPP

#include <exception>
#include <string>

#include "ade/passes/pass_base.hpp"

namespace ade
{

namespace passes
{
class CycleFound : public std::exception
{
public:
    virtual const char* what() const noexcept override;
};

struct CheckCycles
{
    void operator()(const PassContext& context) const;
    static std::string name();
};
}
}

#endif // ADE_CHECK_CYCLES_HPP
