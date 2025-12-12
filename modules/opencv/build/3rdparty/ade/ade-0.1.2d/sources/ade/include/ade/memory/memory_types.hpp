// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file memory_types.hpp

#ifndef ADE_MEMORY_TYPES_HPP
#define ADE_MEMORY_TYPES_HPP

#include "ade/util/md_size.hpp"
#include "ade/util/md_span.hpp"
#include "ade/util/md_view.hpp"

namespace ade
{
namespace memory
{
static const constexpr std::size_t MaxDimensions = 6;
using DynMdSize = util::DynMdSize<MaxDimensions>;
using DynMdSpan = util::DynMdSpan<MaxDimensions>;

template<typename T>
using DynMdView = util::DynMdView<MaxDimensions, T>;

}
}

#endif // ADE_MEMORY_TYPES_HPP
