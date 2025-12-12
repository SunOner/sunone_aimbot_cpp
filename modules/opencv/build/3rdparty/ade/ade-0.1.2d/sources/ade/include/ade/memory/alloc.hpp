// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file alloc.hpp

#ifndef ADE_ALLOC_HPP
#define ADE_ALLOC_HPP

#include <cstddef> //size_t

namespace ade
{
void* aligned_alloc(std::size_t size, std::size_t alignment);
void aligned_free(void* ptr);

}

#endif // ADE_ALLOC_HPP
