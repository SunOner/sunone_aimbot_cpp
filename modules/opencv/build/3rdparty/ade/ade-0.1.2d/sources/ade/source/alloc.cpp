// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/memory/alloc.hpp"

#if defined(_WIN32) || defined(__ANDROID__) || defined(ANDROID)
#include <malloc.h>
#else
#include <algorithm>
#include <stdlib.h>
#endif

#include "ade/util/math.hpp"
#include "ade/util/assert.hpp"

namespace ade
{

void* aligned_alloc(std::size_t size, std::size_t alignment)
{
    ADE_ASSERT(util::is_pow2(alignment));
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__ANDROID__) || defined(ANDROID)
    return memalign(alignment, size);
#else
    void* ret = nullptr;
    auto res = posix_memalign(&ret, std::max(sizeof(void*), alignment), size);
    (void)res;
    return ret;
#endif
}

void aligned_free(void* ptr)
{
#if defined(_WIN32)
    return _aligned_free(ptr);
#else
    return free(ptr);
#endif
}

}
