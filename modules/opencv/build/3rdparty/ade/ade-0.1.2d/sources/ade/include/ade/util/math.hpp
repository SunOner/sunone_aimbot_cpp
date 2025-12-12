// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file math.hpp

#ifndef ADE_UTIL_MATH_HPP
#define ADE_UTIL_MATH_HPP

#include <type_traits>

#include "ade/util/assert.hpp"

namespace ade
{
namespace util
{
template<typename T>
inline auto is_pow2(T val)
->typename std::enable_if<std::is_integral<T>::value, bool>::type
{
    return (val & (val - 1)) == 0;
}

template<typename T>
inline auto align_size(T size, T align)
->typename std::enable_if<std::is_integral<T>::value, T>::type
{
    ADE_ASSERT(size > 0);
    ADE_ASSERT(align > 0);
    ADE_ASSERT(is_pow2(align));
    return (size + (align - 1)) & ~(align - 1);
}

}
} // namespace ade

#endif // ADE_UTIL_MATH_HPP
