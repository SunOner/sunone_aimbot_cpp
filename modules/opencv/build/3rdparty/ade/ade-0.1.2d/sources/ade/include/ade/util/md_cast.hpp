// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file md_cast.hpp

#ifndef ADE_UTIL_MD_CAST_HPP
#define ADE_UTIL_MD_CAST_HPP

namespace ade
{
namespace util
{
// TODO: find a proper place for this
constexpr static const std::size_t MaxDimensions = 6;

namespace detail
{
template<typename Target>
struct md_cast_helper; // Undefined
}

template<typename Dst, typename Src>
Dst md_cast(const Src& src)
{
    return detail::md_cast_helper<Dst>(src);
}
}
} // namespace ade

#endif // ADE_UTIL_MD_CAST_HPP
