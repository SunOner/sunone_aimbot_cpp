// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file hash.hpp

#ifndef ADE_HASH_HPP
#define ADE_HASH_HPP

#include <cstddef> //size_t

namespace ade
{
namespace util
{

/// Combines hash with seed.
/// Suitable for combining multiple hashes together.
///
/// @param seed
/// @param hash
/// @return Resulting hash
inline std::size_t hash_combine(std::size_t seed, std::size_t val)
{
    // Hash combine formula from boost
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
} // namespace util
} // namespace ade

#endif // ADE_HASH_HPP
