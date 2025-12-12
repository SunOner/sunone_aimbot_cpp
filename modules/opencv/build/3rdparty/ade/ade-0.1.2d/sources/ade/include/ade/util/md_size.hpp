// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file md_size.hpp

#ifndef ADE_UTIL_MD_SIZE_HPP
#define ADE_UTIL_MD_SIZE_HPP

#include <algorithm>
#include <array>
#include <initializer_list>

#include "ade/util/assert.hpp"
#include "ade/util/iota_range.hpp"
#include "ade/util/checked_cast.hpp"

namespace ade
{
namespace util
{

/// Dinamically sized arbitrary dimensional size
template <std::size_t MaxDimensions>
struct DynMdSize final
{
    using SizeT = int;
    std::array<SizeT, MaxDimensions> sizes;
    std::size_t dims_cnt = 0;

    DynMdSize() = default;

    DynMdSize(std::initializer_list<SizeT> d):
        dims_cnt(util::checked_cast<decltype(this->dims_cnt)>(d.size()))
    {
        ADE_ASSERT(d.size() <= MaxDimensions);
        std::copy(d.begin(), d.end(), sizes.begin());
    }

    DynMdSize(const DynMdSize&) = default;
    DynMdSize& operator=(const DynMdSize&) = default;

    bool operator==(const DynMdSize& other) const
    {
        if (dims_count() != other.dims_count())
        {
            return false;
        }

        for (auto i: util::iota(dims_count()))
        {
            if ((*this)[i] != other[i])
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const DynMdSize& other) const
    {
        return !(*this == other);
    }

    SizeT& operator[](std::size_t index)
    {
        ADE_ASSERT(index < dims_count());
        return sizes[index];
    }

    const SizeT& operator[](std::size_t index) const
    {
        ADE_ASSERT(index < dims_count());
        return sizes[index];
    }

    SizeT* data()
    {
        return sizes.data();
    }

    const SizeT* data() const
    {
        return sizes.data();
    }

    std::size_t dims_count() const
    {
        return dims_cnt;
    }

    void redim(std::size_t count)
    {
        ADE_ASSERT(count <= MaxDimensions);
        dims_cnt = count;
    }

    auto begin()
    ->decltype(this->sizes.begin())
    {
        return sizes.begin();
    }

    auto end()
    ->decltype(this->sizes.begin() + this->dims_count())
    {
        return sizes.begin() + dims_count();
    }

    auto begin() const
    ->decltype(this->sizes.begin())
    {
        return sizes.begin();
    }

    auto end() const
    ->decltype(this->sizes.begin() + this->dims_count())
    {
        return sizes.begin() + dims_count();
    }
};
}
} // namespace ade

#endif // ADE_UTIL_MD_SIZE_HPP
