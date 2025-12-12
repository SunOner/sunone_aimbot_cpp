// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file filter_range.hpp

#ifndef ADE_UTIL_FILTER_RANGE_HPP
#define ADE_UTIL_FILTER_RANGE_HPP

#include "ade/util/tuple.hpp"
#include "ade/util/range.hpp"
#include "ade/util/assert.hpp"
#include "ade/util/iota_range.hpp"
#include "ade/util/range_iterator.hpp"

namespace ade
{
namespace util
{

inline namespace Range
{

template<typename PrevRange, typename Filter>
struct FilterRange : public IterableRange<FilterRange<PrevRange, Filter>>
{
    PrevRange prevRange;
    Filter filter; // TODO: implement size optimization for empty objects

    FilterRange() = default;
    FilterRange(const FilterRange&) = default;
    FilterRange(FilterRange&&) = default;
    FilterRange& operator=(const FilterRange&) = default;
    FilterRange& operator=(FilterRange&&) = default;
    template<typename PR, typename F>
    FilterRange(PR&& pr, F&& f):
        prevRange(std::forward<PR>(pr)),
        filter(std::forward<F>(f))
    {
        while (!prevRange.empty() && !filter(prevRange.front()))
        {
            prevRange.popFront();
        }
    }

    bool empty() const
    {
        return prevRange.empty();
    }

    void popFront()
    {
        ADE_ASSERT(!empty());
        prevRange.popFront();
        while (!prevRange.empty() && !filter(prevRange.front()))
        {
            prevRange.popFront();
        }
    }

    auto front() -> decltype(this->prevRange.front())
    {
        ADE_ASSERT(!empty());
        return prevRange.front();
    }

    auto front() const -> decltype(this->prevRange.front())
    {
        ADE_ASSERT(!empty());
        return prevRange.front();
    }
};

template<typename Filter, typename PrevRange>
FilterRange<PrevRange, Filter> filter(PrevRange&& prevRange, Filter&& filter_)
{
    return FilterRange<PrevRange, Filter>(std::forward<PrevRange>(prevRange), std::forward<Filter>(filter_));
}

template<typename Filter, typename PrevRange>
FilterRange<PrevRange, Filter> filter(PrevRange&& prevRange)
{
    return FilterRange<PrevRange, Filter>(std::forward<PrevRange>(prevRange), Filter());
}

}
}
} // namespace ade

#endif // ADE_UTIL_FILTER_RANGE_HPP
