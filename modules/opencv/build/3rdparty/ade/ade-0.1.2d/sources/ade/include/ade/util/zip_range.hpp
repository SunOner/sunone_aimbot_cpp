// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file zip_range.hpp

#ifndef ADE_UTIL_ZIP_RANGE_HPP
#define ADE_UTIL_ZIP_RANGE_HPP

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

template<typename... Ranges>
struct ZipRange : public IterableRange<ZipRange<Ranges...>>
{
    using tuple_t = decltype(std::make_tuple(toRange(std::declval<Ranges>())...));
    tuple_t ranges;

    ZipRange() = default;
    ZipRange(const ZipRange&) = default;
    ZipRange(ZipRange&&) = default;
    ZipRange& operator=(const ZipRange&) = default;
    ZipRange& operator=(ZipRange&&) = default;
    ZipRange(Ranges&& ...r): ranges{toRange(std::forward<Ranges>(r))...} {}

    template<std::size_t... S>
    auto unpackTuple(details::Seq<S...>) ->
    decltype(tuple_remove_rvalue_refs(std::get<S>(ranges).front()...))
    {
        return tuple_remove_rvalue_refs(std::get<S>(ranges).front()...);
    }

    template<std::size_t... S>
    auto unpackTuple(details::Seq<S...>) const ->
    decltype(tuple_remove_rvalue_refs(std::get<S>(ranges).front()...))
    {
        return tuple_remove_rvalue_refs(std::get<S>(ranges).front()...);
    }

    bool empty() const
    {
        ::ade::util::details::RangeChecker checker;
        tupleForeach(ranges, checker);
        return checker.empty;
    }

    void popFront()
    {
        ADE_ASSERT(!empty());
        tupleForeach(ranges, ade::util::details::RangeIncrementer());
    }

    auto front()->decltype(this->unpackTuple(details::gen_t<sizeof...(Ranges)>{}))
    {
        ADE_ASSERT(!empty());
        return unpackTuple(details::gen_t<sizeof...(Ranges)>{});
    }

    auto front() const->decltype(this->unpackTuple(details::gen_t<sizeof...(Ranges)>{}))
    {
        ADE_ASSERT(!empty());
        return unpackTuple(details::gen_t<sizeof...(Ranges)>{});
    }


};

template<typename... Ranges>
inline ZipRange<Ranges...> zip(Ranges&&... ranges)
{
    return {std::forward<Ranges>(ranges)...};
}

template<typename... Containers>
inline auto indexed(Containers&&... conts) ->
decltype(zip(iota<std::size_t>(), std::forward<Containers>(conts)...))
{
    return zip(iota<std::size_t>(), std::forward<Containers>(conts)...);
}

}
}
} // namespace ade

#endif // ADE_UTIL_ZIP_RANGE_HPP
