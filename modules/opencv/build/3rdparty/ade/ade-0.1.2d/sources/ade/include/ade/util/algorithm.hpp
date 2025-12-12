// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file algorithm.hpp

#ifndef ADE_UTIL_ALGORITHM_HPP
#define ADE_UTIL_ALGORITHM_HPP

#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <iterator>

#include "ade/util/type_traits.hpp"
#include "ade/util/checked_cast.hpp"

namespace ade
{
namespace util
{

// Convert object to string, if it has an operator<<
//
// @param object that needs to be converted to a string
//
// @return value string that we see when using the output stream operator of the object
template<typename T>
std::string to_string(const T& obj)
{
    std::stringstream ss;
    ss << obj;
    return ss.str();
}

/// std::any_of range wrapper.
///
/// @param c range, which implements begin and end iterators
/// @param p predicate
///
/// @return true if predicate returns true for at least one element in range
template <typename container_t, typename predicate_t>
inline bool any_of(container_t&& c, predicate_t&& p)
{
    return std::any_of(std::begin(c), std::end(c),
                       std::forward<predicate_t>(p));
}

/// std::all_of range wrapper.
///
/// @param c range, which implements begin and end iterators
/// @param p predicate
///
/// @return true if predicate returns true for all elements in range
template <typename container_t, typename predicate_t>
inline bool all_of(container_t&& c, predicate_t&& p)
{
    return std::all_of(std::begin(c), std::end(c),
                       std::forward<predicate_t>(p));
}

/// std::find_if range wrapper.
///
/// @param c range, which implements begin and end iterators
/// @param p predicate which returns true for required element
/// @return iterator to first found element or std::end(c)
template<typename container_t, typename predicate_t>
inline decltype(std::begin(std::declval<container_t>()))
find_if(container_t&& c, predicate_t&& p)
{
    return std::find_if(std::begin(c), std::end(c),
                        std::forward<predicate_t>(p));
}

/// std::find range wrapper.
///
/// @param c range, which implements begin and end iterators
/// @param val value to compare range elements to
/// @return iterator to first found element or std::end(c)
template<typename container_t, typename value_t>
inline decltype(std::begin(std::declval<container_t>()))
find(container_t&& c, const value_t& val)
{
    return std::find(std::begin(c), std::end(c), val);
}

/// Check whether element present in range.
///
/// @param cont range, which must implement find member function
/// @param val key to search for
/// @return true if element present
template<typename C, typename T>
inline bool contains(const C& cont, const T& val)
{
    return cont.end() != cont.find(val);
}

/// Removes specified element from range replacing it with last element.
///
/// @param c range to modify, must support back() and pop_back() methods
/// @param it iterator to element to be removed
template<typename container_t, typename iterator_t>
void unstable_erase(container_t&& c, iterator_t&& it)
{
    *it = std::move(c.back());
    c.pop_back();
}

/// std::copy range wrapper.
/// TODO: make second parameter range too.
///
/// @param c range to be copied from
/// @param it iterator to first element of destination range.
/// @return iterator to element in destination range, past the last element
/// copied
template<typename container_t, typename output_iterator_t>
inline remove_reference_t<output_iterator_t>
copy(container_t &&c, output_iterator_t &&it)
{
    return std::copy(std::begin(c), std::end(c),
                     std::forward<output_iterator_t>(it));
}

/// std::copy_if range wrapper.
/// TODO: make second parameter range too.
///
/// @param c range to be copied from
/// @param it iterator to first element of destination range.
/// @param p predicate which returns true for required elements.
/// @return iterator to element in destination range, past the last element
/// copied
template<typename container_t, typename output_iterator_t, typename predicate_t>
inline remove_reference_t<output_iterator_t>
copy_if(container_t &&c, output_iterator_t &&it, predicate_t&& p)
{
    return std::copy_if(std::begin(c), std::end(c),
                        std::forward<output_iterator_t>(it),
                        std::forward<predicate_t>(p));
}

/// std::transform range wrapper.
/// TODO: make second parameter range too.
///
/// @param c range of elements to transform.
/// @param it iterator to first element of destination range.
/// @param p unary predicate that will be applied.
/// @return output iterator to element past the last element transformed
template<typename container_t, typename output_iterator_t, typename predicate_t>
inline remove_reference_t<output_iterator_t>
transform(container_t &&c, output_iterator_t &&it, predicate_t&& p)
{
    return std::transform(std::begin(c), std::end(c),
                          std::forward<output_iterator_t>(it),
                          std::forward<predicate_t>(p));
}

/// std::max_element range wrapper.
///
/// @param c source range.
/// @param p comparison predicate.
/// @return iterator to greatest element in source range or end(c) if range
/// is empty
template<typename container_t, typename predicate_t>
inline decltype(std::begin(std::declval<container_t>()))
max_element(container_t&& c, predicate_t&& p)
{
    return std::max_element(std::begin(c), std::end(c),
                            std::forward<predicate_t>(p));
}

/// Numeric conversion for range of elements.
/// Applies util::checked_cast to all elements in source range.
/// TODO: use ranges, or remove this functions completely.
///
/// @param inFirst iterator to first element in source range.
/// @param inLast iterator to past the last element in source range.
/// @param outFirst iterator to first element in destination range.
template<typename IterIn, typename IterOut>
inline void convert(IterIn inFirst, IterIn inLast, IterOut outFirst)
{
    typedef typename std::iterator_traits<IterIn>::value_type InT;
    typedef typename std::iterator_traits<IterOut>::value_type OutT;
    std::transform(inFirst, inLast, outFirst, [](InT v) {
        return checked_cast<OutT>(v);
    });
}

/// Numeric conversion for range of elements.
/// Applies util::checked_cast to all elements in source range.
/// TODO: remove.
///
/// @param inFirst iterator to first element in source range.
/// @param n number of elements to convert.
/// @param outFirst iterator to first element in destination range.
template<typename IterIn, typename N, typename IterOut>
inline void convert_n(IterIn inFirst, N n, IterOut outFirst)
{
    util::convert(inFirst, inFirst+n, outFirst);
}

/// std::fill range wrapper.
///
/// @param c range to modify.
/// @param val the value ti be assigned.
template<typename container_t, typename value_t>
void fill(container_t&& c, value_t&& val)
{
    std::fill(std::begin(c), std::end(c), std::forward<value_t>(val));
}

namespace details
{

template<std::size_t I, typename Target, typename First, typename... Remaining>
struct type_list_index_helper
{
    static const constexpr bool is_same = std::is_same<Target, First>::value;
    static const constexpr std::size_t value =
            std::conditional<is_same, std::integral_constant<std::size_t, I>,
                             type_list_index_helper<I + 1,
                                                    Target,
                                                    Remaining...>>::type::value;
};

template<std::size_t I, typename Target, typename First>
struct type_list_index_helper<I, Target, First>
{
    static_assert(std::is_same<Target, First>::value, "Type not found");
    static const constexpr std::size_t value = I;
};

}

template<typename Target, typename... Types>
struct type_list_index
{
    static const constexpr std::size_t value =
            ::ade::util::details::type_list_index_helper<0,
                                                         Target,
                                                         Types...>::value;
};

} // namespace util
} // namespace ade

#endif // ADE_UTIL_ALGORITHM_HPP
