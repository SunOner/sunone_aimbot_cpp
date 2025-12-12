// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file assert.hpp

#ifndef ADE_UTIL_ASSERT_HPP
#define ADE_UTIL_ASSERT_HPP

#include <cassert>
#include <utility>

#if defined(_MSC_VER)
#define ade_unreachable() __assume(0)
#elif defined(__GNUC__)
#define ade_unreachable() __builtin_unreachable()
#else
#define ade_unreachable() do{}while(false)
#endif

namespace ade
{
namespace details
{
[[noreturn]] void dev_assert_abort(const char* str, int line, const char* file,
                const char* func);
[[noreturn]] void dev_exception_abort(const char* str);
} // namespace details
} // namespace ade

/// ADE assertion.
/// Alias to standard assert in regular builds.
/// Replaced with stringer version, working in release builds when
/// FORCE_ADE_ASSERTS=ON
///
/// @param expr Expression convertible to bool
#if defined(FORCE_ADE_ASSERTS)
#define ADE_ASSERT(expr)                                \
    do { if (!(expr)) ::ade::details::dev_assert_abort( \
            #expr, __LINE__, __FILE__, __func__);       \
    }while(false)
#else
#define ADE_ASSERT(expr) do {                     \
    constexpr bool _assert_tmp = false && (expr); \
    (void) _assert_tmp;                           \
    assert(expr);                                 \
} while(false)
#endif

/// Stronger version of assert which translates to compiler hint in optimized
/// builds. Do not use it if you have subsequent error recovery code because
/// this code can be optimized away.
/// Expression is always evaluated, avoid functions calls in it.
/// Static analyzers friendly and can silence "possible null pointer
/// dereference" warnings.
///
/// @param expr Expression convertible to bool
#if defined(FORCE_ADE_ASSERTS) || !defined(NDEBUG)
#define ADE_ASSERT_STRONG(expr) ADE_ASSERT(expr)
#else
#define ADE_ASSERT_STRONG(expr) do{ if(!(expr)) { ade_unreachable(); } }while(false)
#endif

/// Mark current location is not supposed to be reachable.
/// Depending of FORCE_ADE_ASSERTS flag can be translated into assert or
/// compiler hint.
///
/// @param str String description.
#define ADE_UNREACHABLE(str) ADE_ASSERT_STRONG(!str)

/// Mark variable UNUSED, suitable if it is used only in ADE_ASSERT
///
/// @param x Variable to mark
#define ADE_UNUSED(x) (void)(x)

namespace ade
{
/// Convenient function for exception throwing.
/// Throws provided exceptions object if exceptions are enabled in compiler
/// aborts othervise.
///
/// @param e Exception object.
template <class ExceptionType>
[[noreturn]] void throw_error(ExceptionType &&e)
{
#if defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    throw std::forward<ExceptionType>(e);
#else
    ::ade::details::dev_exception_abort(e.what());
#endif
}
} // namespace ade

#endif // ADE_UTIL_ASSERT_HPP
