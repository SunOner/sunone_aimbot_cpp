// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <type_traits>

#include <ade/util/tuple.hpp>

TEST(TupleTest, Foreach)
{
    auto t1 = std::make_tuple(42, std::string("123"), "456");
    const auto t2 = std::make_tuple(42, std::string("123"), "456");
    struct
    {
        int callCount = 0;
        void operator()(int i)
        {
            EXPECT_EQ(42, i);
            ++callCount;
        }

        void operator()(const std::string& str)
        {
            EXPECT_EQ("123", str);
            ++callCount;
        }

        void operator()(const char* str)
        {
            EXPECT_EQ("456", std::string(str));
            ++callCount;
        }
    } fun;

    using namespace ade::util;
    tupleForeach(t1, fun);
    EXPECT_EQ(3, fun.callCount);
    tupleForeach(t2, fun);
    EXPECT_EQ(6, fun.callCount);
}

TEST(TupleTest, FixRvals)
{
    int i = 5;
    const int j = 7;
    int& k = i;
    auto t = ade::util::tuple_remove_rvalue_refs(i, j, k, 11);
    EXPECT_EQ(5,  std::get<0>(t));
    EXPECT_EQ(7,  std::get<1>(t));
    EXPECT_EQ(5,  std::get<2>(t));
    EXPECT_EQ(11, std::get<3>(t));

    static_assert(std::is_same<std::tuple_element<0, decltype(t)>::type, int&>::value, "Invalid elements type");
    static_assert(std::is_same<std::tuple_element<1, decltype(t)>::type, const int&>::value, "Invalid elements type");
    static_assert(std::is_same<std::tuple_element<2, decltype(t)>::type, int&>::value, "Invalid elements type");
    static_assert(std::is_same<std::tuple_element<3, decltype(t)>::type, int>::value, "Invalid elements type");
}
