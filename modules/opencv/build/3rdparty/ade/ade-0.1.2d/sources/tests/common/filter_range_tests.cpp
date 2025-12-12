// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/range.hpp>
#include <ade/util/filter_range.hpp>

#include <vector>

TEST(FilterRangeTest, Test)
{
    using namespace ade::util;
    int sum1 = 0;
    std::vector<int> v1 = {1,10,100,1000,10000,100000,1000000};
    for (auto i: filter(toRange(v1), [&](int /*val*/) { return true; }))
    {
        sum1 += i;
    }
    EXPECT_EQ(1111111, sum1);

    struct Filter final
    {
        bool operator()(int /*val*/) const
        {
            return true;
        }
    };

    sum1 = 0;
    for (auto i: filter<Filter>(toRange(v1)))
    {
        sum1 += i;
    }
    EXPECT_EQ(1111111, sum1);

    sum1 = 0;
    for (auto i: filter(toRange(v1), [&](int /*val*/) { return false; }))
    {
        sum1 += i;
    }
    EXPECT_EQ(0, sum1);

    sum1 = 0;
    for (auto i: filter(toRange(v1), [&](int val)
                    {
                        if (1 == val) return true;
                        return false;
                    }))
    {
        sum1 += i;
    }
    EXPECT_EQ(1, sum1);

    sum1 = 0;
    for (auto i: filter(toRange(v1), [&](int val)
                    {
                        if (1000000 == val) return true;
                        return false;
                    }))
    {
        sum1 += i;
    }
    EXPECT_EQ(1000000, sum1);

    sum1 = 0;
    for (auto i: filter(toRange(v1), [&](int val)
                    {
                        if (1 == val || 1000000 == val) return true;
                        return false;
                    }))
    {
        sum1 += i;
    }
    EXPECT_EQ(1000001, sum1);

    sum1 = 0;
    for (auto i: filter(toRange(v1), [&](int val)
                    {
                        if (1000 == val || 10000 == val || 100000 == val) return true;
                        return false;
                    }))
    {
        sum1 += i;
    }
    EXPECT_EQ(111000, sum1);
}
