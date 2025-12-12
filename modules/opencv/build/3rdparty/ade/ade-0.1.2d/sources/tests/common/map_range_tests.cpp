// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/range.hpp>
#include <ade/util/map_range.hpp>

#include <vector>

TEST(MapRangeTest, Simple)
{
    using namespace ade::util;
    int sum1 = 0;
    std::vector<int> v1 = {1,10,100,1000};
    int coeff = 2;
    for (auto i: map(toRange(v1), [&](int val) { return val * coeff; }))
    {
        sum1 += i;
    }
    EXPECT_EQ(4, size(map(toRange(v1), [&](int val) { return val * coeff; })));
    EXPECT_EQ(4, map(toRange(v1), [&](int val) { return val * coeff; }).size());
    EXPECT_EQ(2222, sum1);

    struct Mapper
    {
        int operator()(int val) const
        {
            return val * 3;
        }
    };
    int sum2 = 0;
    for (auto i: map<Mapper>(toRange(v1)))
    {
        sum2 += i;
    }
    EXPECT_EQ(4, size(map<Mapper>(toRange(v1))));
    EXPECT_EQ(4, map<Mapper>(toRange(v1)).size());
    EXPECT_EQ(3333, sum2);
}
