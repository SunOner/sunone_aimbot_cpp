// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/zip_range.hpp>
#include <ade/util/map_range.hpp>

#include <vector>

TEST(ZipRangeTest, Indexed)
{
    using namespace ade::util;
    int sum1 = 0;
    std::vector<int> v1 = {1,10,100,1000};
    int v2[] = {3,30,300};
    int counter = 0;
    for(auto i: indexed(v1))
    {
        EXPECT_EQ(counter, index(i));
        sum1 += value(i);
        ++counter;
    }
    EXPECT_EQ(1111, sum1);

    sum1 = 0;
    int sum2 = 0;
    counter = 0;
    for(auto i: indexed(v1,v2))
    {
        EXPECT_EQ(counter, index(i));
        sum1 += value<0>(i);
        sum2 += value<1>(i);
        ++counter;
    }
    EXPECT_EQ(111, sum1);
    EXPECT_EQ(333, sum2);
}

TEST(ZipRangeTest, IndexedRvalue)
{
    using namespace ade::util;
    int sum1 = 0;
    std::vector<int> v1 = {1,10,100,1000};
    int v2[] = {3,30,300};
    int counter = 0;
    for(auto i: indexed(map(toRange(v1), [](int i) { return i * 2; })))
    {
        EXPECT_EQ(counter, index(i));
        sum1 += value(i);
        ++counter;
    }
    EXPECT_EQ(2222, sum1);

    sum1 = 0;
    int sum2 = 0;
    counter = 0;
    for(auto i: indexed(v1,map(toRange(v2), [](int i) { return i * 2; })))
    {
        EXPECT_EQ(counter, index(i));
        sum1 += value<0>(i);
        sum2 += value<1>(i);
        ++counter;
    }
    EXPECT_EQ(111, sum1);
    EXPECT_EQ(666, sum2);
}

TEST(ZipRangeTest, directRangesAsParameters)
{
    using namespace ade::util;
    int sum1 = 0;
    std::vector<int> v1 = {1,10,100,1000};
    int v2[] = {3,30,300};
    int counter = 0;

    sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    counter = 0;
    enum {vector, array, index};
    for(auto i: zip(v1, v2, iota<int>()))
    {
        using std::get;
        sum1 += get<vector>(i);
        sum2 += get<array>(i);
        EXPECT_EQ(counter, get<index>(i));
        ++counter;
    }
    EXPECT_EQ(111, sum1);
    EXPECT_EQ(333, sum2);
}
