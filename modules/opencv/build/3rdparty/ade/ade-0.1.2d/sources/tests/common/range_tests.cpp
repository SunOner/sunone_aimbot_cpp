// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/range.hpp>

#include <vector>
#include <map>

TEST(RangeTest, ToRange)
{
    using namespace ade::util;
    std::vector<int> v1 = {1,10,100};
    int sum = 0;
    for(auto i: toRange(v1))
    {
        sum += i;
    }
    EXPECT_EQ(3, size(toRange(v1)));
    EXPECT_EQ(3, toRange(v1).size());
    EXPECT_EQ(111,sum);

    int v2[] = {1,10,100};
    sum = 0;
    for(auto i: toRange(v2))
    {
        sum += i;
    }
    EXPECT_EQ(3, size(toRange(v2)));
    EXPECT_EQ(3, toRange(v2).size());
    EXPECT_EQ(111,sum);

    sum = 0;
    for(auto i: toRange(std::make_pair(v1.begin(),v1.end())))
    {
        sum += i;
    }
    EXPECT_EQ(3, size(toRange(std::make_pair(v1.begin(),v1.end()))));
    EXPECT_EQ(3, toRange(std::make_pair(v1.begin(),v1.end())).size());
    EXPECT_EQ(111,sum);
}

TEST(RangeTest, IndexValue)
{
    using namespace ade::util;
    int sum1 = 0;
    int sum2 = 0;
    const std::map<int,int> m = {{1,3},{10,30},{100,300}};
    for(auto i: m)
    {
        sum1 += index(i);
        sum2 += value(i);
    }
    EXPECT_EQ(111,sum1);
    EXPECT_EQ(333,sum2);
}

TEST(RangeTest, Reverse)
{
    using namespace ade::util;
    std::vector<int> v1 = {1,2,3,4};
    std::vector<int> v2;
    for(auto i: toRangeReverse(v1))
    {
        v2.push_back(i);
    }
    ASSERT_EQ(4, v2.size());
    ASSERT_EQ(4, v2[0]);
    ASSERT_EQ(3, v2[1]);
    ASSERT_EQ(2, v2[2]);
    ASSERT_EQ(1, v2[3]);
}

namespace
{

struct TestHasSize1
{

};
struct TestHasSize2
{
    int size();
};
struct TestHasSize3
{

};
int size(const TestHasSize3&);

static_assert(ade::util::details::has_size_fun<TestHasSize1>::value == false, "TestHasSize1");
static_assert(ade::util::details::has_size_fun<TestHasSize2>::value == false, "TestHasSize2");
static_assert(ade::util::details::has_size_fun<TestHasSize3>::value == true,  "TestHasSize3");

}
