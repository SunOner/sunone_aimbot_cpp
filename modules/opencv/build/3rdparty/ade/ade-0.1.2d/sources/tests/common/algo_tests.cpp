// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/algorithm.hpp>

#include <map>
#include <vector>

TEST(AlgoTest, Find)
{
    const std::vector<int> v1 = {1,2,3};
    EXPECT_EQ(v1.begin() + 1, ade::util::find(v1,2));
    EXPECT_EQ(v1.end(), ade::util::find(v1,4));

    const int v2[] = {1,2,3};
    EXPECT_EQ(std::begin(v2) + 1, ade::util::find(v2,2));
    EXPECT_EQ(std::end(v2), ade::util::find(v2,4));
}

TEST(AlgoTest, Contains)
{
    const std::map<int,int> m = {{1,2},{2,3},{3,4}};
    EXPECT_TRUE(ade::util::contains(m, 2));
    EXPECT_FALSE(ade::util::contains(m, 6));
}

static_assert(0 == ade::util::type_list_index<int, int>::value, "type_list_index test failed");
static_assert(0 == ade::util::type_list_index<int, int, float, char>::value, "type_list_index test failed");
static_assert(1 == ade::util::type_list_index<float, int, float, char>::value, "type_list_index test failed");
static_assert(2 == ade::util::type_list_index<char, int, float, char>::value, "type_list_index test failed");
static_assert(0 == ade::util::type_list_index<int, int, int, int>::value, "type_list_index test failed");

// Negative test will fail compilation, keep it here for reference
//static_assert(0 == ade::util::type_list_index<short, int, float, char>::value, "type_list_index test failed");
