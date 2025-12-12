// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/range.hpp>
#include <ade/util/iota_range.hpp>
#include <ade/util/chain_range.hpp>
#include <ade/util/memory_range.hpp>

#include <array>
#include <vector>

TEST(ChainRangeTest, Identity)
{
    int arr[] = {1,2,3,4,5};

    std::vector<int> result;

    for (auto& elem: ade::util::chain(ade::util::toRange(arr)))
    {
        static_assert(std::is_same<decltype(elem), int&>::value, "Invalid type");
        result.push_back(elem);
    }

    EXPECT_EQ((std::vector<int>{1,2,3,4,5}), result);
}

TEST(ChainRangeTest, SameElemTypes2)
{
    std::array<int, 3> arr1 = {1,2,3};
    std::vector<int> arr2 = {4,5};

    std::vector<int> result;

    for (auto& elem: ade::util::chain(ade::util::toRange(arr1),
                                      ade::util::toRange(arr2)))
    {
        static_assert(std::is_same<decltype(elem), int&>::value, "Invalid type");
        result.push_back(elem);
    }

    EXPECT_EQ((std::vector<int>{1,2,3,4,5}), result);
}

TEST(ChainRangeTest, DifferentElemTypes2)
{
    std::array<int, 3> arr1 = {1,2,3};
    short arr2[] = {4,5};

    std::vector<int> result;

    for (auto elem: ade::util::chain(ade::util::toRange(arr1),
                                     ade::util::toRange(arr2)))
    {
        static_assert(std::is_same<decltype(elem), int>::value, "Invalid type");
        result.push_back(elem);
    }

    EXPECT_EQ((std::vector<int>{1,2,3,4,5}), result);
}

TEST(ChainRangeTest, SameElemTypes3)
{
    std::array<int, 3> arr1 = {1,2,3};
    std::vector<int> arr2 = {4,5};
    int arr3[] = {6,7,8};

    std::vector<int> result;

    for (auto& elem: ade::util::chain(ade::util::toRange(arr1),
                                      ade::util::toRange(arr2),
                                      ade::util::toRange(arr3)))
    {
        static_assert(std::is_same<decltype(elem), int&>::value, "Invalid type");
        result.push_back(elem);
    }

    EXPECT_EQ((std::vector<int>{1,2,3,4,5,6,7,8}), result);
}

TEST(ChainRangeTest, DifferentElemTypes3)
{
    std::array<int, 3> arr1 = {1,2,3};
    std::vector<short> arr2 = {4,5};
    long long arr3[] = {6,7,8};

    std::vector<int> result;

    for (auto elem: ade::util::chain(ade::util::toRange(arr1),
                                     ade::util::toRange(arr2),
                                     ade::util::toRange(arr3)))
    {
        static_assert(std::is_same<decltype(elem), long long>::value, "Invalid type");
        result.push_back(static_cast<int>(elem));
    }

    EXPECT_EQ((std::vector<int>{1,2,3,4,5,6,7,8}), result);
}

TEST(ChainRangeTest, Test4)
{
    std::array<int, 3> arr1 = {1,2,3};
    std::vector<int> arr2 = {4,5};
    int arr3[] = {6,7,8};

    std::vector<int> result;

    for (auto elem: ade::util::chain(ade::util::toRange(arr1),
                                     ade::util::toRange(arr2),
                                     ade::util::toRange(arr3),
                                     ade::util::iota(9,12)))
    {
        static_assert(std::is_same<decltype(elem), int>::value, "Invalid type");
        result.push_back(elem);
    }

    EXPECT_EQ((std::vector<int>{1,2,3,4,5,6,7,8,9,10,11}), result);
}

TEST(ChainRangeTest, SizeAndOpIndex)
{
    std::array<int, 3> arr1 = {1,2,3};
    std::vector<int> arr2 = {4,5};
    int arr3[] = {6,7,8};

    auto range = ade::util::chain(ade::util::memory_range(arr1.data(), arr1.size()),
                                  ade::util::memory_range(arr2.data(), arr2.size()),
                                  ade::util::memory_range(&arr3[0], 3));
    const auto& crange = range;

    std::vector<int> result = {1,2,3,4,5,6,7,8};
    const auto count = result.size();
    ASSERT_EQ(count, range.size());
    ASSERT_EQ(count, crange.size());

    for (auto i: ade::util::iota(count))
    {
        EXPECT_EQ(result[i], range[i]);
        EXPECT_EQ(result[i], crange[i]);
    }
    range[1] = 55;
    result[1] = 55;
    range[3] = 66;
    result[3] = 66;
    range[6] = 77;
    result[6] = 77;
    for (auto i: ade::util::iota(count))
    {
        EXPECT_EQ(result[i], range[i]);
        EXPECT_EQ(result[i], crange[i]);
    }
}
