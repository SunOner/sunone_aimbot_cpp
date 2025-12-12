// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/md_view.hpp>

TEST(MdView, CopyView)
{

    {
        //1D
        constexpr std::size_t len = 7;
        std::array<int,len> data1 = {{1,2,3,4,5,6,7}};
        std::array<int,len> data2 = {{0,0,0,0,0,0,0}};
        auto mem1 = ade::util::memory_range(data1).reinterpret<void>();
        auto mem2 = ade::util::memory_range(data2).reinterpret<void>();
        ade::util::DynMdView<6, int> view1(
                    mem1, {ade::util::make_dimension(len, sizeof(int))});
        ade::util::DynMdView<6, int> view2(
                    mem2, {ade::util::make_dimension(len, sizeof(int))});
        ade::util::view_copy(view1, view2);
        EXPECT_EQ(data1, data2);
    }
    {
        //2D flat buffer
        constexpr std::size_t w = 3;
        constexpr std::size_t h = 3;
        constexpr std::size_t step1 = 3;
        constexpr std::size_t step2 = 3;
        constexpr std::size_t len1 = step1 * h;
        constexpr std::size_t len2 = step2 * h;
        std::array<int,len1> data1 = {{1,2,3,
                                       4,5,6,
                                       7,8,9}};
        std::array<int,len2> data2 = {{0,0,0,
                                       0,0,0,
                                       0,0,0}};
        auto mem1 = ade::util::memory_range(data1).reinterpret<void>();
        auto mem2 = ade::util::memory_range(data2).reinterpret<void>();
        ade::util::DynMdView<6, int> view1(
                    mem1, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step1 * sizeof(int))});
        ade::util::DynMdView<6, int> view2(
                    mem2, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step2 * sizeof(int))});
        ade::util::view_copy(view1, view2);
        std::array<int,len2> expected = {{1,2,3,
                                          4,5,6,
                                          7,8,9}};
        EXPECT_EQ(expected, data2);
    }
    {
        //2D buffer with strides
        constexpr std::size_t w = 3;
        constexpr std::size_t h = 3;
        constexpr std::size_t step1 = 5;
        constexpr std::size_t step2 = 6;
        constexpr std::size_t len1 = step1 * h;
        constexpr std::size_t len2 = step2 * h;
        std::array<int,len1> data1 = {{1,2,3,-1,-1,
                                       4,5,6,-1,-1,
                                       7,8,9,-1,-1}};
        std::array<int,len2> data2 = {{0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0}};
        auto mem1 = ade::util::memory_range(data1).reinterpret<void>();
        auto mem2 = ade::util::memory_range(data2).reinterpret<void>();
        ade::util::DynMdView<6, int> view1(
                    mem1, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step1 * sizeof(int))});
        ade::util::DynMdView<6, int> view2(
                    mem2, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step2 * sizeof(int))});
        ade::util::view_copy(view1, view2);
        std::array<int,len2> expected = {{1,2,3,0,0,0,
                                          4,5,6,0,0,0,
                                          7,8,9,0,0,0}};
        EXPECT_EQ(expected, data2);
    }
    {
        //3D flat buffer
        constexpr std::size_t w = 3;
        constexpr std::size_t h = 3;
        constexpr std::size_t d = 2;
        constexpr std::size_t step1 = 3;
        constexpr std::size_t step2 = 3;
        constexpr std::size_t dstep1 = step1 * h;
        constexpr std::size_t dstep2 = step2 * h;
        constexpr std::size_t len1 = dstep1 * d;
        constexpr std::size_t len2 = dstep2 * d;
        std::array<int,len1> data1 = {{1 ,2 ,3 ,
                                       4 ,5 ,6 ,
                                       7 ,8 ,9 ,
                                       10,11,12,
                                       13,14,15,
                                       16,17,18,}};
        std::array<int,len2> data2 = {{0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0}};
        auto mem1 = ade::util::memory_range(data1).reinterpret<void>();
        auto mem2 = ade::util::memory_range(data2).reinterpret<void>();
        ade::util::DynMdView<6, int> view1(
                    mem1, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step1 * sizeof(int)),
                           ade::util::make_dimension(d, dstep1 * sizeof(int))});
        ade::util::DynMdView<6, int> view2(
                    mem2, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step2 * sizeof(int)),
                           ade::util::make_dimension(d, dstep2 * sizeof(int))});
        ade::util::view_copy(view1, view2);
        std::array<int,len2> expected = {{1 ,2 ,3 ,
                                          4 ,5 ,6 ,
                                          7 ,8 ,9 ,
                                          10,11,12,
                                          13,14,15,
                                          16,17,18,}};
        EXPECT_EQ(expected, data2);
    }
    {
        //3D buffer mixed
        constexpr std::size_t w = 3;
        constexpr std::size_t h = 3;
        constexpr std::size_t d = 2;
        constexpr std::size_t step1 = 3;
        constexpr std::size_t step2 = 3;
        constexpr std::size_t dstep1 = step1 * h + step1;
        constexpr std::size_t dstep2 = step2 * h + step2 * 2;
        constexpr std::size_t len1 = dstep1 * d;
        constexpr std::size_t len2 = dstep2 * d;
        std::array<int,len1> data1 = {{1 ,2 ,3 ,
                                       4 ,5 ,6 ,
                                       7 ,8 ,9 ,
                                       -1,-1,-1,
                                       10,11,12,
                                       13,14,15,
                                       16,17,18}};
        std::array<int,len2> data2 = {{0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0,
                                       0,0,0}};
        auto mem1 = ade::util::memory_range(data1).reinterpret<void>();
        auto mem2 = ade::util::memory_range(data2).reinterpret<void>();
        ade::util::DynMdView<6, int> view1(
                    mem1, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step1 * sizeof(int)),
                           ade::util::make_dimension(d, dstep1 * sizeof(int))});
        ade::util::DynMdView<6, int> view2(
                    mem2, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step2 * sizeof(int)),
                           ade::util::make_dimension(d, dstep2 * sizeof(int))});
        ade::util::view_copy(view1, view2);
        std::array<int,len2> expected = {{1 ,2 ,3 ,
                                          4 ,5 ,6 ,
                                          7 ,8 ,9 ,
                                          0 ,0 ,0 ,
                                          0 ,0 ,0 ,
                                          10,11,12,
                                          13,14,15,
                                          16,17,18}};
        EXPECT_EQ(expected, data2);
    }
    {
        //3D buffer with strides
        constexpr std::size_t w = 3;
        constexpr std::size_t h = 3;
        constexpr std::size_t d = 2;
        constexpr std::size_t step1 = 5;
        constexpr std::size_t step2 = 6;
        constexpr std::size_t dstep1 = step1 * h + step1;
        constexpr std::size_t dstep2 = step2 * h + step2 * 2;
        constexpr std::size_t len1 = dstep1 * d;
        constexpr std::size_t len2 = dstep2 * d;
        std::array<int,len1> data1 = {{1 ,2 ,3 ,-1,-1,
                                       4 ,5 ,6 ,-1,-1,
                                       7 ,8 ,9 ,-1,-1,
                                       -1,-1,-1,-1,-1,
                                       10,11,12,-1,-1,
                                       13,14,15,-1,-1,
                                       16,17,18,-1,-1,}};
        std::array<int,len2> data2 = {{0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0,
                                       0,0,0,0,0,0}};
        auto mem1 = ade::util::memory_range(data1).reinterpret<void>();
        auto mem2 = ade::util::memory_range(data2).reinterpret<void>();
        ade::util::DynMdView<6, int> view1(
                    mem1, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step1 * sizeof(int)),
                           ade::util::make_dimension(d, dstep1 * sizeof(int))});
        ade::util::DynMdView<6, int> view2(
                    mem2, {ade::util::make_dimension(w, sizeof(int)),
                           ade::util::make_dimension(h, step2 * sizeof(int)),
                           ade::util::make_dimension(d, dstep2 * sizeof(int))});
        ade::util::view_copy(view1, view2);
        std::array<int,len2> expected = {{1 ,2 ,3 ,0 ,0 ,0 ,
                                          4 ,5 ,6 ,0 ,0 ,0 ,
                                          7 ,8 ,9 ,0 ,0 ,0 ,
                                          0 ,0 ,0 ,0 ,0 ,0 ,
                                          0 ,0 ,0 ,0 ,0 ,0 ,
                                          10,11,12,0 ,0 ,0 ,
                                          13,14,15,0 ,0 ,0 ,
                                          16,17,18,0 ,0 ,0}};
        EXPECT_EQ(expected, data2);
    }
}
