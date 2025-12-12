// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/util/func_ref.hpp>

static int test_func(ade::util::func_ref<int()> func)
{
    return func();
}

static int func()
{
    return 42;
}

TEST(FuncRef, Simple)
{
    struct Functor
    {
        int operator()()
        {
            return 42;
        }
    };

    Functor test;
    EXPECT_EQ(42, test_func(&func));
    EXPECT_EQ(42, test_func(func));
    EXPECT_EQ(42, test_func(test));
    EXPECT_EQ(42, test_func(Functor()));
    EXPECT_EQ(42, test_func([](){ return 42; }));
}
