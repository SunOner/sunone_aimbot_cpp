// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ade/passmanager.hpp>

#include <ade/util/zip_range.hpp>

using namespace ade;

struct EmptyContext{};

TEST(PassManager, PassList)
{
    EmptyContext ctx;
    bool pass1called = false;
    bool pass2called = false;
    bool pass3called = false;
    PassList<EmptyContext> list;
    list.addPass([&](const EmptyContext&)
    {
        EXPECT_FALSE(pass1called);
        EXPECT_FALSE(pass2called);
        EXPECT_FALSE(pass3called);
        pass1called = true;
    });
    list.addPass([&](const EmptyContext&)
    {
        EXPECT_TRUE(pass1called);
        EXPECT_FALSE(pass2called);
        EXPECT_FALSE(pass3called);
        pass2called = true;
    });
    list.addPass([&](const EmptyContext&)
    {
        EXPECT_TRUE(pass1called);
        EXPECT_TRUE(pass2called);
        EXPECT_FALSE(pass3called);
        pass3called = true;
    });
    list.run(ctx);
    EXPECT_TRUE(pass1called);
    EXPECT_TRUE(pass2called);
    EXPECT_TRUE(pass3called);
}

TEST(PassManager, PassStages)
{
    EmptyContext ctx;
    PassManager<EmptyContext> pm;
    bool pass1called = false;
    bool pass2called = false;
    bool pass3called = false;
    pm.addStage("foo");
    pm.addStage("bar");
    pm.addStage("baz","foo");
    pm.addPass("foo",[&](const EmptyContext&)
    {
        EXPECT_FALSE(pass1called);
        EXPECT_FALSE(pass2called);
        EXPECT_FALSE(pass3called);
        pass1called = true;
    });
    pm.addPass("baz",[&](const EmptyContext&)
    {
        EXPECT_TRUE(pass1called);
        EXPECT_FALSE(pass2called);
        EXPECT_FALSE(pass3called);
        pass2called = true;
    });
    pm.addPass("bar",[&](const EmptyContext&)
    {
        EXPECT_TRUE(pass1called);
        EXPECT_TRUE(pass2called);
        EXPECT_FALSE(pass3called);
        pass3called = true;
    });
    pm.run(ctx);
    EXPECT_TRUE(pass1called);
    EXPECT_TRUE(pass2called);
    EXPECT_TRUE(pass3called);

    EXPECT_FALSE(pm.stages().empty());
    for (auto it: indexed(pm.stages()))
    {
        auto i = util::index(it);
        auto& stage = util::value(it);
        if(0 == i)
        {
            EXPECT_EQ("foo", stage.first);
        }
        else if(1 == i)
        {
            EXPECT_EQ("baz", stage.first);
        }
        else if(2 == i)
        {
            EXPECT_EQ("bar", stage.first);
        }
        else
        {
            FAIL();
        }
    }
}
