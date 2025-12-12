// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include <ade/util/intrusive_list.hpp>

namespace
{
struct TestStruct
{
    ade::util::intrusive_list_node node1;
    ade::util::intrusive_list_node node2;
};
}

TEST(IntrusiveList, Basic)
{
    ade::util::intrusive_list<TestStruct, &TestStruct::node1> ilist;

    TestStruct s1{};
    TestStruct s2{};
    TestStruct s3{};

    ASSERT_FALSE(s1.node1.linked());
    ASSERT_FALSE(s2.node1.linked());
    ASSERT_FALSE(s3.node1.linked());

    ASSERT_TRUE(ilist.empty());

    ilist.push_back(s1);

    ASSERT_TRUE(s1.node1.linked());
    ASSERT_FALSE(s2.node1.linked());
    ASSERT_FALSE(s3.node1.linked());

    ASSERT_FALSE(ilist.empty());

    std::vector<TestStruct*> vals;
    for (auto& s: ilist) vals.push_back(&s);

    ASSERT_EQ(std::vector<TestStruct*>{{&s1}}, vals);

    ilist.push_back(s2);

    ASSERT_TRUE(s1.node1.linked());
    ASSERT_TRUE(s2.node1.linked());
    ASSERT_FALSE(s3.node1.linked());

    ASSERT_FALSE(ilist.empty());

    vals.clear();
    for (auto& s: ilist) vals.push_back(&s);

    ASSERT_EQ((std::vector<TestStruct*>{&s1,&s2}), vals);

    ilist.push_front(s3);

    ASSERT_TRUE(s1.node1.linked());
    ASSERT_TRUE(s2.node1.linked());
    ASSERT_TRUE(s3.node1.linked());

    ASSERT_FALSE(ilist.empty());

    vals.clear();
    for (auto& s: ilist) vals.push_back(&s);

    ASSERT_EQ((std::vector<TestStruct*>{&s3,&s1,&s2}), vals);

    ilist.clear();
    ASSERT_TRUE(ilist.empty());

    ASSERT_FALSE(s1.node1.linked());
    ASSERT_FALSE(s2.node1.linked());
    ASSERT_FALSE(s3.node1.linked());
}

TEST(IntrusiveList, AutoUnlink)
{
    {
        ade::util::intrusive_list<TestStruct, &TestStruct::node1> ilist;
        {
            TestStruct s1{};
            TestStruct s2{};
            TestStruct s3{};

            ilist.push_back(s1);
            ilist.push_back(s2);
            ilist.push_back(s3);

            ASSERT_FALSE(ilist.empty());
        }
        ASSERT_TRUE(ilist.empty());
    }
    {
        TestStruct s1{};
        TestStruct s2{};
        TestStruct s3{};

        ASSERT_FALSE(s1.node1.linked());
        ASSERT_FALSE(s2.node1.linked());
        ASSERT_FALSE(s3.node1.linked());
        {
            ade::util::intrusive_list<TestStruct, &TestStruct::node1> ilist;

            ilist.push_back(s1);
            ilist.push_back(s2);
            ilist.push_back(s3);

            ASSERT_TRUE(s1.node1.linked());
            ASSERT_TRUE(s2.node1.linked());
            ASSERT_TRUE(s3.node1.linked());
        }
        ASSERT_FALSE(s1.node1.linked());
        ASSERT_FALSE(s2.node1.linked());
        ASSERT_FALSE(s3.node1.linked());
    }
}

TEST(IntrusiveList, MultipleLists)
{
    ade::util::intrusive_list<TestStruct, &TestStruct::node1> ilist1;
    ade::util::intrusive_list<TestStruct, &TestStruct::node2> ilist2;

    ASSERT_TRUE(ilist1.empty());
    ASSERT_TRUE(ilist2.empty());

    {
        TestStruct s1{};

        ASSERT_FALSE(s1.node1.linked());

        ASSERT_FALSE(s1.node2.linked());

        ilist1.push_back(s1);

        ASSERT_TRUE(s1.node1.linked());

        ASSERT_FALSE(s1.node2.linked());

        ASSERT_FALSE(ilist1.empty());
        ASSERT_TRUE(ilist2.empty());

        ilist2.push_back(s1);

        ASSERT_TRUE(s1.node1.linked());

        ASSERT_TRUE(s1.node2.linked());

        ASSERT_FALSE(ilist1.empty());
        ASSERT_FALSE(ilist2.empty());
    }

    ASSERT_TRUE(ilist1.empty());
    ASSERT_TRUE(ilist2.empty());
}

TEST(IntrusiveList, InsertErase)
{
    ade::util::intrusive_list<TestStruct, &TestStruct::node1> ilist;

    TestStruct s1{};
    TestStruct s2{};
    TestStruct s3{};

    auto iter1 = ilist.insert(ilist.end(), s1);
    ASSERT_FALSE(ilist.empty());
    auto iter2 = ilist.insert(iter1, s2);
    ASSERT_FALSE(ilist.empty());
    auto iter3 = ilist.insert(iter1, s3);
    ASSERT_FALSE(ilist.empty());

    std::vector<TestStruct*> vals;
    for (auto& s: ilist) vals.push_back(&s);

    ASSERT_EQ((std::vector<TestStruct*>{&s2,&s3,&s1}), vals);

    auto iter4 = ilist.erase(iter3);
    ASSERT_FALSE(ilist.empty());
    ASSERT_EQ(iter1, iter4);
    auto iter5 = ilist.erase(iter2);
    ASSERT_FALSE(ilist.empty());
    ASSERT_EQ(iter1, iter5);
    auto iter6 = ilist.erase(iter1);
    ASSERT_TRUE(ilist.empty());
    ASSERT_EQ(ilist.end(), iter6);
}
