// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <limits>

#include <ade/util/checked_cast.hpp>

#include <gtest/gtest.h>

struct CheckedCastFailed
{

};

namespace
{
struct CheckedCastTestHandler final
{
    template<typename T>
    void operator()(bool valid, T&& /*value*/) const
    {
        if (!valid)
        {
            throw CheckedCastFailed();
        }
    }
};


template <typename I, typename J>
I checked_cast_test(J value)
{
    return ade::util::checked_cast_impl<I>(value, CheckedCastTestHandler{});
}
}

TEST(CheckedCastTest,CheckValid)
{
   EXPECT_EQ(0,  checked_cast_test<short>((int)0));
   EXPECT_EQ(0,  checked_cast_test<int>((short)0));
   EXPECT_EQ(0,  checked_cast_test<int>((int)0));
   EXPECT_EQ(0u, checked_cast_test<unsigned>((unsigned)0));
   EXPECT_EQ(0,  checked_cast_test<int>((unsigned)0));
   EXPECT_EQ(0u, checked_cast_test<unsigned>((int)0));

   EXPECT_EQ((int)-1,                checked_cast_test<int>((short)-1));
   EXPECT_EQ((unsigned)0xffff,       checked_cast_test<unsigned>((unsigned short)0xffff));

   EXPECT_EQ((short)0x7fff,          checked_cast_test<short>((int)0x7fff));
   EXPECT_EQ((unsigned short)0xffff, checked_cast_test<unsigned short>((unsigned)0xffff));

   EXPECT_EQ(0x7fffffff,  checked_cast_test<int>((unsigned)0x7fffffff));
   EXPECT_EQ(0x7fffffffu, checked_cast_test<unsigned>((int)0x7fffffff));


   EXPECT_EQ(static_cast<int>(0.0f),   checked_cast_test<int>(0.0f));
   EXPECT_EQ(static_cast<int>(-0.0f),  checked_cast_test<int>(-0.0f));
   EXPECT_EQ(static_cast<int>(1.0f),   checked_cast_test<int>(1.0f));
   EXPECT_EQ(static_cast<int>(-1.0f),  checked_cast_test<int>(-1.0f));
   EXPECT_EQ(static_cast<int>(1e-15),  checked_cast_test<int>(1e-15));
   EXPECT_EQ(static_cast<int>(-1e-15), checked_cast_test<int>(-1e-15));
   EXPECT_EQ(static_cast<int>(1000000.0f),  checked_cast_test<int>(1000000.0f));
   EXPECT_EQ(static_cast<int>(-1000000.0f), checked_cast_test<int>(-1000000.0f));

   EXPECT_EQ(static_cast<unsigned>(0.0f),  checked_cast_test<unsigned>(0.0f));
   EXPECT_EQ(static_cast<unsigned>(-0.0f), checked_cast_test<unsigned>(-0.0f));
   EXPECT_EQ(static_cast<unsigned>(1.0f),  checked_cast_test<unsigned>(1.0f));
   EXPECT_EQ(static_cast<unsigned>(1e-15), checked_cast_test<unsigned>(1e-15));
   EXPECT_EQ(static_cast<unsigned>(1000000.0f), checked_cast_test<unsigned>(1000000.0f));


   EXPECT_EQ(static_cast<float>(0.0),  checked_cast_test<float>(0.0));
   EXPECT_EQ(static_cast<float>(-0.0), checked_cast_test<float>(-0.0));
   EXPECT_EQ(static_cast<float>(1.0),  checked_cast_test<float>(1.0));
   EXPECT_EQ(static_cast<float>(-1.0), checked_cast_test<float>(-1.0));
   EXPECT_EQ(static_cast<float>(static_cast<double>(std::numeric_limits<float>::max())),
             checked_cast_test<float>(static_cast<double>(std::numeric_limits<float>::max())));
   EXPECT_EQ(static_cast<float>(static_cast<double>(std::numeric_limits<float>::min())),
             checked_cast_test<float>(static_cast<double>(std::numeric_limits<float>::min())));
   EXPECT_EQ(static_cast<float>(1000000.0),  checked_cast_test<float>(1000000.0));
   EXPECT_EQ(static_cast<float>(-1000000.0), checked_cast_test<float>(-1000000.0));

   for(int i = std::numeric_limits<char>::min(); i <= std::numeric_limits<char>::max(); ++i)
   {
      EXPECT_EQ(static_cast<char>(static_cast<float>(i)), checked_cast_test<char>(static_cast<float>(i)));
   }
}

TEST(CheckedCastTest,CheckOverflows)
{
   EXPECT_THROW(throw CheckedCastFailed(), CheckedCastFailed);
   EXPECT_THROW(CheckedCastTestHandler()(false, 0), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<short>   ((int)0xffffff), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>((int)-1), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<int>     ((unsigned)0xffffffff), CheckedCastFailed);

   EXPECT_THROW(checked_cast_test<int>(1e15), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<int>(-1e15), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(1e15), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(-1e15), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(-1.0f), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(-1e-15), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(-1000000.0f), CheckedCastFailed);

   EXPECT_THROW(checked_cast_test<float>(std::numeric_limits<double>::max()), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<float>(std::numeric_limits<double>::lowest()), CheckedCastFailed);

   EXPECT_THROW(checked_cast_test<int>(std::numeric_limits<float>::infinity()), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<int>(-std::numeric_limits<float>::infinity()), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<int>(std::numeric_limits<float>::quiet_NaN()), CheckedCastFailed);

   EXPECT_THROW(checked_cast_test<unsigned>(std::numeric_limits<float>::infinity()), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(-std::numeric_limits<float>::infinity()), CheckedCastFailed);
   EXPECT_THROW(checked_cast_test<unsigned>(std::numeric_limits<float>::quiet_NaN()), CheckedCastFailed);
}
