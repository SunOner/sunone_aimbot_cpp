// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ade/communication/callback_connector.hpp"

#include "ade/util/algorithm.hpp"
#include "ade/util/iota_range.hpp"

using namespace ade;

namespace
{
void testCommProducersConsumersCalls(int producers, int consumers)
{
    // Producers consumers connections
    CallbackConnector<> comm(producers, consumers);

    std::vector<int> consumersCalled(consumers);

    for (auto i: util::iota(consumers))
    {
        comm.addConsumerCallback([i, &consumersCalled]()
        {
            ++consumersCalled[i];
        });
    }

    auto resetter = comm.finalize();

    for (auto t: util::iota(3))
    {
        util::fill(consumersCalled, 0);
        std::vector<std::function<void(void)>> producersCallbacks(producers);

        for (auto i: util::iota(producers))
        {
            producersCallbacks[i] = comm.getProducerCallback();
        }

        for (auto i: util::iota(producers))
        {
            for (auto j: util::iota(consumers))
            {
                EXPECT_EQ(0, consumersCalled[j]);
            }

            producersCallbacks[i]();

            // All consumers must be called exactly 1 time after all producers calls
            const auto callCount = (i == (producers - 1) ? 1 : 0);

            for (auto j: util::iota(consumers))
            {
                EXPECT_EQ(callCount, consumersCalled[j]);
            }
        }

        for (auto j: util::iota(consumers))
        {
            // Each consumer must be called exactly 1 time
            EXPECT_EQ(1, consumersCalled[j]);
        }

        if (nullptr != resetter)
        {
            resetter();
        }
    }
}
}

TEST(Communications, CallbackConnector)
{
    for (auto i: util::iota(1,4))
    {
        for (auto j: util::iota(1,4))
        {
            std::stringstream ss;
            ss << "Test comm producer/consumer connections " << i << " " << j;
            SCOPED_TRACE(ss.str());
            testCommProducersConsumersCalls(i, j);
        }
    }
}
