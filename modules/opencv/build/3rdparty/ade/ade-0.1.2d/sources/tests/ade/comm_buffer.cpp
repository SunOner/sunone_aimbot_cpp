// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>

#include <ade/communication/comm_buffer.hpp>

#include <ade/util/iota_range.hpp>

using namespace ade;

namespace
{
class TestBuffer : public IDataBuffer
{
public:
    int mapCalled = 0;
    int unmapCalled = 0;
    memory::DynMdView<void> view;

    // IDataBuffer interface
    virtual MapId map(const Span& /*span*/, Access /*access*/) override
    {
        ++mapCalled;
        return MapId{view,0};
    }
    virtual void unmap(const MapId& /*id*/) override
    {
        ++unmapCalled;
    }
    virtual void finalizeWrite(const Span& /*span*/) override
    {
    }
    virtual void finalizeRead(const Span& /*span*/) override
    {
    }
    virtual Size alignment(const Span& /*span*/) override
    {
        return Size{};
    }
};

}

TEST(CommBuffer, DataBufferMapper)
{
    memory::DynMdSpan span({util::Span(0,16),util::Span(0,16)});
    std::array<int,256> data;
    util::MemoryRange<void> mem(data.data(), data.size() * sizeof(int));
    memory::DynMdView<void> view(mem, {util::make_dimension(16,sizeof(int)),
                                       util::make_dimension(16,16 * sizeof(int))});

    TestBuffer buff;
    buff.view = view;
    DataBufferView buffview(&buff, span);
    {
        EXPECT_EQ(0, buff.mapCalled);
        EXPECT_EQ(0, buff.unmapCalled);
        DataBufferMapper mapper(buffview, span, IDataBuffer::Read);
        EXPECT_EQ(1, buff.mapCalled);
        EXPECT_EQ(0, buff.unmapCalled);
        auto mappedView = mapper.view();
        EXPECT_EQ(mem.data, mappedView.mem.data);
        EXPECT_EQ(mem.size, mappedView.mem.size);
        ASSERT_EQ(mappedView.count(), view.count());
        for (auto i: util::iota(mappedView.count()))
        {
            EXPECT_EQ(view.dimensions[i].step, view.dimensions[i].step);
            EXPECT_EQ(view.dimensions[i].length, view.dimensions[i].length);
        }
    }
    EXPECT_EQ(1, buff.mapCalled);
    EXPECT_EQ(1, buff.unmapCalled);
}
