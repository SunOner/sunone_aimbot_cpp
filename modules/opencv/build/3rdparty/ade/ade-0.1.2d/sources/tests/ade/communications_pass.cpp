// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <functional>

#include <gtest/gtest.h>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>

#include <ade/communication/comm_interface.hpp>
#include <ade/communication/comm_buffer.hpp>

#include <ade/memory/memory_descriptor.hpp>
#include <ade/memory/memory_descriptor_view.hpp>
#include <ade/memory/memory_descriptor_ref.hpp>

#include <ade/metatypes/metatypes.hpp>

#include <ade/passes/communications.hpp>

#include <ade/util/iota_range.hpp>

//=================== Comm channels tests=======================================

namespace
{

class TestCommChannel : public ade::ICommChannel
{
public:
    // ICommChannel interface
    virtual BufferPrefs getBufferPrefs(const BufferDesc& desc) override;
    virtual std::unique_ptr<ade::IDataBuffer> getBuffer(const BufferDesc& desc, const BufferPrefs& prefs) override;
    virtual void setBuffer(const ade::DataBufferView& buffer, const BufferDesc& desc) override;

    ade::IDataBuffer* buff = nullptr;
};

ade::ICommChannel::BufferPrefs TestCommChannel::getBufferPrefs(const BufferDesc& desc)
{
    BufferPrefs ret;
    ret.preferredAlignment.redim(desc.memoryRef.size().dims_count());
    ade::util::fill(ret.preferredAlignment, 1);
    return ret;
}

std::unique_ptr<ade::IDataBuffer> TestCommChannel::getBuffer(const BufferDesc& /*desc*/,
                                                             const BufferPrefs& /*prefs*/)
{
    return nullptr;
}

void TestCommChannel::setBuffer(const ade::DataBufferView& buffer, const BufferDesc& desc)
{
    EXPECT_EQ(nullptr, buff);
    EXPECT_NE(nullptr, buffer.getBuffer());
    buff = buffer.getBuffer();
    EXPECT_EQ(desc.memoryRef.span(), buffer.getSpan());
}

}


TEST(CommTest, CommChannelSimple)
{
    // (src)->[img]->(comm)->[img]->(dst)

    ade::Graph srcGr;
    using GraphT = ade::TypedGraph<ade::meta::NodeInfo,
                                   ade::meta::DataObject,
                                   ade::meta::CommNode,
                                   ade::meta::CommChannel,
                                   ade::meta::CommConsumerCallback,
                                   ade::meta::CommProducerCallback,
                                   ade::meta::Finalizers>;
    GraphT gr(srcGr);

    auto srcNode  = gr.createNode();
    auto srcImg   = gr.createNode();
    auto dstNode  = gr.createNode();
    auto dstImg   = gr.createNode();
    auto commNode = gr.createNode();

    gr.link(srcNode, srcImg);
    gr.link(srcImg, commNode);
    gr.link(commNode, dstImg);
    gr.link(dstImg, dstNode);

    gr.metadata(srcNode).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());
    gr.metadata(dstNode).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());

    ade::MemoryDescriptor desc(1, {10,10});
    ade::MemoryDescriptorView view(desc, {ade::util::Span(0,10),
                                          ade::util::Span(0, 10)});

    gr.metadata(srcImg).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(dstImg).set<ade::meta::DataObject>(ade::meta::DataObject());

    gr.metadata(srcImg).get<ade::meta::DataObject>().dataRef = view;
    gr.metadata(dstImg).get<ade::meta::DataObject>().dataRef = view;

    gr.metadata(commNode).set<ade::meta::CommNode>(ade::meta::CommNode(1));

    ade::passes::PassContext ctx{srcGr};
    ASSERT_THROW(ade::passes::ConnectCommChannels()(ctx), std::runtime_error);

    gr.metadata(srcImg).set<ade::meta::CommChannel>(ade::meta::CommChannel());
    gr.metadata(dstImg).set<ade::meta::CommChannel>(ade::meta::CommChannel());

    ASSERT_THROW(ade::passes::ConnectCommChannels()(ctx), std::runtime_error);

    auto chan1 = std::make_shared<TestCommChannel>();
    auto chan2 = std::make_shared<TestCommChannel>();

    gr.metadata(srcImg).get<ade::meta::CommChannel>().channel = chan1;
    gr.metadata(dstImg).get<ade::meta::CommChannel>().channel = chan2;

    ASSERT_THROW(ade::passes::ConnectCommChannels()(ctx), std::runtime_error);

    gr.metadata(dstNode).set(ade::meta::CommConsumerCallback{});

    ASSERT_THROW(ade::passes::ConnectCommChannels()(ctx), std::runtime_error);

    int callbackCallCount = 0;

    gr.metadata(dstNode).get<ade::meta::CommConsumerCallback>().callback = [&]()
    {
        ++callbackCallCount;
    };

    ade::passes::ConnectCommChannels()(ctx);

    ASSERT_TRUE(gr.metadata(srcNode).contains<ade::meta::CommProducerCallback>());
    auto producerCallback = gr.metadata(srcNode).get<ade::meta::CommProducerCallback>().callback;
    ASSERT_NE(nullptr, producerCallback);
    ASSERT_EQ(0, callbackCallCount);

    auto finalizers = gr.metadata().get(ade::meta::Finalizers{}).finalizers;

    for (auto i: ade::util::iota(10))
    {
        (void)i;
        callbackCallCount = 0;
        producerCallback();
        ASSERT_EQ(1, callbackCallCount);

        for (auto& fin: finalizers)
        {
            fin();
        }
    }
}

TEST(CommTest, CommChannelComplexDeps)
{
    //
    // (src1)->[img]
    //              \
    //                ->[img]->(comm)->[img]->(dst1)
    //              /
    // (src2)->[img]
    //              \
    //                ->[img]->(comm)->[img]->(dst2)
    //              /
    // (src3)->[img]
    //
    // (src4)->[img]->(comm)->[img]->(dst3)

    ade::Graph srcGr;
    using GraphT = ade::TypedGraph<ade::meta::NodeInfo,
                                   ade::meta::DataObject,
                                   ade::meta::CommNode,
                                   ade::meta::CommChannel,
                                   ade::meta::CommConsumerCallback,
                                   ade::meta::CommProducerCallback,
                                   ade::meta::Finalizers>;
    GraphT gr(srcGr);

    auto srcNode1 = gr.createNode();
    auto srcNode2 = gr.createNode();
    auto srcNode3 = gr.createNode();
    auto srcNode4 = gr.createNode();

    auto srcImg1 = gr.createNode();
    auto srcImg2 = gr.createNode();
    auto srcImg3 = gr.createNode();
    auto srcImg4 = gr.createNode();

    auto tempImg1 = gr.createNode();
    auto tempImg2 = gr.createNode();

    auto commNode1 = gr.createNode();
    auto commNode2 = gr.createNode();
    auto commNode3 = gr.createNode();

    auto dstImg1 = gr.createNode();
    auto dstImg2 = gr.createNode();
    auto dstImg3 = gr.createNode();

    auto dstNode1 = gr.createNode();
    auto dstNode2 = gr.createNode();
    auto dstNode3 = gr.createNode();

    gr.link(srcNode1, srcImg1);
    gr.link(srcNode2, srcImg2);
    gr.link(srcNode3, srcImg3);
    gr.link(srcNode4, srcImg4);

    gr.link(srcImg1, tempImg1);
    gr.link(srcImg2, tempImg1);
    gr.link(srcImg2, tempImg2);
    gr.link(srcImg3, tempImg2);

    gr.link(tempImg1, commNode1);
    gr.link(tempImg2, commNode2);
    gr.link(srcImg4,  commNode3);

    gr.link(commNode1, dstImg1);
    gr.link(commNode2, dstImg2);
    gr.link(commNode3, dstImg3);

    gr.link(dstImg1, dstNode1);
    gr.link(dstImg2, dstNode2);
    gr.link(dstImg3, dstNode3);

    ade::MemoryDescriptor desc1(1, {10,30});
    ade::MemoryDescriptorView srcView1(desc1, {ade::util::Span(0,10),
                                               ade::util::Span(0,30)});
    ade::MemoryDescriptor desc2(1, {20,20});
    ade::MemoryDescriptorView srcView2(desc2, {ade::util::Span(0,20),
                                               ade::util::Span(0,20)});

    ade::MemoryDescriptorView view1(srcView1, {ade::util::Span(0,10),
                                               ade::util::Span(0, 10)});
    ade::MemoryDescriptorView view2(srcView1, {ade::util::Span(0,10),
                                               ade::util::Span(10,20)});
    ade::MemoryDescriptorView view3(srcView1, {ade::util::Span(0,10),
                                               ade::util::Span(20,30)});

    ade::MemoryDescriptorView view5(srcView1, {ade::util::Span(0,10),
                                               ade::util::Span(5, 15)});
    ade::MemoryDescriptorView view6(srcView1, {ade::util::Span(0,10),
                                               ade::util::Span(15,25)});

    ade::MemoryDescriptorView view7(srcView2, {ade::util::Span(0,20),
                                               ade::util::Span(0,20)});

    gr.metadata(srcNode1).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());
    gr.metadata(srcNode2).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());
    gr.metadata(srcNode3).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());
    gr.metadata(srcNode4).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());

    gr.metadata(dstNode1).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());
    gr.metadata(dstNode2).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());
    gr.metadata(dstNode3).set<ade::meta::NodeInfo>(ade::meta::NodeInfo());

    gr.metadata(commNode1).set<ade::meta::CommNode>(ade::meta::CommNode(2));
    gr.metadata(commNode2).set<ade::meta::CommNode>(ade::meta::CommNode(2));
    gr.metadata(commNode3).set<ade::meta::CommNode>(ade::meta::CommNode(1));

    gr.metadata(srcImg1).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(srcImg2).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(srcImg3).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(srcImg4).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(tempImg1).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(tempImg2).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(dstImg1).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(dstImg2).set<ade::meta::DataObject>(ade::meta::DataObject());
    gr.metadata(dstImg3).set<ade::meta::DataObject>(ade::meta::DataObject());

    gr.metadata(srcImg1).get<ade::meta::DataObject>().dataRef = view1;
    gr.metadata(srcImg2).get<ade::meta::DataObject>().dataRef = view2;
    gr.metadata(srcImg3).get<ade::meta::DataObject>().dataRef = view3;

    gr.metadata(tempImg1).get<ade::meta::DataObject>().dataRef = view5;
    gr.metadata(tempImg2).get<ade::meta::DataObject>().dataRef = view6;

    gr.metadata(dstImg1).get<ade::meta::DataObject>().dataRef = view5;
    gr.metadata(dstImg2).get<ade::meta::DataObject>().dataRef = view6;

    gr.metadata(srcImg4).get<ade::meta::DataObject>().dataRef = view7;
    gr.metadata(dstImg3).get<ade::meta::DataObject>().dataRef = view7;

    ade::passes::PassContext ctx{srcGr};
    ASSERT_THROW(ade::passes::ConnectCommChannels()(ctx), std::runtime_error);

    auto chan1 = std::make_shared<TestCommChannel>();
    auto chan2 = std::make_shared<TestCommChannel>();
    auto chan3 = std::make_shared<TestCommChannel>();
    auto chan4 = std::make_shared<TestCommChannel>();
    auto chan5 = std::make_shared<TestCommChannel>();
    auto chan6 = std::make_shared<TestCommChannel>();
    auto chan7 = std::make_shared<TestCommChannel>();

    gr.metadata(srcImg1).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan1});
    gr.metadata(srcImg2).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan2});
    gr.metadata(srcImg3).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan3});
    gr.metadata(srcImg4).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan4});
    gr.metadata(dstImg1).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan5});
    gr.metadata(dstImg2).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan6});
    gr.metadata(dstImg3).set<ade::meta::CommChannel>(ade::meta::CommChannel{chan7});

    int consumerCallbackCalled1 = 0;
    int consumerCallbackCalled2 = 0;
    int consumerCallbackCalled3 = 0;

    gr.metadata(dstNode1).set(ade::meta::CommConsumerCallback{[&]()
    {
        ++consumerCallbackCalled1;
    }});

    gr.metadata(dstNode2).set(ade::meta::CommConsumerCallback{[&]()
    {
        ++consumerCallbackCalled2;
    }});

    gr.metadata(dstNode3).set(ade::meta::CommConsumerCallback{[&]()
    {
        ++consumerCallbackCalled3;
    }});

    ade::passes::ConnectCommChannels()(ctx);

    auto buff1 = chan1->buff;
    auto buff2 = chan2->buff;
    auto buff3 = chan3->buff;
    auto buff4 = chan4->buff;

    auto buff5 = chan5->buff;
    auto buff6 = chan6->buff;
    auto buff7 = chan7->buff;

    // First group
    EXPECT_NE(nullptr, buff1);
    EXPECT_EQ(buff1, buff2);
    EXPECT_EQ(buff1, buff3);
    EXPECT_EQ(buff1, buff5);
    EXPECT_EQ(buff1, buff6);

    // Second group
    EXPECT_NE(nullptr, buff4);
    EXPECT_EQ(buff4, buff7);

    EXPECT_NE(buff1, buff4);

    ASSERT_TRUE(gr.metadata(srcNode1).contains<ade::meta::CommProducerCallback>());
    ASSERT_TRUE(gr.metadata(srcNode2).contains<ade::meta::CommProducerCallback>());
    ASSERT_TRUE(gr.metadata(srcNode3).contains<ade::meta::CommProducerCallback>());
    ASSERT_TRUE(gr.metadata(srcNode4).contains<ade::meta::CommProducerCallback>());

    auto producerCallback1 = gr.metadata(srcNode1).get<ade::meta::CommProducerCallback>().callback;
    auto producerCallback2 = gr.metadata(srcNode2).get<ade::meta::CommProducerCallback>().callback;
    auto producerCallback3 = gr.metadata(srcNode3).get<ade::meta::CommProducerCallback>().callback;
    auto producerCallback4 = gr.metadata(srcNode4).get<ade::meta::CommProducerCallback>().callback;

    ASSERT_NE(nullptr, producerCallback1);
    ASSERT_NE(nullptr, producerCallback2);
    ASSERT_NE(nullptr, producerCallback3);
    ASSERT_NE(nullptr, producerCallback4);

    ASSERT_EQ(0, consumerCallbackCalled1);
    ASSERT_EQ(0, consumerCallbackCalled2);
    ASSERT_EQ(0, consumerCallbackCalled3);

    auto finalizers = gr.metadata().get(ade::meta::Finalizers{}).finalizers;

    ASSERT_TRUE(!finalizers.empty());

    for (auto i: ade::util::iota(10))
    {
        (void)i;
        consumerCallbackCalled1 = 0;
        consumerCallbackCalled2 = 0;
        consumerCallbackCalled3 = 0;

        producerCallback1();

        ASSERT_EQ(0, consumerCallbackCalled1);
        ASSERT_EQ(0, consumerCallbackCalled2);
        ASSERT_EQ(0, consumerCallbackCalled3);

        producerCallback2();

        ASSERT_EQ(1, consumerCallbackCalled1);
        ASSERT_EQ(0, consumerCallbackCalled2);
        ASSERT_EQ(0, consumerCallbackCalled3);

        producerCallback3();

        ASSERT_EQ(1, consumerCallbackCalled1);
        ASSERT_EQ(1, consumerCallbackCalled2);
        ASSERT_EQ(0, consumerCallbackCalled3);

        producerCallback4();

        ASSERT_EQ(1, consumerCallbackCalled1);
        ASSERT_EQ(1, consumerCallbackCalled2);
        ASSERT_EQ(1, consumerCallbackCalled3);

        for (auto& fin: finalizers)
        {
            fin();
        }
    }
}
