// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/passes/communications.hpp"

#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <stdexcept>

#include "ade/typed_graph.hpp"

#include "ade/communication/comm_buffer.hpp"
#include "ade/communication/comm_interface.hpp"
#include "ade/communication/callback_connector.hpp"

#include "ade/memory/memory_descriptor.hpp"
#include "ade/memory/memory_descriptor_view.hpp"

#include "ade/util/algorithm.hpp"
#include "ade/util/chain_range.hpp"

#include "ade/memory/alloc.hpp"

namespace
{

using NodeHasher = ade::HandleHasher<ade::Node>;

struct CacheEntry final
{
    std::unordered_set<ade::NodeHandle, NodeHasher> commNodes;
    std::unordered_set<ade::NodeHandle, NodeHasher> producers;
    std::unordered_set<ade::NodeHandle, NodeHasher> consumers;
};

using Cache = std::unordered_map<ade::MemoryDescriptorView*, CacheEntry>;

struct CallbackCacheEntry final
{
    std::unordered_set<ade::NodeHandle, NodeHasher> producers;
    std::unordered_set<ade::NodeHandle, NodeHasher> consumers;
};

using CallbackCache = std::unordered_map<ade::NodeHandle, CallbackCacheEntry, NodeHasher>;


ade::MemoryDescriptorView* findParentView(ade::MemoryDescriptorView* view)
{
    ADE_ASSERT(nullptr != view);

    auto parent = view->getParentView();
    if (nullptr != parent)
    {
        return findParentView(parent);
    }
    return view;
}

void visitProducer(Cache& cache,
                   CallbackCache& callbackCache,
                   const ade::NodeHandle& commNode,
                   const ade::NodeHandle& node,
                   ade::passes::ConnectCommChannels::Context& ctx)
{
    ADE_ASSERT(nullptr != node);
    ADE_ASSERT(ctx.graph.metadata(node).contains<ade::meta::DataObject>());
    auto memDesc = findParentView(ctx.graph.metadata(node).get<ade::meta::DataObject>().dataRef.getView());
    ADE_ASSERT(nullptr != memDesc);
    bool connectedToNode = false;
    for (auto edge: node->inEdges())
    {
        auto srcNode = edge->srcNode();
        if (ctx.graph.metadata(srcNode).contains<ade::meta::NodeInfo>())
        {
            connectedToNode = true;
            callbackCache[commNode].producers.insert(srcNode);
        }
        else if (ctx.graph.metadata(srcNode).contains<ade::meta::DataObject>())
        {
            visitProducer(cache, callbackCache, commNode, srcNode, ctx);
        }
    }

    if (connectedToNode)
    {
        cache[memDesc].producers.insert(node);
        cache[memDesc].commNodes.insert(commNode);
    }
}

void visitConsumer(Cache& cache,
                   CallbackCache& callbackCache,
                   const ade::NodeHandle& commNode,
                   const ade::NodeHandle& node,
                   ade::passes::ConnectCommChannels::Context& ctx)
{
    ADE_ASSERT(nullptr != node);
    ADE_ASSERT(ctx.graph.metadata(node).contains<ade::meta::DataObject>());
    auto memDesc = findParentView(ctx.graph.metadata(node).get<ade::meta::DataObject>().dataRef.getView());
    ADE_ASSERT(nullptr != memDesc);
    bool connectedToNode = false;
    for (auto edge: node->outEdges())
    {
        auto dstNode = edge->dstNode();
        if (ctx.graph.metadata(dstNode).contains<ade::meta::NodeInfo>())
        {
            connectedToNode = true;
            callbackCache[commNode].consumers.insert(dstNode);
        }
        else if (ctx.graph.metadata(dstNode).contains<ade::meta::DataObject>())
        {
            visitConsumer(cache, callbackCache, commNode, dstNode, ctx);
        }
    }

    if (connectedToNode)
    {
        cache[memDesc].consumers.insert(node);
        cache[memDesc].commNodes.insert(commNode);
    }
}

struct DataObject final
{
    ade::MemoryDescriptorRef memory_ref;
    std::vector<ade::NodeHandle> commNodes;
    std::vector<ade::NodeHandle> producers;
    std::vector<ade::NodeHandle> consumers;
};

struct CallbackObject final
{
    ade::NodeHandle commNode;
    std::vector<ade::NodeHandle> producers;
    std::vector<ade::NodeHandle> consumers;
};

struct CommObjects
{
    std::vector<DataObject> dataObjects;
    std::vector<CallbackObject> callbackObjects;
};

CommObjects collectDataObjects(ade::passes::ConnectCommChannels::Context& ctx)
{
    Cache cache;
    CallbackCache callbackCache;
    for (auto node: ctx.graph.nodes())
    {
        auto meta = ctx.graph.metadata(node);
        if (meta.contains<ade::meta::CommNode>())
        {
            for (auto edge: node->inEdges())
            {
                auto srcNode = edge->srcNode();
                visitProducer(cache, callbackCache, node, srcNode, ctx);
            }

            for (auto edge: node->outEdges())
            {
                auto dstNode = edge->dstNode();
                visitConsumer(cache, callbackCache, node, dstNode, ctx);
            }
        }
    }

    CommObjects ret;
    for (auto& obj: cache)
    {
        DataObject newObj;
        newObj.memory_ref = *obj.first;
        newObj.commNodes.reserve(obj.second.commNodes.size());
        newObj.producers.reserve(obj.second.producers.size());
        newObj.consumers.reserve(obj.second.consumers.size());
        ade::util::copy(obj.second.commNodes, std::back_inserter(newObj.commNodes));
        ade::util::copy(obj.second.producers, std::back_inserter(newObj.producers));
        ade::util::copy(obj.second.consumers, std::back_inserter(newObj.consumers));
        ADE_ASSERT(!newObj.commNodes.empty());
        ADE_ASSERT(!newObj.producers.empty());
        ADE_ASSERT(!newObj.consumers.empty());
        ret.dataObjects.emplace_back(std::move(newObj));
    }

    for (auto& obj: callbackCache)
    {
        CallbackObject newObj;
        newObj.commNode = obj.first;
        newObj.producers.reserve(obj.second.producers.size());
        newObj.consumers.reserve(obj.second.consumers.size());
        ade::util::copy(obj.second.producers, std::back_inserter(newObj.producers));
        ade::util::copy(obj.second.consumers, std::back_inserter(newObj.consumers));
        ADE_ASSERT(!newObj.producers.empty());
        ADE_ASSERT(!newObj.consumers.empty());
        ret.callbackObjects.emplace_back(std::move(newObj));
    }
    return ret;
}

/// Fill common part of the BufferDesc
template<typename T>
ade::ICommChannel::BufferDesc fillBufferDesc(T& elem)
{
    auto memRef = elem.memory_ref;
    ADE_ASSERT(nullptr != memRef);
    ade::ICommChannel::BufferDesc bufferDesc;

    // Fill common part of the BufferDesc
    bufferDesc.writersCount = ade::util::checked_cast<int>(elem.producers.size());
    bufferDesc.readersCount = ade::util::checked_cast<int>(elem.consumers.size());
    bufferDesc.memoryRef    = memRef;
    return bufferDesc;
}

class HostBufferImpl final : public ade::IDataBuffer
{
public:
    HostBufferImpl(std::size_t elementSize_,
                   const ade::memory::DynMdSize& size_,
                   const ade::memory::DynMdSize& alignment_);

    HostBufferImpl(const ade::MemoryDescriptorRef& memRef_);

    ~HostBufferImpl();

    // IDataBuffer interface
    virtual MapId map(const Span& span, Access access) override;
    virtual void unmap(const MapId& id) override;
    virtual void finalizeWrite(const ade::IDataBuffer::Span& span) override;
    virtual void finalizeRead(const ade::IDataBuffer::Span& span) override;
    virtual Size alignment(const Span& span) override;

private:
    struct Deleter
    {
        void operator()(void* ptr) const
        {
            ADE_ASSERT(nullptr != ptr);
            ade::aligned_free(ptr);
        }
    };

    std::atomic<int> m_accessCount = {0};
    ade::memory::DynMdSize m_size;
    ade::memory::DynMdSize m_alignment;
    ade::memory::DynMdView<void> m_view;
    std::unique_ptr<void, Deleter> m_memory;
    ade::MemoryDescriptorRef m_memRef;
};

HostBufferImpl::HostBufferImpl(std::size_t elementSize_,
                               const ade::memory::DynMdSize& size_,
                               const ade::memory::DynMdSize& alignment_):
    m_size(size_),
    m_alignment(alignment_),
    m_view(ade::util::alloc_view<ade::memory::MaxDimensions>
           (elementSize_,
            ade::util::memory_range(size_.data(),      size_.dims_count()),
            ade::util::memory_range(alignment_.data(), alignment_.dims_count()),
            [](std::size_t sz, std::size_t align)
            {
                auto ptr = ade::aligned_alloc(sz, align);
                if (nullptr == ptr)
                {
                    ade::throw_error(std::bad_alloc());
                }
                return ptr;
            })),
    m_memory(m_view.mem.data)
{

}

HostBufferImpl::HostBufferImpl(const ade::MemoryDescriptorRef& memRef_):
    m_size(memRef_.span().size()),
    m_memRef(memRef_)
{

}

HostBufferImpl::~HostBufferImpl()
{
    ADE_ASSERT(0 == m_accessCount);
}

ade::IDataBuffer::MapId HostBufferImpl::map(const Span& span, Access /*access*/)
{
    auto view = (nullptr != m_view ? m_view : m_memRef.getExternalView());
    ADE_ASSERT(nullptr != view);
    ADE_ASSERT(span.dims_count() == m_size.dims_count());
    auto accessCount = ++m_accessCount;
    ADE_UNUSED(accessCount);
    ADE_ASSERT(accessCount > 0);
    return MapId{view.slice(span), 0};
}

void HostBufferImpl::unmap(const MapId& /*id*/)
{
    auto accessCount = --m_accessCount;
    ADE_UNUSED(accessCount);
    ADE_ASSERT(accessCount >= 0);
}

void HostBufferImpl::finalizeWrite(const ade::IDataBuffer::Span& /*span*/)
{
    //Nothing
}

void HostBufferImpl::finalizeRead(const ade::IDataBuffer::Span& /*span*/)
{
    //Nothing
}

ade::IDataBuffer::Size HostBufferImpl::alignment(const Span& span)
{
    ADE_ASSERT(span.dims_count() == m_size.dims_count());
    // TODO: report actual alignment
    Size ret;
    ret.redim(span.dims_count());
    ade::util::fill(ret, 1);
    return ret;
}

}

void ade::passes::ConnectCommChannels::operator()(ade::passes::ConnectCommChannels::Context ctx) const
{
    // Step 1:
    // Collect all data objects directly or indirectly connected to comm nodes
    // group them by MemoryDescriptor and by a commnode
    const auto commObjects = collectDataObjects(ctx);

    // Step 2:
    // Check comm channels and callbacks validity
    {
        // Step 2.1
        // Check comm channels validity
        for (auto& elem: commObjects.dataObjects)
        {
            for (auto node: util::chain(util::toRange(elem.producers),
                                        util::toRange(elem.consumers)))
            {
                auto meta = ctx.graph.metadata(node);
                if (!meta.contains<ade::meta::DataObject>() ||
                    !meta.contains<ade::meta::CommChannel>() ||
                    nullptr == meta.get<ade::meta::CommChannel>().channel)
                {
                    throw_error(std::runtime_error("Comm channel wasn't setup properly"));
                }
            }
        }

        // Step 2.2
        // Check comm callbacks validity
        for (auto& elem: commObjects.callbackObjects)
        {
            for (auto node: elem.consumers)
            {
                auto meta = ctx.graph.metadata(node);
                if (!meta.contains<ade::meta::CommConsumerCallback>() ||
                    nullptr == meta.get<ade::meta::CommConsumerCallback>().callback)
                {
                    throw_error(std::runtime_error("Consumer callback metadata error"));
                }
            }
        }
    }

    // Step 3:
    // Connect comm channels
    for (auto& elem: commObjects.dataObjects)
    {
        ade::ICommChannel::BufferDesc bufferDesc = fillBufferDesc(elem);

        // Step 3.1:
        // Collect buffer preferences
        ade::ICommChannel::BufferPrefs summary;
        summary.preferredAlignment.redim(bufferDesc.memoryRef.span().dims_count());
        util::fill(summary.preferredAlignment, 1);
        for (auto node: util::chain(util::toRange(elem.producers),
                                    util::toRange(elem.consumers)))
        {
            auto meta = ctx.graph.metadata(node);
            auto channel = meta.get<ade::meta::CommChannel>().channel;
            ADE_ASSERT(nullptr != channel);
            ade::ICommChannel::BufferPrefs prefs = channel->getBufferPrefs(bufferDesc);
            ADE_ASSERT(prefs.preferredAlignment.dims_count() == summary.preferredAlignment.dims_count());
            for (auto i: util::iota(summary.preferredAlignment.dims_count()))
            {
                ADE_ASSERT(prefs.preferredAlignment[i] > 0);
                // TODO: assert alignment is power of 2
                summary.preferredAlignment[i] =
                        std::max(summary.preferredAlignment[i],
                                 prefs.preferredAlignment[i]);
            }
        }

        // Step 3.2:
        // Try to get buffer from channels
        std::unique_ptr<ade::IDataBuffer> buffer;
        for (auto node: util::chain(util::toRange(elem.producers),
                                    util::toRange(elem.consumers)))
        {
            ADE_ASSERT(nullptr == buffer);
            auto meta = ctx.graph.metadata(node);
            auto channel = meta.get<ade::meta::CommChannel>().channel;
            ADE_ASSERT(nullptr != channel);
            buffer = channel->getBuffer(bufferDesc, summary);
            if (nullptr != buffer)
            {
                break;
            }
        }

        if (nullptr == buffer)
        {
            // Step 3.3:
            // Buffer wasn't allocated by plugins, allocate it by framework
            if (nullptr == bufferDesc.memoryRef.getExternalView())
            {
                buffer.reset(new HostBufferImpl(bufferDesc.memoryRef.elementSize(),
                                                bufferDesc.memoryRef.size(),
                                                summary.preferredAlignment));
            }
            else
            {
                // Use existing buffer (e.g. from non-virtual object)
                buffer.reset(new HostBufferImpl(bufferDesc.memoryRef));
            }
        }

        // Step 3.4:
        // Notify plugins about buffer object
        ADE_ASSERT(nullptr != buffer);
        for (auto node: util::chain(util::toRange(elem.producers),
                                    util::toRange(elem.consumers)))
        {
            auto meta = ctx.graph.metadata(node);
            auto channel = meta.get<ade::meta::CommChannel>().channel;
            channel->setBuffer(ade::DataBufferView(buffer.get(), bufferDesc.memoryRef.span()), bufferDesc);
        }
        std::shared_ptr<ade::IDataBuffer> sharedBuffer(std::move(buffer));
        for (auto commNode: elem.commNodes)
        {
            auto meta = ctx.graph.metadata(commNode);
            meta.get<ade::meta::CommNode>().addDataBuffer(sharedBuffer);
        }
    }

    // Step 4
    // Connect comm objects callbacks
    {
        // Multiple comm nodes can be attached to single producer data object
        // so we need to collect and merge them
        std::unordered_map<ade::NodeHandle, std::vector<std::function<void()>>, NodeHasher> producerCallbacks;
        for (auto& elem: commObjects.callbackObjects)
        {
            ADE_ASSERT(nullptr != elem.commNode);
            ADE_ASSERT(!elem.producers.empty() && !elem.consumers.empty());

            ade::CallbackConnector<> connector(util::checked_cast<int>(elem.producers.size()),
                                               util::checked_cast<int>(elem.consumers.size()));

            // Step 4.1
            // Collect callbacks from consumers
            for (auto& consumer: elem.consumers)
            {
                auto meta = ctx.graph.metadata(consumer);
                auto callback = std::move(meta.get<ade::meta::CommConsumerCallback>().callback);
                ADE_ASSERT(nullptr != callback);
                connector.addConsumerCallback(std::move(callback));
            }

            // Step 4.2
            // Create producer callbacks
            auto resetter = connector.finalize();
            if (nullptr != resetter)
            {
                auto meta = ctx.graph.metadata();
                if (!meta.contains<ade::meta::Finalizers>())
                {
                    meta.set(ade::meta::Finalizers());
                }
                meta.get<ade::meta::Finalizers>().finalizers.emplace_back(std::move(resetter));
            }

            // Step 4.3
            // Collect producer callbacks
            for (auto& producer: elem.producers)
            {
                auto callback = connector.getProducerCallback();
                ADE_ASSERT(nullptr != callback);
                producerCallbacks[producer].emplace_back(std::move(callback));
            }
        }

        // Step 4.4
        // Assign producer callbacks
        for (auto& elem: producerCallbacks)
        {
            auto producer = elem.first;

            auto callbacks = std::move(elem.second);
            ADE_ASSERT(!callbacks.empty());

            auto meta = ctx.graph.metadata(producer);
            if (!meta.contains<ade::meta::CommProducerCallback>())
            {
                meta.set(ade::meta::CommProducerCallback());
            }

            if (1 == callbacks.size())
            {
                // Assign directly
                meta.get<ade::meta::CommProducerCallback>().callback = callbacks[0];
            }
            else
            {
                // Create wrapper to call all callbacks
                struct Connector final
                {
                    std::vector<std::function<void()>> callbacks;

                    void operator()() const
                    {
                        ADE_ASSERT(!callbacks.empty());
                        for (auto& callback: callbacks)
                        {
                            ADE_ASSERT(nullptr != callback);
                            callback();
                        }
                    }
                };

                meta.get<ade::meta::CommProducerCallback>().callback = Connector{std::move(callbacks)};
            }
        }
    }
}

const char* ade::passes::ConnectCommChannels::name()
{
    return "ade::passes::ConnectCommChannels";
}
