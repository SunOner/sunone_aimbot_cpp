// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file typed_metadata.hpp

#ifndef ADE_TYPED_METADATA_HPP
#define ADE_TYPED_METADATA_HPP

#include <array>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "ade/util/algorithm.hpp"
#include "ade/util/range.hpp"

namespace ade
{
class Graph;
namespace details
{
class Metadata;

class MetadataId final
{
    friend class ::ade::Graph;
    friend class ::ade::details::Metadata;

    MetadataId(void* id);

    void* m_id = nullptr;
public:
    MetadataId() = default;
    MetadataId(std::nullptr_t) {}
    MetadataId(const MetadataId&) = default;
    MetadataId& operator=(const MetadataId&) = default;
    MetadataId& operator=(std::nullptr_t) { m_id = nullptr; return *this; }

    bool operator==(const MetadataId& other) const;
    bool operator!=(const MetadataId& other) const;

    bool isNull() const;
};

bool operator==(std::nullptr_t, const MetadataId& other);
bool operator==(const MetadataId& other, std::nullptr_t);
bool operator!=(std::nullptr_t, const MetadataId& other);
bool operator!=(const MetadataId& other, std::nullptr_t);

class Metadata final
{
    struct IdHash final
    {
        std::size_t operator()(const MetadataId& id) const;
    };

    struct MetadataHolderBase;
    using MetadataHolderPtr = std::unique_ptr<MetadataHolderBase>;
    struct MetadataHolderBase
    {
        virtual ~MetadataHolderBase() = default;
        virtual MetadataHolderPtr clone() const = 0;
    };
    template<typename T>
    struct MetadataHolder : public MetadataHolderBase
    {
        T data;

        MetadataHolder(const MetadataHolder&) = default;
        MetadataHolder(MetadataHolder&&) = default;

        template<typename T1>
        MetadataHolder(T1&& val): data(std::forward<T1>(val)) {}

        MetadataHolder& operator=(const MetadataHolder&) = delete;

        virtual MetadataHolderPtr clone() const override
        {
            return MetadataHolderPtr(new MetadataHolder<T>(*this));
        }
    };

    template<typename T>
    static T& access(MetadataHolderBase& holder)
    {
        using DT = typename std::decay<T>::type;
#if defined(__GXX_RTTI) || defined(_CPPRTTI)
       ADE_ASSERT(nullptr != dynamic_cast<MetadataHolder<DT>*>(&holder));
#endif
       return static_cast<MetadataHolder<DT>*>(&holder)->data;
    }

    template<typename T>
    static const T& access(const MetadataHolderBase& holder)
    {
        using DT = typename std::decay<T>::type;
#if defined(__GXX_RTTI) || defined(_CPPRTTI)
       ADE_ASSERT(nullptr != dynamic_cast<const MetadataHolder<DT>*>(&holder));
#endif
       return static_cast<const MetadataHolder<DT>*>(&holder)->data;
    }

    template<typename T>
    static MetadataHolderPtr createHolder(T&& val)
    {
        using DT = typename std::decay<T>::type;
        return MetadataHolderPtr(new MetadataHolder<DT>{std::forward<T>(val)});
    }
public:
    using MetadataStore = std::unordered_map<MetadataId, MetadataHolderPtr, IdHash>;

    Metadata() = default;

    Metadata(const Metadata&) = delete ;
    Metadata& operator=(const Metadata&) = delete;

    Metadata(Metadata&&) = default;
    Metadata& operator=(Metadata&&) = default;

    bool contains(const MetadataId& id) const;
    void erase(const MetadataId& id);

    template<typename T>
    void set(const MetadataId& id, T&& val)
    {
        ADE_ASSERT(nullptr != id);
        m_data.erase(id);
        m_data.emplace(id, createHolder(std::forward<T>(val)));
    }

    template<typename T>
    T& get(const MetadataId& id)
    {
        ADE_ASSERT(nullptr != id);
        ADE_ASSERT(contains(id));
        return access<T>(*(m_data.find(id)->second));
    }

    template<typename T>
    const T& get(const MetadataId& id) const
    {
        ADE_ASSERT(nullptr != id);
        ADE_ASSERT(contains(id));
        return access<T>(*(m_data.find(id)->second));
    }

private:
    MetadataStore m_data;
};

}

template<bool IsConst, typename... Types>
class TypedMetadata
{
    using IdArray = std::array<ade::details::MetadataId, sizeof...(Types)>;
    using MetadataT = typename std::conditional<IsConst, const ade::details::Metadata&, ade::details::Metadata&>::type;
    const IdArray& m_ids;
    MetadataT m_metadata;

    template<typename T>
    ade::details::MetadataId getId() const
    {
        const auto index = util::type_list_index<typename std::decay<T>::type, Types...>::value;
        return m_ids[index];
    }

public:
    TypedMetadata(const IdArray& ids, MetadataT meta):
        m_ids(ids), m_metadata(meta) {}

    TypedMetadata(const TypedMetadata& other):
        m_ids(other.m_ids), m_metadata(other.m_metadata) {}

    TypedMetadata& operator=(const TypedMetadata&) = delete;

    template<bool, typename...>
    friend class TypedMetadata;

    template<typename T>
    bool contains() const
    {
        return m_metadata.contains(getId<T>());
    }

    template<typename T>
    void erase()
    {
        m_metadata.erase(getId<T>());
    }

    template<typename T>
    void set(T&& val)
    {
        m_metadata.set(getId<T>(), std::forward<T>(val));
    }

    template<typename T>
    auto get() const
    ->typename std::conditional<IsConst, const T&, T&>::type
    {
        return m_metadata.template get<T>(getId<T>());
    }

    template<typename T>
    T get(T&& def) const
    {
        if (contains<T>())
        {
            return m_metadata.template get<T>(getId<T>());
        }
        else
        {
            return std::forward<T>(def);
        }
    }
};

}

#endif // ADE_TYPED_METADATA_HPP
