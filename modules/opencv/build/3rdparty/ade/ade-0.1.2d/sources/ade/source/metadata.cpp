// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/typed_metadata.hpp"

#include "ade/util/assert.hpp"

namespace ade
{
namespace details
{

bool Metadata::contains(const MetadataId& id) const
{
    ADE_ASSERT(nullptr != id);
    return m_data.end() != m_data.find(id);
}

void Metadata::erase(const MetadataId& id)
{
    m_data.erase(id);
}

std::size_t Metadata::IdHash::operator()(const MetadataId& id) const
{
    return std::hash<decltype(MetadataId::m_id)>()(id.m_id);
}

MetadataId::MetadataId(void* id):
    m_id(id)
{
    ADE_ASSERT(nullptr != m_id);
}

bool MetadataId::operator==(const MetadataId& other) const
{
    return m_id == other.m_id;
}

bool MetadataId::operator!=(const MetadataId& other) const
{
    return m_id != other.m_id;
}

bool MetadataId::isNull() const
{
    return nullptr == m_id;
}

bool operator==(std::nullptr_t, const MetadataId& other)
{
    return other.isNull();
}

bool operator==(const MetadataId& other, std::nullptr_t)
{
    return other.isNull();
}

bool operator!=(std::nullptr_t, const MetadataId& other)
{
    return !other.isNull();
}

bool operator!=(const MetadataId& other, std::nullptr_t)
{
    return !other.isNull();
}

}
}
