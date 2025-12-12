// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file communications.hpp

#ifndef ADE_COMMUNICATIONS_HPP
#define ADE_COMMUNICATIONS_HPP

#include "ade/passes/pass_base.hpp"

#include "ade/metatypes/metatypes.hpp"

namespace ade
{
namespace passes
{

struct ConnectCommChannels final
{
    using Context = ade::passes::TypedPassContext<ade::meta::CommNode,
                                                  ade::meta::DataObject,
                                                  ade::meta::CommChannel,
                                                  ade::meta::NodeInfo,
                                                  ade::meta::CommConsumerCallback,
                                                  ade::meta::CommProducerCallback,
                                                  ade::meta::Finalizers>;
    void operator()(Context ctx) const;
    static const char* name();
};

}
}

#endif // ADE_COMMUNICATIONS_HPP
