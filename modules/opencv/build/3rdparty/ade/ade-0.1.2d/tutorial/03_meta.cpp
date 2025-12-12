// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <string>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>

namespace tutor {
// Custom metadata type
struct MetaInt
{
    int val;
    static const char* name() { return "MetaInt"; }
};
struct MetaFloat
{
    float val;
};
struct MetaString
{
    std::string val;
    static std::string name() { return "MetaString"; }
};

const char* getMetadataName(ade::MetadataNameTag<MetaFloat>)
{
    return "MetaFloat";
};
}

int main(int argc, char *argv[])
{
    ade::Graph graph;
    ade::TypedGraph<tutor::MetaInt,
                    tutor::MetaFloat,
                    tutor::MetaString> tg(graph);

    // Set data to metadata
    tg.metadata().set(tutor::MetaInt{42});
    tg.metadata().set(tutor::MetaFloat{3.14f});
    tg.metadata().set(tutor::MetaString{"ADE"});

    std::cout << "tg.metadata().get<tutor::MetaInt>    = " << tg.metadata().get<tutor::MetaInt>().val   << ",\n"
              << "tg.metadata().get<tutor::MetaFloat>  = " << tg.metadata().get<tutor::MetaFloat>().val << ",\n"
              << "tg.metadata().get<tutor::MetaString> = " << tg.metadata().get<tutor::MetaString>().val
              << std::endl;

    tg.metadata().set(tutor::MetaInt{32});
    std::cout << "tg.metadata().get<tutor::MetaInt> = " << tg.metadata().get<tutor::MetaInt>().val << std::endl;

    std::cout << "tg.metadata().contains<tutor::MetaFloat>() = "
              << std::boolalpha << tg.metadata().contains<tutor::MetaFloat>() << std::endl;
    // You may erase data type
    tg.metadata().erase<tutor::MetaFloat>();

    // Check contains MetaFloat in metadata
    std::cout << "tg.metadata().contains<tutor::MetaFloat>() = "
              << std::boolalpha << tg.metadata().contains<tutor::MetaFloat>() << std::endl;

    std::cout << std::endl;
    ade::NodeHandle nh = tg.createNode();

    // Node metadata can be set/queried exactly in the same way, the only difference is
    // that NodeHandle now needs to be passes to ::metadata()
    std::cout << std::endl;
    tg.metadata(nh).set<tutor::MetaString>(tutor::MetaString{"ColorConvert"});
    std::cout << "Node meta(" << nh << ")=" << tg.metadata(nh).get<tutor::MetaString>().val << std::endl;

    // Same applies to Edge metadata
    ade::NodeHandle nh2 = tg.createNode();
    ade::EdgeHandle eh  = tg.link(nh, nh2);

    tg.metadata(eh).set<tutor::MetaInt>(tutor::MetaInt{100500});
    std::cout << "Edge meta(" << eh << ")=" << tg.metadata(eh).get<tutor::MetaInt>().val << std::endl;

    return 0;
}
