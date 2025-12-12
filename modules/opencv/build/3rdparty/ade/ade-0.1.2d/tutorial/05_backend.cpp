// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <iomanip>
#include <memory>
#include <functional>
#include <unordered_map>
#include <stack>
#include <vector>
#include <cmath> // std::sqrt

#include <ade/util/assert.hpp>
#include <ade/util/zip_range.hpp>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/memory/memory_types.hpp>
#include <ade/passes/check_cycles.hpp>
#include <ade/passes/topological_sort.hpp>
#include <ade/execution_engine/executable.hpp>
#include <ade/execution_engine/backend.hpp>
#include <ade/execution_engine/execution_engine.hpp>

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

////////////////////////////////////////////////////////////////////////////////
//
// "X" API
//
namespace X
{
struct Op;

// "X" ADE Metadata ////////////////////////////////////////////////////////////
struct Dirty
{
    bool d;
    static const char *name() { return "Dirty";}
};

struct Format
{
    ade::memory::DynMdSize size;
    static const char *name() { return "Format";}
};

struct Type
{
    enum {OP, DATA} t;
    static const char *name() { return "Type"; }
};

struct Operation
{
    // Metadata refers to real execution unit here (under shared_ptr
    // to enable copy c-tor) - in real examples, metadata doesn't refer
    // to worker objets directly but represents it (like vx_kernel in OVX)
    std::string label;
    std::shared_ptr<Op> op;
    static const char *name() { return "Operation";};
};

enum Access    {VIRT, REAL};
enum Direction {IN, OUT};

struct Data
{
    Access acc;
    static const char *name() { return "Data"; }
};

struct Storage
{
    float *ptr;
    bool external;
    static const char *name() { return "Storage"; }
};

struct Const
{
    float value;
    static const char *name() { return "Const"; }
};

struct Link
{
    int port;
    static const char *name() { return "Link"; }
};

struct Internals
{
    std::vector<ade::NodeHandle> operations;
    std::vector<ade::NodeHandle> data;
    static const char *name() { return "Internals"; }
};

struct Visited
{
    bool is_visited;
    static const char *name() { return "Visited"; }
};

using TGraph = ade::TypedGraph
     < Dirty
     , Format
     , Type
     , Operation
     , Data
     , Const
     , Storage
     , Link
     , Internals
     , Visited
     , ade::passes::TopologicalSortData
     >;

using CTGraph = ade::ConstTypedGraph
     < Dirty
     , Format
     , Type
     , Operation
     , Data
     , Const
     , Storage
     , Link
     , Internals
     , Visited
     , ade::passes::TopologicalSortData
     >;


// "X" Operation interface. Can be ADE-independent, but not in this example ////
struct Op
{
    virtual int inputs()  const = 0;
    virtual int outputs() const = 0;
    virtual void validate(const TGraph::CMetadataT&  node_meta,           // in
                          const std::vector<ade::memory::DynMdSize> &ins, // in
                                std::vector<ade::memory::DynMdSize> &outs // out
                          ) const = 0;
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const = 0;
};

// "X" Utility functions
size_t BufferSize(const ade::memory::DynMdSize &sz)
{
    // Buffer size for continious tensor, in float elements (not bytes)
    return std::accumulate(sz.begin(), sz.end(), 1, std::multiplies<size_t>());
}

size_t BufferOffset(const ade::memory::DynMdSize &dims, int i, int j, int k)
{
    return k + dims[0]*j + (dims[0]*dims[1])*i;
}

void Dump(const ade::memory::DynMdSize &sz, const float *p)
{
    for (int i = 0; i < sz[2]; i++)
    {
        for (int j = 0; j < sz[1]; j++)
        {
            for (int k = 0; k < sz[0]; k++)
            {
                std::cout << std::setw(4) << std::setprecision(2) << p[BufferOffset(sz, i, j, k)] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// "X" Graph analysis and transformation passes ////////////////////////////////
namespace Passes
{
    void ExtractInternals(ade::passes::PassContext &ctx)
    {
        // This pass stores function nodes and data nodes in two different lists (for easier access)
        std::cout << "Extracting internals...";

        X::TGraph g(ctx.graph);
        X::Internals ints;
        for (auto nh : g.nodes())
        {
            switch (g.metadata(nh).get<X::Type>().t)
            {
            case X::Type::OP:   ints.operations.push_back(nh); break;
            case X::Type::DATA: ints.data.push_back(nh);       break;
            default: ADE_ASSERT(false);
            }
        }
        std::cout << " Done: "
                  << ints.operations.size() << " operations, "
                  << ints.data.size()       << " data objects."
                  << std::endl;
        g.metadata().set(ints);
    }

    void CheckIOData(ade::passes::PassContext &ctx)
    {
        // This pass checks if there's physical buffers assigned to graph inputs/outputs
        std::cout << "Checking I/O data...";

        X::TGraph g(ctx.graph);
        auto ints = g.metadata().get<X::Internals>();
        bool in_found = false, out_found = false;
        for (auto nh : ints.data)
        {
            if (g.metadata(nh).get<X::Data>().acc == X::REAL)
            {
                if (nh->inNodes().size() == 0 && nh->outNodes().size() != 0)
                {
                    in_found  = true;
                    if (!g.metadata(nh).contains<X::Format>() || !g.metadata(nh).contains<X::Storage>())
                        throw std::logic_error("Format/storage information is not found for input object");
                }
                if (nh->inNodes().size() != 0 && nh->outNodes().size() == 0)
                {
                    out_found = true;
                    if (!g.metadata(nh).contains<X::Format>() || !g.metadata(nh).contains<X::Storage>())
                        throw std::logic_error("Format/storage information is not found for output object");
                }
                if (nh->inNodes().size() != 0 && nh->outNodes().size() != 0)
                    throw std::logic_error("Write through real objects is not supported in this sample");
            }
        }
        if (!in_found)
            throw std::logic_error("No graph inputs found");
        if (!out_found)
            throw std::logic_error("No graph outputs found");

        std::cout << " Done." << std::endl;
    }

    void CheckPorts(ade::passes::PassContext &ctx)
    {
        // This pass checks connection validity for existing edges
        std::cout << "Checking ports connectivity...";

        X::TGraph g(ctx.graph);
        auto ints = g.metadata().get<X::Internals>();
        for (auto nh : ints.data)
        {
            if (nh->inNodes().size() > 1)
                throw std::logic_error("Multiple writers to the same data object");
        }
        for (auto nh : ints.operations)
        {
            auto op_meta = g.metadata(nh).get<X::Operation>();

            std::unordered_map<int, int> port_refs;
            for (auto e : nh->inEdges())
            {
                const int port = g.metadata(e).get<X::Link>().port;
                if (port < 0 || port >= op_meta.op->inputs())
                    throw std::logic_error("Invalid port reference on input");

                const int writes = ++port_refs[port];
                if (writes > 1)
                    throw std::logic_error("Multiple data objects on the same operation input port");
            }

            for (auto out_e : nh->outEdges())
            {
                const int port = g.metadata(out_e).get<X::Link>().port;
                if (port < 0 || port >= op_meta.op->outputs())
                    throw std::logic_error("Invalid port reference on output");
            }
        }
        std::cout << " Done." << std::endl;
    }

    void CheckInputs(ade::passes::PassContext &ctx)
    {
        // This pass checks if all required inputs were actually provided to nodes
        std::cout << "Checking inputs...";

        X::TGraph g(ctx.graph);
        auto ints = g.metadata().get<X::Internals>();
        for (auto nh : ints.operations)
        {
            auto op_meta = g.metadata(nh).get<X::Operation>();
            if (op_meta.op->inputs() != nh->inNodes().size())
                throw std::logic_error("Not all node inputs are connected");
        }

        std::cout << " Done." << std::endl;
    }

    void ResolveMeta(ade::passes::PassContext &ctx)
    {
        // This pass checks asks every operation to check its input data format and fill output data format
        // It is important to iterate over operations in topological order (to ensure validators are
        // called with valid/resolved data on inputs
        std::cout << "BEGIN Metadata resolution" << std::endl;

        const X::TGraph g(ctx.graph);   // constant typed graph (for invoking validators)
        X::TGraph mg(ctx.graph);  // mutable typed graph (for propagating metadata)

        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>();
        for (auto nh : sorted.nodes())
        {
            // Node sorted meta contains ALL nodes (not only operations)
            if (g.metadata(nh).get<X::Type>().t == X::Type::OP)
            {
                auto op_meta = g.metadata(nh).get<X::Operation>();

                // Put input/output handles in an expected order. inputs/outputs are DATA objects!
                std::vector<ade::memory::DynMdSize> inputs (op_meta.op->inputs());
                std::vector<ade::memory::DynMdSize> outputs(op_meta.op->outputs());

                for (auto in_e : nh->inEdges())
                {
                    inputs[ g.metadata(in_e).get<X::Link>().port ] = g.metadata(in_e->srcNode()).get<X::Format>().size;
                }
                std::cout << "  Invoke " << op_meta.label << " validator...";
                op_meta.op->validate(g.metadata(nh), inputs, outputs);
                std::cout << " Done." << std::endl;

                // If there's no formats in output objects, assign metadata from validator
                // if there're formats, compare and check validity
                for (auto out_e : nh->outEdges())
                {
                    auto out_data = out_e->dstNode();
                    auto out_port = g.metadata(out_e).get<X::Link>().port;
                    if (g.metadata(out_data).contains<X::Format>())
                    {
                        if (g.metadata(out_data).get<X::Format>().size != outputs[out_port])
                            throw std::logic_error("Metadata mismatch!");
                    }
                    else
                    {
                        mg.metadata(out_data).set(X::Format{outputs[out_port]});
                    }
                }
            }
        }
        std::cout << "END Metadata resolution" << std::endl;
    }

    void RemoveDeadEnds(ade::passes::PassContext &ctx)
    {
        // This pass identifies which nodes don't contribute to generating REAL outputs, thus can be removed safely.
        // If there were dead ends removed, this pass also sets "Dirty" flag, causing one more compilation round.

        X::TGraph g(ctx.graph);
        X::Internals ints = g.metadata().get<X::Internals>();

        // Put output data objects to stack first (as starting points)
        std::stack<ade::NodeHandle> stack;
        for (auto nh : ints.data)
        {
            if (g.metadata(nh).get<Data>().acc == REAL && nh->outEdges().size() == 0)
                stack.push(nh);
        }

        // Do recursive traversal from outputs to inputs
        while (!stack.empty())
        {
            auto nh = stack.top();
            stack.pop();
            g.metadata(nh).set(Visited{true});

            for (auto src : nh->inNodes())
                stack.push(src);
        }

        // Now do a clean-up, based on visited flag. Iterate over a copy of nodes.
        std::vector<ade::NodeHandle> nodes_copy;
        ade::util::copy(g.nodes(), std::back_inserter(nodes_copy));
        bool dirty = false;
        for (auto nh : nodes_copy)
            if (!g.metadata(nh).contains<Visited>())
            {
                std::cout << "Removing unused " << nh << std::endl;
                g.erase(nh);
                dirty = true;
            }

        g.metadata().set(Dirty{dirty});
    }
} // namespace Passes

// "X" ADE Backend /////////////////////////////////////////////////////////////
class Executable final: public ade::BackendExecutable
{
    CTGraph m_gr;

    std::unordered_map<ade::Node*, std::unique_ptr<float[]> > m_memory;

public:
    explicit Executable(const ade::Graph &gr) : m_gr(gr)
    {
        // Allocate memory for internal buffers
        std::size_t total = 0;
        auto ints = m_gr.metadata().get<X::Internals>();
        for (auto data_obj : ints.data)
        {
            auto data_desc = m_gr.metadata(data_obj).get<X::Data>();
            if (data_desc.acc == VIRT)
            {
                const auto dims = m_gr.metadata(data_obj).get<X::Format>().size;
                const auto len  = X::BufferSize(dims);
                m_memory[data_obj.get()].reset(new float[len]);
                total += len;
            }
        }
        std::cout << "X::Executable: " << sizeof(float)*total << " bytes allocated" << std::endl;
    }

    virtual void run() override
    {
        // Invoke functional units in topological order, supplying neccessary information
        auto sorted = m_gr.metadata().get<ade::passes::TopologicalSortData>();
        for (auto nh : sorted.nodes())
        {
            // Node sorted meta contains ALL nodes (not only operations)
            if (m_gr.metadata(nh).get<X::Type>().t == X::Type::OP)
            {
                auto op_meta = m_gr.metadata(nh).get<X::Operation>();

                // Here we prepare a invocation context for every functional unit we vist,
                // on every run. Sure it can be done only once and stored in Meta in backend-specific
                // format
                const int
                    num_inputs  = op_meta.op->inputs(),
                    num_outputs = op_meta.op->outputs();

                std::vector<const float*>           in_data  (num_inputs);
                std::vector<ade::memory::DynMdSize> in_metas (num_inputs);
                std::vector<float*>                 out_data (num_outputs);
                std::vector<ade::memory::DynMdSize> out_metas(num_outputs);

                for (auto in_e : nh->inEdges())
                {
                    const int port = m_gr.metadata(in_e).get<X::Link>().port;
                    in_data [port] = m_gr.metadata(in_e->srcNode()).get<X::Data>().acc == VIRT
                        ? m_memory[in_e->srcNode().get()].get()
                        : m_gr.metadata(in_e->srcNode()).get<X::Storage>().ptr;
                    in_metas[port] = m_gr.metadata(in_e->srcNode()).get<X::Format>().size;
                }

                for (auto out_e : nh->outEdges())
                {
                    const int port  = m_gr.metadata(out_e).get<X::Link>().port;
                    out_data [port] = m_gr.metadata(out_e->dstNode()).get<X::Data>().acc == VIRT
                        ? m_memory[out_e->dstNode().get()].get()
                        : m_gr.metadata(out_e->dstNode()).get<X::Storage>().ptr;
                    out_metas[port] = m_gr.metadata(out_e->dstNode()).get<X::Format>().size;
                }

                op_meta.op->apply(m_gr.metadata(nh), in_metas, in_data, out_metas, out_data);
            }
        }
    }

    virtual void runAsync() override
    {
        run(); // "Async" version not really async
    }

    virtual void wait() override
    {
    }

    virtual void cancel() override
    {
    }
};

struct Backend: public ade::ExecutionBackend
{
    virtual void setupExecutionEngine(ade::ExecutionEngineSetupContext& ectx) override
    {
        // Add new "init" pass right before "validation" - we do put graph transformations
        // here which then needs to be verified with common routines
        ectx.addPass("init", "put_extra_buffers", [](ade::passes::PassContext &ctx) {
                // On the backend level, we know our operations don't support partial outputs -
                // i.e., when there's operation with multiple outputs and only one of them is
                // consumed. So put extra data virtual objects to make all outputs valid
                X::TGraph g(ctx.graph);
                auto ints = g.metadata().get<X::Internals>();
                for (auto nh : ints.operations)
                {
                    auto op_meta = g.metadata(nh).get<X::Operation>();
                    auto op = op_meta.op;
                    if (op->outputs() == nh->outEdges().size())
                        continue;

                    std::vector<ade::EdgeHandle> out_edges(op->outputs());
                    for (auto eh : nh->outEdges())
                    {
                        out_edges[ g.metadata(eh).get<Link>().port ] = eh;
                    }
                    for (auto it : ade::util::indexed(out_edges))
                    {
                        auto port = ade::util::index(it);
                        auto oe   = ade::util::value(it);
                        if (!oe.get())
                        {
                            auto dh = g.createNode();
                            g.metadata(dh).set(Type{Type::DATA});
                            g.metadata(dh).set(Data{VIRT});
                            auto de = g.link(nh, dh);
                            g.metadata(de).set(Link{static_cast<int>(port)});
                            std::cout << "Put extra data for " << op_meta.label << " port " << port << std::endl;
                        }
                    }
                }
            });
    }

    virtual std::unique_ptr<ade::BackendExecutable> createExecutable(const ade::Graph& gr) override
    {
        return make_unique<Executable>(gr);
    }
};

// "X" Operation implementations ///////////////////////////////////////////////
struct UnaryOp: public Op
{
    virtual int inputs()  const override final { return 1; }
    virtual int outputs() const override final { return 1; }

    virtual void validate(const TGraph::CMetadataT& node_meta,            // in
                          const std::vector<ade::memory::DynMdSize> &ins, // in
                                std::vector<ade::memory::DynMdSize> &outs // out
                          ) const override
    {
        outs[0] = ins[0];
    }
};

struct UnaryCOp: public UnaryOp
{
    virtual void validate(const TGraph::CMetadataT& node_meta,            // in
                          const std::vector<ade::memory::DynMdSize> &ins, // in
                                std::vector<ade::memory::DynMdSize> &outs // out
                          ) const override
    {
        // Check if this node meta also contains a constant value required for operation
        if (!node_meta.contains<Const>())
            throw std::logic_error("No constant data found for operation with Constant!");
        return UnaryOp::validate(node_meta, ins, outs);
    }
};

struct BinaryOp: public Op
{
    virtual int inputs()  const override final { return 2; }
    virtual int outputs() const override final { return 1; }

    virtual void validate(const TGraph::CMetadataT &node_meta,            // in
                          const std::vector<ade::memory::DynMdSize> &ins, // in
                                std::vector<ade::memory::DynMdSize> &outs // out
                          ) const override
    {
        if (ins[0] != ins[1])
            throw std::logic_error("Binary operation inputs must have equal sizes!");
        outs[0] = ins[0];
    }
};

template<typename F>
void Apply(const ade::memory::DynMdSize &dims, F f)
{
    ADE_ASSERT(dims.dims_count() == 3);
    for (int i = 0; i < dims[2]; i++)
        for (int j = 0; j < dims[1]; j++)
            for (int k = 0; k < dims[0]; k++)
            {
                f(X::BufferOffset(dims, i, j, k));
            }
}

struct Add: public BinaryOp
{
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const override
    {
        std::cout << "ADD" << std::endl;
        ADE_ASSERT(in_metas.size() == 2);
        ADE_ASSERT(out_metas.size() == 1);
        Apply(out_metas[0], [&](const int p) {
                out_data[0][p] = in_data[0][p] + in_data[1][p];
            });
    }
};

struct Sub: public BinaryOp
{
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const override
    {
        std::cout << "SUB" << std::endl;
        ADE_ASSERT(in_metas.size() == 2);
        ADE_ASSERT(out_metas.size() == 1);
        Apply(out_metas[0], [&](const int p) {
                out_data[0][p] = in_data[0][p] - in_data[1][p];
            });
    }
};

struct Sqrt: public UnaryOp
{
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const override
    {
        std::cout << "SQRT" << std::endl;
        ADE_ASSERT(in_metas.size() == out_metas.size() == 1);
        Apply(out_metas[0], [&](const int p) {
                out_data[0][p] = std::sqrt(in_data[0][p]);
            });
    }
};

struct AddC: public UnaryCOp
{
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const override
    {
        std::cout << "ADDC" << std::endl;
        ADE_ASSERT(in_metas.size() == out_metas.size() == 1);
        const float cval = node_meta.get<Const>().value;
        Apply(out_metas[0], [&](const int p) {
                out_data[0][p] = in_data[0][p] + cval;
            });
    }
};

struct SubC: public UnaryCOp
{
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const override
    {
        std::cout << "SUBC" << std::endl;
        ADE_ASSERT(in_metas.size() == out_metas.size() == 1);
        const float cval = node_meta.get<Const>().value;
        Apply(out_metas[0], [&](const int p) {
                out_data[0][p] = in_data[0][p] - cval;
            });
    }
};

struct MulC: public UnaryCOp
{
    virtual void apply(const CTGraph::CMetadataT&  node_meta,                // in
                       const std::vector<ade::memory::DynMdSize> &in_metas,  // in
                       const std::vector<const float *>          &in_data,   // in
                       const std::vector<ade::memory::DynMdSize> &out_metas, // in
                       const std::vector<float *>                &out_data   // out
                       ) const override
    {
        std::cout << "MULC" << std::endl;
        ADE_ASSERT(in_metas.size() == out_metas.size() == 1);
        const float cval = node_meta.get<Const>().value;
        Apply(out_metas[0], [&](const int p) {
                out_data[0][p] = in_data[0][p] * cval;
            });
    }
};


// "X" Graph construction API (DSL) ////////////////////////////////////////////
class Builder
{
    TGraph &m_g;

    std::unordered_map<std::string, ade::NodeHandle> m_ops;
    std::unordered_map<std::string, ade::NodeHandle> m_objs;

    ade::NodeHandle data_handle(const char *name)
    {
        ade::NodeHandle nh;
        auto it = m_objs.find(name);
        if (it == m_objs.end())
        {
            nh = m_g.createNode();
            m_objs[name] = nh;
            m_g.metadata(nh).set(Type{Type::DATA});
            m_g.metadata(nh).set(Data{REAL});
        }
        else
        {
            nh = it->second;
        }
        return nh;
    }

public:
    explicit Builder(TGraph &g) : m_g(g)
    {
        m_g.metadata().set(Dirty{false});
    }

    template<typename T>
    Builder& node(const char *name)
    {
        ADE_ASSERT(m_ops.find(name) == m_ops.end());

        ade::NodeHandle nh = m_g.createNode();
        m_g.metadata(nh).set(Type{Type::OP});
        m_g.metadata(nh).set(Operation{name, std::make_shared<T>()});
        m_ops[name] = nh;
        return *this;
    }

    Builder& cnst(const char *name, float v)
    {
        auto op_it = m_ops.find(name);
        ADE_ASSERT(op_it != m_ops.end());
        m_g.metadata(op_it->second).set(Const{v});
        return *this;
    }

    Builder& link(const char *src, int src_port,
                  const char *dst, int dst_port)
    {
        auto src_it = m_ops.find(src);
        auto dst_it = m_ops.find(dst);
        ADE_ASSERT(src_it != m_ops.end());
        ADE_ASSERT(dst_it != m_ops.end());

        // Find data object among src's outputs (one can be read
        // by multiple other nodes)
        ade::NodeHandle
            nh,
            srch = src_it->second,
            dsth = dst_it->second;

        for (auto eh : srch->outEdges()) // outEdges() don't wor with find_if :(
        {
            if (m_g.metadata(eh).get<X::Link>().port == src_port)
            {
                nh = eh->dstNode();
                break;
            }
        }
        // create a new one, of the output was't referenced already
        if (!nh.get())
        {
            nh = m_g.createNode();
            m_g.metadata(nh).set(Type{Type::DATA});
            m_g.metadata(nh).set(Data{VIRT});

            ade::EdgeHandle e1 = m_g.link(srch, nh);
            m_g.metadata(e1).set(Link{src_port});
        }

        ade::EdgeHandle e2 = m_g.link(nh, dsth);
        m_g.metadata(e2).set(Link{dst_port});

        return *this;
    }

    Builder& data(const char *name, Direction d, const char *op, int port)
    {
        auto op_it = m_ops.find(op);
        ADE_ASSERT(op_it != m_ops.end());

        ade::NodeHandle nh = data_handle(name);
        ade::EdgeHandle eh;
        if (d == IN)
        {
            eh = m_g.link(nh, op_it->second);
        }
        else if (d == OUT)
        {
            eh = m_g.link(op_it->second, nh);
        }
        else ADE_ASSERT(false);
        m_g.metadata(eh).set(Link{port});
        return *this;
    }

    Builder& bind(const char *name, float *data, const ade::memory::DynMdSize& dims)
    {
        ade::NodeHandle nh = data_handle(name);
        m_g.metadata(nh).set(Format{dims});
        m_g.metadata(nh).set(Storage{data,true});
        return *this;
    }
};

}
//
////////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[])
{
    ade::Graph graph;

    // Build operation graph
    //
    //           (A)     (B)     (D)    (F)    (G)
    // (in1) --> Add -> AddC -> MulC -> Sub -> Sqrt -> (out)
    //           ^                      ^
    //           :       (C)     (E)    :      (H)
    // (in2) ----'----> AddC -> SubC ---'----> MulC
    //                       :                  :
    //                       :  (I)             :
    //                       '> Sub <-----------'

    X::TGraph xgr(graph);
    X::Builder builder(xgr);

    builder
        .node<X::Add >("A")
        .node<X::AddC>("B")
        .node<X::AddC>("C")
        .node<X::MulC>("D")
        .node<X::SubC>("E")
        .node<X::Sub >("F")
        .node<X::Sqrt>("G")
        .node<X::MulC>("H")
        .node<X::Sub >("I")

        .cnst("B", 1.f)
        .cnst("C", 2.f)
        .cnst("D", 3.f)
        .cnst("E", 4.f)
        .cnst("H", 5.f)

        .link("A", 0, "B", 0)
        .link("B", 0, "D", 0)
        .link("C", 0, "E", 0)
        .link("D", 0, "F", 0)
        .link("E", 0, "F", 1)
        .link("F", 0, "G", 0)
        .link("E", 0, "H", 0)
        .link("C", 0, "I", 0)
        .link("H", 0, "I", 1)

        .data("in1", X::IN, "A", 0)
        .data("in2", X::IN, "A", 1)
        .data("in2", X::IN, "C", 0)
        .data("out", X::OUT,"G", 0);

    // Bind data objects to data
    const ade::memory::DynMdSize sz_in({4, 4, 3});

    std::vector<float> in_1 = {
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f,

        2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f,

        3.f, 3.f, 3.f, 3.f,
        3.f, 3.f, 3.f, 3.f,
        3.f, 3.f, 3.f, 3.f,
        3.f, 3.f, 3.f, 3.f,
    };
    std::vector<float> in_2 = {
        4.f, 4.f, 4.f, 4.f,
        4.f, 4.f, 4.f, 4.f,
        4.f, 4.f, 4.f, 4.f,
        4.f, 4.f, 4.f, 4.f,

        5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f,

        6.f, 6.f, 6.f, 6.f,
        6.f, 6.f, 6.f, 6.f,
        6.f, 6.f, 6.f, 6.f,
        6.f, 6.f, 6.f, 6.f,
    };

    // In this example, size/buf_size/strides for input/output buffers area actually
    // equal, but still defined separately for modifications/testing purposes (demo).

    const ade::memory::DynMdSize sz_out = sz_in;
    const size_t buf_size_out = X::BufferSize(sz_out);

    std::vector<float> out(buf_size_out);

    builder
        .bind("in1", in_1.data(), sz_in)
        .bind("in2", in_2.data(), sz_in)
        .bind("out", out.data(),  sz_out);

    // Setup execution engine: passes
    ade::ExecutionEngine engine;
    engine.addPassStage("init");
    engine.addPass("init", "extract_internals",         X::Passes::ExtractInternals);

    engine.addPassStage("validation");
    engine.addPass("validation", "extract_internals",   X::Passes::ExtractInternals);
    engine.addPass("validation", "check_cycles",      ade::passes::CheckCycles());
    engine.addPass("validation", "check_data",          X::Passes::CheckIOData);
    engine.addPass("validation", "check_ports",         X::Passes::CheckPorts);
    engine.addPass("validation", "check_inputs",        X::Passes::CheckInputs);
    engine.addPass("validation", "topological_sort",  ade::passes::TopologicalSort());
    engine.addPass("validation", "resolve_meta",        X::Passes::ResolveMeta);

    engine.addPassStage("optimization");
    engine.addPass("optimization", "remove_dead_ends",  X::Passes::RemoveDeadEnds);

    // Setup execution engine: backends
    engine.addBackend(make_unique<X::Backend>());
    engine.setupBackends();

    // Finalize & run
    int round = 0;
    do
    {
        std::cout << "====================\nCompilation round " << round++ << "\n" << std::endl;
        engine.runPasses(graph);
        std::cout << std::endl;
    }
    while (xgr.metadata().get<X::Dirty>().d);
    std::cout << "Compilation is done\n====================" << std::endl;

    auto exec = engine.createExecutable(graph);
    exec->run();

    // Inspect the result
    std::cout << "====================\n"
              << "Output:\n" << std::endl;
    X::Dump(sz_out, out.data());
    return 0;
}
