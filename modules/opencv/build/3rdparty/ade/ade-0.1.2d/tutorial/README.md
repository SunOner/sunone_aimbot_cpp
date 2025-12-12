# Welcome to ADE Tutorial

This tutorial will teach you how to use ADE to:
* Construct graphs;
* Apply graph transformations;
* Write backends which execute ADE graphs.

## Contents

Current version of tutorial contains the following steps:

### `hello`

The very basic "hello world"-like example, demonstrating the bare minimum which is required to build an ADE-powered application.

### `add_link_remove`

This example shows general graph concepts, operations, and its semantics.

### `meta`

This example illustrates graph/node/edge manipulation principles.

### `typed_meta`

This example shows how to make metadata access simplified and type-safe at the same time.

### `passes`

A more advanced example shows how to implement basic operation squashing ("fusion") in a 33-SLOC pass.

## `backend`

The most advanced example in the suite provides you with an End-to-End demo on how a graph-based computational engine can be built with ADE.

The following concepts are implemented in this sample:
* A graph construction API built atop of ADE:
  - Named operations;
  - Explicit input/output data ports;
  - Virtual data objects;
* Graph validation passes:
  - Cycle detection, data format & port connection validity;
  - Domain-specific graph metadata resolution ("virtual objects format inference");
* Graph optimization passes:
  - Removing unused operations from the graph;
* Extending compiler with implementation/backend/plugin-specific passes;
* Implementing a backend and an executable (graph interpreter).
