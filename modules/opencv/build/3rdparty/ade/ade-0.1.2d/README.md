# ADE Framework

[![Build Status](https://github.com/opencv/ade/actions/workflows/workflow.yaml/badge.svg)](https://github.com/opencv/ade/actions/workflows/workflow.yaml)

## Intro

ADE Framework is a graph construction, manipulation, and processing
framework.  ADE Framework is suitable for organizing data flow
processing and execution.

## Prerequisites and building

The only prerequisites for library are CMake 3.5+ and a C++11
compiler.

Building:

    $ mkdir build
    $ cd build
    $ cmake /path/to/ade/repository
    $ make -j

After a successfull compilation binaries should reside in `./lib` and
`./bin` directories. Use

    $ make test

to run ADE Framework test suite (ADE Framework tests + utility tests).

If you want to build tutorial samples set `-DBUILD_ADE_TUTORIAL=ON` to
cmake.

Building with tutorial:

    $ cmake -DBUILD_ADE_TUTORIAL=ON /path/to/ade/repository
    $ make -j

Additional information on tutorial samples can be found in
`./tutorial/README.md`.

If you want to build library tests set `-DENABLE_ADE_TESTING=ON` to cmake.
Tests require gtest (https://github.com/google/googletest/releases).

Building gtest:

    $ cmake -DCMAKE_INSTALL_PREFIX=/gtest/install/path  /path/to/gtest
    $ make && make install

Building with tests:

    $ cmake -DENABLE_ADE_TESTING=ON -DGTEST_ROOT=/gtest/install/path /path/to/ade/repository
    $ make -j

You can build library with hardened asserts via
`-DFORCE_ADE_ASSERTS=ON` option, forcing `ADE_ASSERT` to be present
even in release builds.

This library only does error checking in debug or `FORCE_ADE_ASSERTS=ON`
builds due to performance reasons.  Library doesn't process any user
input directly and doesn't read any files or sockets.  If you want to
use this library to process any input from external source you must
validate it before doing any library calls.

To build documentation set `-DBUILD_ADE_DOCUMENTATION=ON`. Documentation
can be found in `./doc` directory. Doxygen is required.

## Support

Current ADE Framework support model is:
* ADE Framework is mainly used as a building block for other projects.
* ADE Framework major version are synchronized with that other
  projects releases.
* ADE Framework accepts pull requests but is stabilized prior to a
  major parent project release only.

## Branches

* `master` -- a default development branch. All further PRs are merged
  there by default. Projects which use ADE pull code from other
  (stable) branches by default.
  - `master` is not guaranteed to be stable (some tests may be failing
    on some platforms).
  - `master` is stabilized before other components major release.
* `release_XX.YY` -- a release branch for version XX.YY where XX is a
  major release and YY is an update number. Mostly used by other
  projects as "frozen versions", support is limited (by request).

## License

ADE Framework is distributed under terms of Apache License v2.0 (see
`LICENSE`).
