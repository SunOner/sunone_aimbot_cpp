// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/util/assert.hpp"

#include <stdlib.h>
#include <stdio.h>

namespace ade
{
namespace details
{

void dev_assert_abort(const char* str, int line, const char* file,
                      const char* func)
{
    fprintf(stderr,
            "%s:%d: Assertion \"%s\" in function \"%s\" failed\n",
            file, line, str, func);
    fflush(stderr);
    abort();
}
void dev_exception_abort(const char* str)
{
    fprintf(stderr, "An exception thrown! %s\n" , str);
    fflush(stderr);
    abort();
}

} // namespace details
} // namespace ade
