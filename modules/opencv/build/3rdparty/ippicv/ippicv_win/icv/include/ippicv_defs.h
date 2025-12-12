/*
// Copyright 1999 Intel Corporation All Rights Reserved.
//
//
// This software and the related documents are Intel copyrighted materials, and your use of them is governed by
// the express license under which they were provided to you ('License'). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related
// documents without Intel's prior written permission.
// This software and the related documents are provided as is, with no express or implied warranties, other than
// those that are expressly stated in the License.
//
*/

/*
//              Intel(R) Integrated Performance Primitives (Intel(R) IPP)
//              Common Types and Macro Definitions
//
//
*/


#ifndef __IPPICV_DEFS_H__
#define __IPPICV_DEFS_H__

#ifdef __cplusplus
extern "C" {
#endif

#if defined( IPPDEFS_H__ )
#error "You try to include ippicv_defs.h and ippdefs.h together. You can't use Intel(R) IPP and ICV libraries together, please use only one and please use header files only for one library"
#endif


#if defined( _IPP_PARALLEL_STATIC ) || defined( _IPP_PARALLEL_DYNAMIC )
  #pragma message("Threaded versions of Intel(R) IPP libraries are deprecated and will be removed in one of the future Intel(R) IPP releases. Use the following link for details: https://www.intel.com/content/www/us/en/developer/articles/release-notes/release-notes-for-oneapi-integrated-performance-primitives.html")
#endif

#if defined (_WIN64)
#define IPP_LIB_SUBDIR_ARCH_SUFFIX ""
#elif defined (_WIN32)
#define IPP_LIB_SUBDIR_ARCH_SUFFIX "32"
#endif

#if !defined( IPPAPI )

  #if defined( IPP_W32DLL ) && (defined( _WIN32 ) || defined( _WIN64 ))
    #if defined( _MSC_VER ) || defined( __ICL )
      #define IPPAPI( type,name,arg ) \
                     __declspec(dllimport)   type IPP_STDCALL name arg;
    #else
      #define IPPAPI( type,name,arg )        type IPP_STDCALL name arg;
    #endif
  #else
    #define   IPPAPI( type,name,arg )        type IPP_STDCALL name arg;
  #endif

#endif

#if (defined( __ICL ) || defined( __ECL ) || defined(_MSC_VER)) && !defined( _PCS ) && !defined( _PCS_GENSTUBS )
  #if( __INTEL_COMPILER >= 1100 ) /* icl 11.0 supports additional comment */
    #if( _MSC_VER >= 1400 )
      #define IPP_DEPRECATED( comment ) __declspec( deprecated ( comment ))
    #else
      #pragma message ("your icl version supports additional comment for deprecated functions but it can't be displayed")
      #pragma message ("because internal _MSC_VER macro variable setting requires compatibility with MSVC7.1")
      #pragma message ("use -Qvc8 switch for icl command line to see these additional comments")
      #define IPP_DEPRECATED( comment ) __declspec( deprecated )
    #endif
  #elif( _MSC_FULL_VER >= 140050727 )&&( !defined( __INTEL_COMPILER )) /* VS2005 supports additional comment */
    #define IPP_DEPRECATED( comment ) __declspec( deprecated ( comment ))
  #elif( _MSC_VER <= 1200 )&&( !defined( __INTEL_COMPILER )) /* VS 6 doesn't support deprecation */
    #define IPP_DEPRECATED( comment )
  #else
    #define IPP_DEPRECATED( comment ) __declspec( deprecated )
  #endif
#elif (defined(__ICC) || defined(__ECC) || defined( __GNUC__ )) && !defined( _PCS ) && !defined( _PCS_GENSTUBS )
  #if defined( __GNUC__ )
    #if __GNUC__ >= 4 && __GNUC_MINOR__ >= 5
      #define IPP_DEPRECATED( message ) __attribute__(( deprecated( message )))
    #else
      #define IPP_DEPRECATED( message ) __attribute__(( deprecated ))
    #endif
  #else
    #define IPP_DEPRECATED( comment ) __attribute__(( deprecated ))
  #endif
#else
  #define IPP_DEPRECATED( comment )
#endif

#include "ippicv_base.h"
#include "ippicv_types.h"

#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_DEFS_H__ */
