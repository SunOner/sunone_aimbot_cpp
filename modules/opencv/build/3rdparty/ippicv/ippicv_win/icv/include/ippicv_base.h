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
//              Basic Types and Macro Definitions
// 
// 
*/


#ifndef __IPPICV_BASE_H__
#define __IPPICV_BASE_H__

#ifdef __cplusplus
extern "C" {
#endif

#if defined( IPPBASE_H__ )
#error "You try to include ippicv_base.h and ippbase.h together. You can't use Intel(R) IPP and ICV libraries together, please use only one and please use header files only for one library"
#endif

#if defined( _WIN32 ) || defined ( _WIN64 )
  #define IPP_STDCALL  __stdcall
  #define IPP_CDECL    __cdecl
  #define IPP_INT64    __int64
  #define IPP_UINT64    unsigned __int64
#else
  #define IPP_STDCALL
  #define IPP_CDECL
  #define IPP_INT64    long long
  #define IPP_UINT64    unsigned long long
#endif

#define IPP_PI    ( 3.14159265358979323846 )  /* ANSI C does not support M_PI */
#define IPP_2PI   ( 6.28318530717958647692 )  /* 2*pi                         */
#define IPP_PI2   ( 1.57079632679489661923 )  /* pi/2                         */
#define IPP_PI4   ( 0.78539816339744830961 )  /* pi/4                         */
#define IPP_PI180 ( 0.01745329251994329577 )  /* pi/180                       */
#define IPP_RPI   ( 0.31830988618379067154 )  /* 1/pi                         */
#define IPP_SQRT2 ( 1.41421356237309504880 )  /* sqrt(2)                      */
#define IPP_SQRT3 ( 1.73205080756887729353 )  /* sqrt(3)                      */
#define IPP_LN2   ( 0.69314718055994530942 )  /* ln(2)                        */
#define IPP_LN3   ( 1.09861228866810969139 )  /* ln(3)                        */
#define IPP_E     ( 2.71828182845904523536 )  /* e                            */
#define IPP_RE    ( 0.36787944117144232159 )  /* 1/e                          */
#define IPP_EPS23 ( 1.19209289e-07f )
#define IPP_EPS52 ( 2.2204460492503131e-016 )

#define IPP_MAX_8U     ( 0xFF )
#define IPP_MAX_16U    ( 0xFFFF )
#define IPP_MAX_32U    ( 0xFFFFFFFF )
#define IPP_MIN_8U     ( 0 )
#define IPP_MIN_16U    ( 0 )
#define IPP_MIN_32U    ( 0 )
#define IPP_MIN_8S     (-128 )
#define IPP_MAX_8S     ( 127 )
#define IPP_MIN_16S    (-32768 )
#define IPP_MAX_16S    ( 32767 )
#define IPP_MIN_32S    (-2147483647 - 1 )
#define IPP_MAX_32S    ( 2147483647 )
#define IPP_MIN_64U    ( 0 )

#if defined( _WIN32 ) || defined ( _WIN64 )
  #define IPP_MAX_64S  ( 9223372036854775807i64 )
  #define IPP_MIN_64S  (-9223372036854775807i64 - 1 )
  #define IPP_MAX_64U  ( 0xffffffffffffffffL ) /* 18446744073709551615 */
#else
  #define IPP_MAX_64S  ( 9223372036854775807LL )
  #define IPP_MIN_64S  (-9223372036854775807LL - 1 )
  #define IPP_MAX_64U  ( 0xffffffffffffffffLL ) /* 18446744073709551615 */
#endif

#define IPP_MINABS_32F ( 1.175494351e-38f )
#define IPP_MAXABS_32F ( 3.402823466e+38f )
#define IPP_EPS_32F    ( 1.192092890e-07f )
#define IPP_MINABS_64F ( 2.2250738585072014e-308 )
#define IPP_MAXABS_64F ( 1.7976931348623158e+308 )
#define IPP_EPS_64F    ( 2.2204460492503131e-016 )

#define IPP_MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define IPP_MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#define IPP_ABS( a ) ( ((a) < 0) ? (-(a)) : (a) )

typedef struct {
    int    major;                     /* e.g. 1                               */
    int    minor;                     /* e.g. 2                               */
    int    majorBuild;                /* e.g. 3                               */
    int    build;                     /* e.g. 10, always >= majorBuild        */
    char  targetCpu[4];               /* corresponding to Intel(R) processor  */
    const char* Name;                 /* e.g. "ippsw7"                        */
    const char* Version;              /* e.g. "v1.2 Beta"                     */
    const char* BuildDate;            /* e.g. "Jul 20 99"                     */
} IppLibraryVersion;

typedef unsigned char  Ipp8u;
typedef unsigned short Ipp16u;
typedef unsigned int   Ipp32u;
typedef signed char    Ipp8s;
typedef signed short   Ipp16s;
typedef signed int     Ipp32s;
typedef float          Ipp32f;
typedef IPP_INT64        Ipp64s;
typedef IPP_UINT64       Ipp64u;
typedef double         Ipp64f;
typedef Ipp16s         Ipp16f;

typedef struct {
    Ipp8s  re;
    Ipp8s  im;
} Ipp8sc;

typedef struct {
    Ipp16s  re;
    Ipp16s  im;
} Ipp16sc;

typedef struct {
    Ipp16u  re;
    Ipp16u  im;
} Ipp16uc;

typedef struct {
    Ipp32s  re;
    Ipp32s  im;
} Ipp32sc;

typedef struct {
    Ipp32f  re;
    Ipp32f  im;
} Ipp32fc;

typedef struct {
    Ipp64s  re;
    Ipp64s  im;
} Ipp64sc;

typedef struct {
    Ipp64f  re;
    Ipp64f  im;
} Ipp64fc;

typedef enum {
   ippUndef = -1,
   ipp1u    =  0,
   ipp8u    =  1,
   ipp8uc   =  2,
   ipp8s    =  3,
   ipp8sc   =  4,
   ipp16u   =  5,
   ipp16uc  =  6,
   ipp16s   =  7,
   ipp16sc  =  8,
   ipp32u   =  9,
   ipp32uc  = 10,
   ipp32s   = 11,
   ipp32sc  = 12,
   ipp32f   = 13,
   ipp32fc  = 14,
   ipp64u   = 15,
   ipp64uc  = 16,
   ipp64s   = 17,
   ipp64sc  = 18,
   ipp64f   = 19,
   ipp64fc  = 20
} IppDataType;

typedef enum { 
    ippFalse = 0, 
    ippTrue = 1 
} IppBool;

#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_BASE_H__ */

