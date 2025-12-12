/*******************************************************************************
* Copyright 2016 Intel Corporation.
*
*
* This software and the related documents are Intel copyrighted materials, and your use of them is governed by
* the express license under which they were provided to you ('License'). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related
* documents without Intel's prior written permission.
* This software and the related documents are provided as is, with no express or implied warranties, other than
* those that are expressly stated in the License.
*******************************************************************************/

#if !defined( __IPP_IW_OWN__ )
#define __IPP_IW_OWN__

#include <string.h>

#include "iw_config.h"
#include "iw/iw_core.h"
#include "iw/iw_image.h"

#ifdef ICV_BASE
#include "ippicv.h"
#else
#include "ipp.h"
#if IW_ENABLE_THREADING_LAYER
#include "ippcore_tl.h"
#include "ippi_tl.h"
#endif
#endif

#ifdef _MSC_VER
#pragma warning (disable:4505) /* Unreferenced local function has been removed */
#endif

#ifndef IW_BUILD
#error this is a private header
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Base IW internal definitions
///////////////////////////////////////////////////////////////////////////// */
void* IPP_STDCALL ownAlignedMalloc(size_t iSize, size_t iAlign);
void  IPP_STDCALL ownAlignedFree(void* pBuffer);

#define OWN_IDX_INVALID   0xFFFFFFFF

#define OWN_IS_EXCEED(LEN, TYPE) ((LEN) > (TYPE)(LEN))
#define OWN_IS_EXCEED_INT(LEN) OWN_IS_EXCEED((LEN), Ipp32s)

#define OWN_MEM_ALLOC(SIZE) ippMalloc_L(SIZE)
#define OWN_MEM_RESET(SPEC) ippsZero_8u((Ipp8u*)(SPEC), sizeof(*(SPEC)))
#define OWN_MEM_FREE(SPEC)  ippFree(SPEC)

// Static space safe memory operations
#define OWN_SAFE_MALLOC(SIZE)         ownAlignedMalloc((size_t)(SIZE), 64);
#define OWN_SAFE_RESET(PTR, SIZE)     memset((void*)(PTR), 0, (size_t)(SIZE));
#define OWN_SAFE_FREE(PTR)            ownAlignedFree((void*)(PTR));
#define OWN_SAFE_COPY(SRC, DST, SIZE) memcpy((void*)(DST), (void*)(SRC), (size_t)(SIZE));

#define OWN_STATUS_OK    0
#define OWN_STATUS_FAIL -1

#define OWN_INIT_MAGIC_NUM 0x8117e881

// Additional build definitions for data type groups
#if defined IW_ENABLE_DATA_TYPE_8U || defined IW_ENABLE_DATA_TYPE_8S
#define IW_ENABLE_DATA_DEPTH_8  1
#endif
#if defined IW_ENABLE_DATA_TYPE_16U || defined IW_ENABLE_DATA_TYPE_16S
#define IW_ENABLE_DATA_DEPTH_16 1
#endif
#if defined IW_ENABLE_DATA_TYPE_32U || defined IW_ENABLE_DATA_TYPE_32S || defined IW_ENABLE_DATA_TYPE_32F
#define IW_ENABLE_DATA_DEPTH_32 1
#endif
#if defined IW_ENABLE_DATA_TYPE_64U || defined IW_ENABLE_DATA_TYPE_64S || defined IW_ENABLE_DATA_TYPE_64F
#define IW_ENABLE_DATA_DEPTH_64 1
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Utility functions
///////////////////////////////////////////////////////////////////////////// */
#define OWN_ROUND_FUN(SRC, DST, LEN) ippsRound_64f(SRC, DST, (int)LEN)

#define OWN_ARRAY_SATURATED_CAST(TYPE, NAME)                                                                    \
{                                                                                                               \
    IwSize i;                                                                                                   \
    OWN_ROUND_FUN(pVal, pBuffer, len);                                                                          \
    for(i = 0; i < len; i++)                                                                                    \
    {                                                                                                           \
        if(pVal[i] > IPP_MIN_##NAME)                                                                            \
        {                                                                                                       \
            if(pVal[i] < IPP_MAX_##NAME)                                                                        \
                ((TYPE*)pBuffer)[i] = (TYPE)(pBuffer[i]);                                                       \
            else                                                                                                \
                ((TYPE*)pBuffer)[i] = (TYPE)IPP_MAX_##NAME;                                                     \
        }                                                                                                       \
        else                                                                                                    \
            ((TYPE*)pBuffer)[i] = (TYPE)IPP_MIN_##NAME;                                                         \
    }                                                                                                           \
}
#define OWN_SATURATED_CAST(TYPE, NAME)                                                                          \
{                                                                                                               \
    OWN_ROUND_FUN(&val, &val, 1);                                                                               \
    if(val > IPP_MIN_##NAME)                                                                                    \
    {                                                                                                           \
        if(val < IPP_MAX_##NAME)                                                                                \
            val = (TYPE)(val);                                                                                  \
        else                                                                                                    \
            val = (TYPE)IPP_MAX_##NAME;                                                                         \
    }                                                                                                           \
    else                                                                                                        \
        val = (TYPE)IPP_MIN_##NAME;                                                                             \
}
static IW_INLINE Ipp8u* ownCastArray_64f8u(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    OWN_ARRAY_SATURATED_CAST(Ipp8u, 8U)
    return (Ipp8u*)pBuffer;
}
static IW_INLINE Ipp8s* ownCastArray_64f8s(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    OWN_ARRAY_SATURATED_CAST(Ipp8s, 8S)
    return (Ipp8s*)pBuffer;
}
static IW_INLINE Ipp16u* ownCastArray_64f16u(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    OWN_ARRAY_SATURATED_CAST(Ipp16u, 16U)
    return (Ipp16u*)pBuffer;
}
static IW_INLINE Ipp16s* ownCastArray_64f16s(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    OWN_ARRAY_SATURATED_CAST(Ipp16s, 16S)
    return (Ipp16s*)pBuffer;
}
static IW_INLINE Ipp32u* ownCastArray_64f32u(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    OWN_ARRAY_SATURATED_CAST(Ipp32u, 32U)
    return (Ipp32u*)pBuffer;
}
static IW_INLINE Ipp32s* ownCastArray_64f32s(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    OWN_ARRAY_SATURATED_CAST(Ipp32s, 32S)
    return (Ipp32s*)pBuffer;
}
static IW_INLINE Ipp32f* ownCastArray_64f32f(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    ippsConvert_64f32f(pVal, (Ipp32f*)pBuffer, (int)len);
    return (Ipp32f*)pBuffer;
}
static IW_INLINE Ipp64f* ownCastArray_64f64f(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len)
{
    ippsCopy_64f(pVal, pBuffer, (int)len);
    return pBuffer;
}
static void ownCastArray_64f(const Ipp64f *pVal, Ipp64f *pBuffer, IppDataType dataType, IwSize len)
{
    switch(dataType)
    {
    case ipp8u:  ownCastArray_64f8u (pVal, pBuffer, len);    return;
    case ipp8s:  ownCastArray_64f8s (pVal, pBuffer, len);    return;
    case ipp16u: ownCastArray_64f16u(pVal, pBuffer, len);    return;
    case ipp16s: ownCastArray_64f16s(pVal, pBuffer, len);    return;
    case ipp32u: ownCastArray_64f32u(pVal, pBuffer, len);    return;
    case ipp32s: ownCastArray_64f32s(pVal, pBuffer, len);    return;
    case ipp32f: ownCastArray_64f32f(pVal, pBuffer, len);    return;
    case ipp64f: ownCastArray_64f64f(pVal, pBuffer, len);    return;
    default: return;
    }
}

static IW_INLINE Ipp8u ownCast_64f8u(Ipp64f val)
{
    OWN_SATURATED_CAST(Ipp8u, 8U)
    return (Ipp8u)val;
}
static IW_INLINE Ipp8s ownCast_64f8s(Ipp64f val)
{
    OWN_SATURATED_CAST(Ipp8s, 8S)
    return (Ipp8s)val;
}
static IW_INLINE Ipp16u ownCast_64f16u(Ipp64f val)
{
    OWN_SATURATED_CAST(Ipp16u, 16U)
    return (Ipp16u)val;
}
static IW_INLINE Ipp16s ownCast_64f16s(Ipp64f val)
{
    OWN_SATURATED_CAST(Ipp16s, 16S)
    return (Ipp16s)val;
}
static IW_INLINE Ipp32u ownCast_64f32u(Ipp64f val)
{
    OWN_SATURATED_CAST(Ipp32u, 32U)
    return (Ipp32u)val;
}
static IW_INLINE Ipp32s ownCast_64f32s(Ipp64f val)
{
    OWN_SATURATED_CAST(Ipp32s, 32S)
    return (Ipp32s)val;
}
static IW_INLINE Ipp32f ownCast_64f32f(Ipp64f val)
{
    if(val > IPP_MAXABS_32F)       val = IPP_MAXABS_32F;
    else if(val < -IPP_MAXABS_32F) val = -IPP_MAXABS_32F;
    return (Ipp32f)val;
}

typedef void* (*OwnCastArray_ptr)(const Ipp64f *pVal, Ipp64f *pBuffer, IwSize len);

/* /////////////////////////////////////////////////////////////////////////////
//                   Long types compatibility checkers
///////////////////////////////////////////////////////////////////////////// */
static IW_INLINE IppStatus ownLongCompatCheckValue(IwSize val, int *pVal)
{
#if defined (_M_AMD64) || defined (__x86_64__)
    if(OWN_IS_EXCEED_INT(val))
        return ippStsSizeErr;
    else
#endif
    if(pVal)
        *pVal = (int)val;
    return ippStsNoErr;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnVector - C Vector
///////////////////////////////////////////////////////////////////////////// */

// Simple C vector interface
typedef struct _OwnVector
{
    Ipp8u  *m_pBuffer;
    size_t  m_bufferLen;
    size_t  m_elemSize;
    size_t  m_size;
} OwnVector;

IW_DECL(void) ownVector_Reserve(OwnVector *pVector, size_t reserveSize);
IW_DECL(void) ownVector_Init(OwnVector *pVector, size_t elemSize, size_t reserve);
IW_DECL(void) ownVector_Release(OwnVector *pVector);
IW_DECL(void) ownVector_Resize(OwnVector *pVector, size_t newSize);
IW_DECL(void) ownVector_PushBack(OwnVector *pVector, void *pData);
IW_DECL(void) ownVector_PopBack(OwnVector *pVector, void *pData);

/* /////////////////////////////////////////////////////////////////////////////
//                   Shared memory interface for temporary buffers
///////////////////////////////////////////////////////////////////////////// */
void* IPP_STDCALL ownSharedMalloc(IwSize size);
void  IPP_STDCALL ownSharedFree(void* pBuffer);

/* /////////////////////////////////////////////////////////////////////////////
//                   OWN ROI manipulation
///////////////////////////////////////////////////////////////////////////// */
typedef enum _OwnTileInitType
{
    ownTileInitNone   = 0,
    ownTileInitSimple = 0xA1A2A3,
    ownTileInitPipe   = 0xB1B2B3
} OwnRoiInitType;

#ifdef __cplusplus
}
#endif

#endif
