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

#if !defined( __IPP_IW_OWNI__ )
#define __IPP_IW_OWNI__

#include "iw_own.h"

#ifndef IW_BUILD
#error this is a private header
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Base IW Image internal definitions
///////////////////////////////////////////////////////////////////////////// */
#define OWN_ALIGN_ROW 64

#define OWN_ROI_FIT(SIZE, ROI)\
{\
    if((ROI).x < 0)\
        (ROI).x = 0;\
    if((ROI).y < 0)\
        (ROI).y = 0;\
    if((ROI).x > (SIZE).width)\
        (ROI).x = (SIZE).width;\
    if((ROI).y > (SIZE).height)\
        (ROI).y = (SIZE).height;\
    if(!(ROI).width || ((ROI).width + (ROI).x) > (SIZE).width)\
        (ROI).width = (SIZE).width - (ROI).x;\
    if(!(ROI).height || ((ROI).height + (ROI).y) > (SIZE).height)\
        (ROI).height = (SIZE).height - (ROI).y;\
}

#define OWN_GET_PURE_BORDER(BORDER) ((IwiBorderType)((BORDER)&0xF))
#define OWN_GET_BORDER_VAL(TYPE)  ((OWN_GET_PURE_BORDER(border) == ippBorderConst && pBorderVal)?(ownCast_64f##TYPE(pBorderVal[0])):0)
#define OWN_GET_BORDER_VALP(TYPE, CH) ((OWN_GET_PURE_BORDER(border) == ippBorderConst && pBorderVal)?(ownCastArray_64f##TYPE(pBorderVal, borderVal, (CH))):0)
#define OWN_GET_BORDER_VAL2(PRECOMP, TYPE)  ((OWN_GET_PURE_BORDER(border) == ippBorderConst && pBorderVal)?((PRECOMP)?*((Ipp##TYPE*)pBorderVal):(ownCast_64f##TYPE(pBorderVal[0]))):0)
#define OWN_GET_BORDER_VALP2(PRECOMP, TYPE, CH) ((OWN_GET_PURE_BORDER(border) == ippBorderConst && pBorderVal)?((PRECOMP)?((Ipp##TYPE*)pBorderVal):(ownCastArray_64f##TYPE(pBorderVal, borderVal, (CH)))):0)
#define OWN_GET_BORDER_VAL3(BORDER, ARR) ((OWN_GET_PURE_BORDER(BORDER) == ippBorderConst && (ARR))?(ARR):0)

typedef enum _OwniChCodes
{
    owniC_Invalid = 0,
    owniC1,
    owniC1C3,
    owniC1C4,
    owniC3,
    owniC3C1,
    owniC3C4,
    owniC4,
    owniC4C1,
    owniC4C3,
    owniC4M1110,
    owniC4M1000,
    owniC4M1001,
    owniC4M1RR0,
    owniC4M1RR1

} OwniChCodes;

#define OWN_DESC_GET_CH(DESC) (((DESC)>>12)&0xF)
#define OWN_DESC_GET_REPL_CH(DESC) (((DESC)>>24)&0xF)
#define OWN_DESC_CHECK_MASK(DESC, CH) (((DESC)>>(CH))&0x1)
#define OWN_DESC_CHECK_REPL(DESC, CH) (((DESC)>>((CH)+16))&0x1)

/* /////////////////////////////////////////////////////////////////////////////
//                   Utility functions
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(OwniChCodes) owniChDescriptorToCode(IwiChDescriptor chDesc, int srcChannels, int dstChannels);

static IW_INLINE const void* owniShiftPtrConst(const void *pPtr, IwSize step, int typeSize, int channels, IwSize x, IwSize y)
{
    return (((Ipp8u*)pPtr) + step*y + typeSize*channels*x);
}

static IW_INLINE void owniShiftPtrArrConst(const void* const pSrcPtr[], const void* pDstPtr[], IwSize step[], int pixelSize[], int planes, IwSize x, IwSize y)
{
    int i;
    for(i = 0; i < planes; i++)
    {
        if(pSrcPtr[i])
            pDstPtr[i] = owniShiftPtrConst(pSrcPtr[i], step[i], pixelSize[i], 1, x, y);
    }
}

static IW_INLINE void* owniShiftPtr(const void *pPtr, IwSize step, int typeSize, int channels, IwSize x, IwSize y)
{
    return (((Ipp8u*)pPtr) + step*y + typeSize*channels*x);
}

static IW_INLINE void owniShiftPtrArr(void* const pSrcPtr[], void* pDstPtr[], IwSize step[], int pixelSize[], int planes, IwSize x, IwSize y)
{
    int i;
    for(i = 0; i < planes; i++)
    {
        if(pSrcPtr[i])
            pDstPtr[i] = owniShiftPtr(pSrcPtr[i], step[i], pixelSize[i], 1, x, y);
    }
}

static IW_INLINE IwSize owniAlignStep(IwSize step, int align)
{
    return (step + (align - 1)) & -align;
}

static IW_INLINE IwiSize owniGetMinSize(const IwiSize *pFirst, const IwiSize *pSecond)
{
    IwiSize size;
    size.width  = IPP_MIN(pFirst->width, pSecond->width);
    size.height = IPP_MIN(pFirst->height, pSecond->height);
    return size;
}
static IW_INLINE IwiSize owniGetMinSizeFromRect(const IwiRoi *pFirst, const IwiRoi *pSecond)
{
    IwiSize size;
    size.width  = IPP_MIN(pFirst->width, pSecond->width);
    size.height = IPP_MIN(pFirst->height, pSecond->height);
    return size;
}

static IW_INLINE IppStatus owniCheckImageRead(const IwiImage *pImage)
{
    if(!pImage)
        return ippStsNullPtrErr;
    if(!pImage->m_size.width || !pImage->m_size.height)
        return ippStsNoOperation;
    if(!pImage->m_ptrConst)
        return ippStsNullPtrErr;
    return ippStsNoErr;
}

static IW_INLINE IppStatus owniCheckImageWrite(const IwiImage *pImage)
{
    if(!pImage)
        return ippStsNullPtrErr;
    if(!pImage->m_size.width || !pImage->m_size.height)
        return ippStsNoOperation;
    if(!pImage->m_ptr)
        return ippStsNullPtrErr;
    return ippStsNoErr;
}

static IW_INLINE int owniCheckBorderValidity(IwiBorderType border)
{
    // Create bit-mask for all valid values
#if IPP_VERSION_COMPLEX >= 20170002
    static const int validMask = ippBorderInMem|ippBorderFirstStageInMem|ippBorderRepl|ippBorderWrap|ippBorderMirror|ippBorderMirrorR|ippBorderDefault|ippBorderConst|ippBorderTransp;
#else
    static const int validMask = ippBorderInMem|ippBorderRepl|ippBorderWrap|ippBorderMirror|ippBorderMirrorR|ippBorderDefault|ippBorderConst|ippBorderTransp;
#endif

    // Check if fully in memory
    if(!((border&ippBorderInMem) == ippBorderInMem))
    {
        // If border is only partially in memory it must have extrapolation type for non-in-memory parts
        if(!OWN_GET_PURE_BORDER(border))
            return 0;
    }

    // Check for invalid bits
    if(border&(~validMask))
        return 0;
    else
        return 1;
}

static IW_INLINE int owniSuggestThreadsNum(int maxThreads, const IwiImage *pImage, double multiplier)
{
    if(pImage->m_size.height > maxThreads)
    {
        size_t opMemory = (int)(pImage->m_step*pImage->m_size.height*multiplier);
        int   l2cache  = 0;
        if(ippGetL2CacheSize(&l2cache) < 0 || !l2cache)
            l2cache = 100000;

        return IPP_MAX(1, (IPP_MIN((int)(opMemory/(l2cache*0.6)), maxThreads)));
    }
    return 1;
}

static IW_INLINE int owniBorderSizeIsNegative(const IwiBorderSize *pBorderSize)
{
    if(pBorderSize->left < 0 || pBorderSize->top < 0 || pBorderSize->right < 0 || pBorderSize->bottom < 0)
        return 1;
    return 0;
}

static IW_INLINE void owniBorderSizeSaturate(IwiBorderSize *pBorderSize)
{
    if(pBorderSize->left < 0)
        pBorderSize->left = 0;
    if(pBorderSize->top < 0)
        pBorderSize->top = 0;
    if(pBorderSize->right < 0)
        pBorderSize->right = 0;
    if(pBorderSize->bottom < 0)
        pBorderSize->bottom = 0;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   Long types compatibility checkers
///////////////////////////////////////////////////////////////////////////// */
static IW_INLINE IppStatus owniLongCompatCheckPoint(IppiPointL pointL, IppiPoint *pPoint)
{
#if defined (_M_AMD64) || defined (__x86_64__)
    if(OWN_IS_EXCEED_INT(pointL.x) || OWN_IS_EXCEED_INT(pointL.y))
        return ippStsSizeErr;
    else
#endif
    if(pPoint)
    {
        pPoint->x = (int)pointL.x;
        pPoint->y = (int)pointL.y;
    }
    return ippStsNoErr;
}

static IW_INLINE IppStatus owniLongCompatCheckSize(IwiSize sizeL, IppiSize *pSize)
{
#if defined (_M_AMD64) || defined (__x86_64__)
    if(OWN_IS_EXCEED_INT(sizeL.width) || OWN_IS_EXCEED_INT(sizeL.height))
        return ippStsSizeErr;
    else
#endif
    if(pSize)
    {
        pSize->width  = (int)sizeL.width;
        pSize->height = (int)sizeL.height;
    }
    return ippStsNoErr;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   OWN ROI manipulation
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IwiSize) owniSuggestTileSize_k2(const IwiImage *pImage, IwiSize kernelSize, double multiplier);
static IW_INLINE IwiSize owniSuggestTileSize_k1(const IwiImage *pImage, IwSize kernelLen, double multiplier)
{
    IwiSize kernelSize;
    kernelSize.width  = kernelLen;
    kernelSize.height = kernelLen;
    return owniSuggestTileSize_k2(pImage, kernelSize, multiplier);
}
static IW_INLINE IwiSize owniSuggestTileSize_k0(const IwiImage *pImage, double multiplier)
{
    IwiSize kernelSize = {1, 1};
    return owniSuggestTileSize_k2(pImage, kernelSize, multiplier);
}

IW_DECL(int)       owniTile_BoundToSize(IwiRoi *pRoi, IwiSize *pMinSize);
IW_DECL(int)       owniTile_CorrectBordersOverlap(IwiRoi *pRoi, IwiSize *pMinSize, const IwiBorderType *pBorder, const IwiBorderSize *pBorderSize, const IwiBorderSize *pBorderSizeAcc, const IwiSize *pSrcImageSize);
IW_DECL(void)      owniTile_GetTileBorder(IwiBorderType *pBorder, const IwiRoi *pRoi, const IwiBorderSize *pBorderSize, const IwiSize *pSrcImageSize);
IW_DECL(IppStatus) owniTilePipeline_ProcBorder(const IwiTile *pTile, IwiImage *pSrcImage, IwiBorderType *pBorder, const Ipp64f *pBorderVal);

#ifdef __cplusplus
}
#endif

#endif
