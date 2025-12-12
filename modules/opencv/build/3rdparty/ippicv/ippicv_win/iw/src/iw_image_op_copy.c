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

#include "iw/iw_image_op.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiCopy(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
                              IwiSize size, int typeSize, int channels);
IW_DECL(IppStatus) llwiCopyMask(const void *pSrc, int srcStep, void *pDst, int dstStep,
                                  IppiSize size, int typeSize, int channels, const Ipp8u *pMask, int maskStep);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopy
///////////////////////////////////////////////////////////////////////////// */
static IppStatus iwiCopy_NoMask(const IwiImage *pSrcImage, IwiImage *pDstImage, const IwiTile *pTile)
{
    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);
        if(!size.width || !size.height)
            return ippStsNoOperation;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pSrc = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                pDst = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

                size = owniGetMinSizeFromRect(&srcLim, &dstLim);
            }
            else
                return ippStsContextMatchErr;
        }

        return llwiCopy(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step, size, pSrcImage->m_typeSize, pSrcImage->m_channels);
    }
}

IW_DECL(IppStatus) iwiCopy(const IwiImage *pSrcImage, IwiImage *pDstImage, const IwiImage *pMaskImage, const IwiCopyParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsNoOperation;

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(!pMaskImage || !pMaskImage->m_ptrConst)
        return iwiCopy_NoMask(pSrcImage, pDstImage, pTile);

    status = owniCheckImageRead(pMaskImage);
    if(status)
        return status;

    if(pMaskImage->m_dataType != ipp8u ||
        pMaskImage->m_channels != 1)
        return ippStsBadArgErr;

    {
        const void *pSrc  = pSrcImage->m_ptrConst;
        const void *pMask = pMaskImage->m_ptrConst;
        void       *pDst  = pDstImage->m_ptr;
        IwiSize     size  = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);
                    size  = owniGetMinSize(&size, &pMaskImage->m_size);

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pSrc  = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                pMask = iwiImage_GetPtrConst(pMaskImage, dstRoi.y, dstRoi.x, 0);
                pDst  = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc  = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                pMask = iwiImage_GetPtrConst(pMaskImage, dstLim.y, dstLim.x, 0);
                pDst  = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

                size = owniGetMinSizeFromRect(&srcLim, &dstLim);
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pMaskImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiCopyMask(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step, _size, pSrcImage->m_typeSize, pSrcImage->m_channels, (const Ipp8u*)pMask, (int)pMaskImage->m_step);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCopy(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
                              IwiSize size, int typeSize, int channels)
{
    if(pSrc == pDst)
        return ippStsNoOperation;

    size.width = size.width*channels*typeSize;
    return ippiCopy_8u_C1R_L((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
}

IW_DECL(IppStatus) llwiCopyMask(const void *pSrc, int srcStep, void *pDst, int dstStep,
                                  IppiSize size, int typeSize, int channels, const Ipp8u *pMask, int maskStep)
{
    switch(typeSize)
    {
#if IW_ENABLE_DATA_DEPTH_8
    case 1:
        switch(channels)
        {
#if IW_ENABLE_CHANNELS_C1
        case 1:  return ippiCopy_8u_C1MR((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
#endif
#if IW_ENABLE_CHANNELS_C3
        case 3:  return ippiCopy_8u_C3MR((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
#endif
#if IW_ENABLE_CHANNELS_C4
        case 4:  return ippiCopy_8u_C4MR((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
#endif
        default: return ippStsNumChannelsErr;
        }
#endif
#if IW_ENABLE_DATA_DEPTH_16
    case 2:
        switch(channels)
        {
#if IW_ENABLE_CHANNELS_C1
        case 1:  return ippiCopy_16u_C1MR((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
#endif
#if IW_ENABLE_CHANNELS_C3
        case 3:  return ippiCopy_16u_C3MR((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
#endif
#if IW_ENABLE_CHANNELS_C4
        case 4:  return ippiCopy_16u_C4MR((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
#endif
        default: return ippStsNumChannelsErr;
        }
#endif
#if IW_ENABLE_DATA_DEPTH_32
    case 4:
        switch(channels)
        {
#if IW_ENABLE_CHANNELS_C1
        case 1:  return ippiCopy_32f_C1MR((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pMask, maskStep);
#endif
#if IW_ENABLE_CHANNELS_C3
        case 3:  return ippiCopy_32f_C3MR((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pMask, maskStep);
#endif
#if IW_ENABLE_CHANNELS_C4
        case 4:  return ippiCopy_32f_C4MR((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pMask, maskStep);
#endif
        default: return ippStsNumChannelsErr;
        }
#endif
#if IW_ENABLE_DATA_DEPTH_64
    case 8:
        switch(channels)
        {
#if IW_ENABLE_CHANNELS_C1
        case 1:  return ippiCopy_16u_C4MR((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
#endif
        default: return ippStsNumChannelsErr;
        }
#endif
    default: return ippStsDataTypeErr;
    }
}
