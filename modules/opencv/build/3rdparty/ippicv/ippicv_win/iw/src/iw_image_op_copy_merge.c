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

IW_DECL(IppStatus) llwiCopyMerge(const void* const pSrc[], int srcStep, void *pDst, int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial);

IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMergeChannels
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiMergeChannels(const IwiImage* const pSrcImage[], IwiImage *pDstImage, const IwiMergeChannelsParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(!pSrcImage)
        return ippStsNullPtrErr;

    if(pDstImage->m_channels == 1)
        return iwiCopy(pSrcImage[0], pDstImage, NULL, NULL, pTile);

    if(pDstImage->m_channels > 4)
        return ippStsNumChannelsErr;

    {
        const void* pSrc[4]       = {NULL};
        IwSize      srcStep[4]    = {0};
        int         srcPixSize[4] = {0};
        void*       pDst          = pDstImage->m_ptr;
        IwiSize     size          = pDstImage->m_size;
        int         channels      = pDstImage->m_channels;
        int         i;

        for(i = 0; i < pDstImage->m_channels; i++)
        {
            if(pSrcImage[i] && pSrcImage[i]->m_ptrConst)
            {
                if(pSrcImage[i]->m_ptrConst == pDstImage->m_ptrConst)
                    return ippStsInplaceModeNotSupportedErr;

                if(pSrcImage[i]->m_typeSize != pDstImage->m_typeSize)
                    return ippStsBadArgErr;

                size          = owniGetMinSize(&pSrcImage[i]->m_size, &size);
                pSrc[i]       = pSrcImage[i]->m_ptrConst;
                srcStep[i]    = pSrcImage[i]->m_step;
                srcPixSize[i] = pSrcImage[i]->m_typeSize*pSrcImage[i]->m_channels;
                if(srcStep[i] != srcStep[0])
                    return ippStsStepErr;
                if(srcPixSize[i] != srcPixSize[0])
                    return ippStsBadArgErr;
            }
            else
                channels--;
        }
        if(!size.width || !size.height)
            return ippStsNoOperation;
        if(!channels)
            return ippStsNoOperation;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                owniShiftPtrArrConst(pSrc, pSrc, srcStep, srcPixSize, pDstImage->m_channels, dstRoi.x, dstRoi.y);
                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                owniShiftPtrArrConst(pSrc, pSrc, srcStep, srcPixSize, pDstImage->m_channels, srcLim.x, srcLim.y);
                pDst = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

                size = owniGetMinSizeFromRect(&srcLim, &dstLim);
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pSrcImage[0]->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiCopyMerge((const void* const*)pSrc, (int)srcStep[0], pDst, (int)pDstImage->m_step, _size, pDstImage->m_typeSize, pDstImage->m_channels, (channels != pDstImage->m_channels)?1:0);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCopyMerge(const void* const pSrc[], int srcStep, void *pDst, int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial)
{
    if(partial)
    {
        IppStatus status = ippStsNoErr;
        int       i = 0;
        for(i = 0; i < channels; i++)
        {
            if(pSrc[i])
            {
                status = llwiCopyChannel(pSrc[i], srcStep, 1, 0, pDst, dstStep, channels, i, size, typeSize);
                if(status < 0)
                    return status;
            }
        }
        return status;
    }
    else
    {
        switch(typeSize)
        {
        case 1:
            switch(channels)
            {
            case 3:  return ippiCopy_8u_P3C3R((const Ipp8u**)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
            case 4:  return ippiCopy_8u_P4C4R((const Ipp8u**)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 2:
            switch(channels)
            {
            case 3:  return ippiCopy_16u_P3C3R((const Ipp16u**)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size);
            case 4:  return ippiCopy_16u_P4C4R((const Ipp16u**)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(channels)
            {
            case 3:  return ippiCopy_32f_P3C3R((const Ipp32f**)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size);
            case 4:  return ippiCopy_32f_P4C4R((const Ipp32f**)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsDataTypeErr;
        }
    }
}
