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

IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopyChannel
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiCopyChannel(const IwiImage *pSrcImage, int srcChannel, IwiImage *pDstImage, int dstChannel, const IwiCopyChannelParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst && srcChannel == dstChannel)
        return ippStsNoOperation;

    if(srcChannel >= pSrcImage->m_channels || srcChannel < 0 || dstChannel >= pDstImage->m_channels || dstChannel < 0)
        return ippStsBadArgErr;

    if(pSrcImage->m_channels == 1 && pDstImage->m_channels == 1)
        return iwiCopy(pSrcImage, pDstImage, NULL, NULL, pTile);

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize)
        return ippStsBadArgErr;

    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

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

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiCopyChannel(pSrc, (int)pSrcImage->m_step, pSrcImage->m_channels, srcChannel, pDst, (int)pDstImage->m_step, pDstImage->m_channels, dstChannel, _size, pSrcImage->m_typeSize);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize)
{
    switch(typeSize)
    {
    case 1:
        switch(srcChannels)
        {
        case 1:
            switch(dstChannels)
            {
            case 3:  return ippiCopy_8u_C1C3R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_8u_C1C4R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 3:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_8u_C3C1R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            case 3:  return ippiCopy_8u_C3CR (((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_8u_C4C1R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_8u_C4CR (((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsNumChannelsErr;
        }
    case 2:
        switch(srcChannels)
        {
        case 1:
            switch(dstChannels)
            {
            case 3:  return ippiCopy_16u_C1C3R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_16u_C1C4R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 3:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_16u_C3C1R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            case 3:  return ippiCopy_16u_C3CR (((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_16u_C4C1R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_16u_C4CR (((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsNumChannelsErr;
        }
    case 4:
        switch(srcChannels)
        {
        case 1:
            switch(dstChannels)
            {
            case 3:  return ippiCopy_32f_C1C3R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_32f_C1C4R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 3:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_32f_C3C1R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            case 3:  return ippiCopy_32f_C3CR (((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_32f_C4C1R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_32f_C4CR (((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsNumChannelsErr;
        }
    default: return ippStsDataTypeErr;
    }
}
