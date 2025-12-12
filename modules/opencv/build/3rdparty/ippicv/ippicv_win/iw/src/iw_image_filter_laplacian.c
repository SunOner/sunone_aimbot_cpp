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

#include "iw/iw_image_filter.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiFilterLaplacian(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                                         IppiSize size, int channels, IppiMaskSize kernelSize, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterLaplacian
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterLaplacian(const IwiImage *pSrcImage, IwiImage *pDstImage, IppiMaskSize kernelSize, const IwiFilterLaplacianParams *pAuxParams,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
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
        return ippStsInplaceModeNotSupportedErr;

    if(pSrcImage->m_channels > 1)
        return ippStsNumChannelsErr;

    if(pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
                return ippStsNotSupportedModeErr;

            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi         dstRoi     = pTile->m_dstRoi;
                IwiBorderSize  borderSize = iwiSizeToBorderSize(iwiMaskToSize(kernelSize));

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;
                owniTile_CorrectBordersOverlap(&dstRoi, &size, &border, &borderSize, &borderSize, &pSrcImage->m_size);
                owniTile_GetTileBorder(&border, &dstRoi, &borderSize, &pSrcImage->m_size);

                pSrc = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc   = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                pDst   = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);
                iwiTilePipeline_GetTileBorder(pTile, &border);

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

            return llwiFilterLaplacian(pSrc, (int)pSrcImage->m_step, pSrcImage->m_dataType, pDst, (int)pDstImage->m_step, pDstImage->m_dataType,
                _size, pSrcImage->m_channels, kernelSize, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterLaplacian(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                                         IppiSize size, int channels, IppiMaskSize kernelSize, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;

    Ipp8u *pTmpBuffer    = 0;
    int    tmpBufferSize = 0;

    for(;;)
    {
        if(srcType == ipp8u && dstType == ipp16s)
            status = ippiFilterLaplacianGetBufferSize_8u16s_C1R(size, kernelSize, &tmpBufferSize);
        else if(srcType == ipp32f && dstType == ipp32f)
            status = ippiFilterLaplacianGetBufferSize_32f_C1R(size, kernelSize, &tmpBufferSize);
        else
            status = ippStsDataTypeErr;
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        switch(channels)
        {
        case 1:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterLaplacianBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterLaplacianBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        default: status = ippStsNumChannelsErr;
        }
        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
