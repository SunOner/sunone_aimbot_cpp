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

IW_DECL(IppStatus) llwiFilterScharr(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                                      IppiSize size, int channels, IwiDerivativeType opType, IppiMaskSize kernelSize, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterScharr
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterScharr(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiDerivativeType opType,
                                   IppiMaskSize kernelSize, const IwiFilterScharrParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
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

    if(pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);
        if(!size.width || !size.height)
            return ippStsNoOperation;

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

            return llwiFilterScharr(pSrc, (int)pSrcImage->m_step, pSrcImage->m_dataType, pDst, (int)pDstImage->m_step, pDstImage->m_dataType,
                _size, pSrcImage->m_channels, opType, kernelSize, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterScharr(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                                      IppiSize size, int channels, IwiDerivativeType opType, IppiMaskSize kernelSize, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;

    Ipp8u *pTmpBuffer    = 0;
    int    tmpBufferSize = 0;

    for(;;)
    {
        switch(opType)
        {
        case iwiDerivHorFirst: status = ippiFilterScharrHorizMaskBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize); break;
        case iwiDerivVerFirst: status = ippiFilterScharrVertMaskBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize);  break;
        default:              status = ippStsNotSupportedModeErr; break;
        }
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        switch(opType)
        {
        case iwiDerivHorFirst:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterScharrHorizMaskBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp16s && dstType == ipp16s)
                status = ippiFilterScharrHorizMaskBorder_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(16s), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterScharrHorizMaskBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        case iwiDerivVerFirst:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterScharrVertMaskBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp16s && dstType == ipp16s)
                status = ippiFilterScharrVertMaskBorder_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(16s), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterScharrVertMaskBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        default:
            status = ippStsNotSupportedModeErr; break;
        }
        if(status < 0)
            break;

        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
