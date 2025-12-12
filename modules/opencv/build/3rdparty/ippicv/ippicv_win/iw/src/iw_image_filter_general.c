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

IW_DECL(IppStatus) llwiFilter(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType,
    int channels, const void *pKernel, IppiSize kernelSize, IppDataType kernelType, int divisor, int offset, IppRoundMode roundMode,
    IppHintAlgorithm algoMode, IwiBorderType border, const Ipp64f *pBorderVal);


/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilter
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilter(const IwiImage *pSrcImage, IwiImage *pDstImage, const IwiImage *pKernel,
    const IwiFilterParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus       status;
    IwiFilterParams auxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageRead(pKernel);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsInplaceModeNotSupportedErr;

    if(pSrcImage->m_dataType != pDstImage->m_dataType ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if((pKernel->m_step && pKernel->m_size.width*pKernel->m_typeSize != pKernel->m_step))
        return ippStsNotSupportedModeErr;

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiFilter_SetDefaultParams(&auxParams);

    if(pKernel->m_dataType == ipp16s)
    {
        if(auxParams.divisor == 0)
        {
            IppiSize kernelSize;
            Ipp64f   sum;
            kernelSize.width  = (int)pKernel->m_size.width;
            kernelSize.height = (int)pKernel->m_size.height;
            ippiSum_16s_C1R((const Ipp16s*)pKernel->m_ptrConst, (int)pKernel->m_step, kernelSize, &sum);
            auxParams.divisor = (sum)?(int)(sum):1;
        }
    }
    else if(auxParams.divisor != 0 && auxParams.divisor != 1)
        return ippStsBadArgErr;

    {
        const void   *pSrc   = pSrcImage->m_ptrConst;
        void         *pDst   = pDstImage->m_ptr;
        IwiSize       size   = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
                return ippStsNotSupportedModeErr;

            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi         dstRoi     = pTile->m_dstRoi;
                IwiBorderSize  borderSize = iwiSizeToBorderSize(pKernel->m_size);

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
            IppiSize _kernelSize;

            status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(pKernel->m_size, &_kernelSize);
            if(status < 0)
                return status;

            return llwiFilter(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step, _size, pSrcImage->m_dataType,
                pSrcImage->m_channels, pKernel->m_ptr, _kernelSize, pKernel->m_dataType, (int)auxParams.divisor, auxParams.offset,
                auxParams.roundMode, auxParams.algoMode, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilter(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType,
    int channels, const void *pKernel, IppiSize kernelSize, IppDataType kernelType, int divisor, int offset, IppRoundMode roundMode,
    IppHintAlgorithm algoMode, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBorderSpec *pSpec    = 0;
    int                   specSize = 0;

    Ipp8u   *pTmpBuffer    = 0;
    int      tmpBufferSize = 0;

    for(;;)
    {
        // Initialize Intel IPP functions and check parameters
        status = ippiFilterBorderGetSize(kernelSize, size, dataType, kernelType, channels, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBorderSpec*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        if(kernelType == ipp16s)
            status = ippiFilterBorderInit_16s((const Ipp16s*)pKernel, kernelSize, divisor, dataType, channels, roundMode, pSpec);
        else if(kernelType == ipp32f)
            status = ippiFilterBorderInit_32f((const Ipp32f*)pKernel, kernelSize, dataType, channels, roundMode, pSpec);
        if(status < 0)
            break;

        status = ippiFilterBorderSetMode(algoMode, offset, pSpec);
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        // Apply filter
        switch(dataType)
        {
        case ipp8u:
            switch(channels)
            {
            case 1:  status = ippiFilterBorder_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBorder_8u_C3R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 3), pSpec, pTmpBuffer); break;
            case 4:  status = ippiFilterBorder_8u_C4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        case ipp16u:
            switch(channels)
            {
            case 1:  status = ippiFilterBorder_16u_C1R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(16u, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBorder_16u_C3R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(16u, 3), pSpec, pTmpBuffer); break;
            case 4:  status = ippiFilterBorder_16u_C4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(16u, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        case ipp16s:
            switch(channels)
            {
            case 1:  status = ippiFilterBorder_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(16s, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBorder_16s_C3R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(16s, 3), pSpec, pTmpBuffer); break;
            case 4:  status = ippiFilterBorder_16s_C4R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(16s, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        case ipp32f:
            switch(channels)
            {
            case 1:  status = ippiFilterBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBorder_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 3), pSpec, pTmpBuffer); break;
            case 4:  status = ippiFilterBorder_32f_C4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        default: status = ippStsDataTypeErr; break;
        }
        break;
    }

    if(pSpec)
        ownSharedFree(pSpec);
    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
