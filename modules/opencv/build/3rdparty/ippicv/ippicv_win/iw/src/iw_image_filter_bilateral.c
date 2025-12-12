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

IW_DECL(IppStatus) llwiFilterBilateral(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal);

#if IW_ENABLE_THREADING_LAYER
IW_DECL(IppStatus) llwiFilterBilateral_TL(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal);
#endif

IW_DECL(IppStatus) llwiFilterBilateral_classic(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBilateral
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterBilateral(const IwiImage *pSrcImage, IwiImage *pDstImage, int radius,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, const IwiFilterBilateralParams *pAuxParams,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus                status;
    IwiFilterBilateralParams auxParams;

    status = owniCheckImageRead(pSrcImage);
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

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiFilterBilateral_SetDefaultParams(&auxParams);

#if IW_ENABLE_THREADING_LAYER
    if(iwGetThreadsNum() > 1)
    {
        IwiSize size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);
        return llwiFilterBilateral_TL(pSrcImage->m_ptr, pSrcImage->m_step, pDstImage->m_ptr, pDstImage->m_step,
            size, pSrcImage->m_dataType, pSrcImage->m_channels, auxParams.filter, radius, auxParams.distMethod, valSquareSigma, posSquareSigma, border, pBorderVal);
    }
    else
#endif
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
                IwiRoi         dstRoi = pTile->m_dstRoi;
                IwiBorderSize  borderSize = iwiSizeSymToBorderSize(radius*2);

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
        if(pSrcImage->m_dataType == ipp32f)
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

            return llwiFilterBilateral_classic(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step,
                _size, pSrcImage->m_dataType, pSrcImage->m_channels, auxParams.filter, radius, auxParams.distMethod, valSquareSigma, posSquareSigma, border, pBorderVal);
        }
        else
        {
            return llwiFilterBilateral(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step,
                size, pSrcImage->m_dataType, pSrcImage->m_channels, auxParams.filter, radius, auxParams.distMethod, valSquareSigma, posSquareSigma, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterBilateral(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBilateralSpec *pSpec       = 0;
    IwSize                   specSize    = 0;

    Ipp8u    *pTmpBuffer    = 0;
    IwSize    tmpBufferSize = 0;

    for(;;)
    {
        status = ippiFilterBilateralBorderGetBufferSize_L(filter, size, radius, dataType, channels, distMethod, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBilateralSpec*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        status = ippiFilterBilateralBorderInit_L(filter, size, radius, dataType, channels, distMethod, valSquareSigma, posSquareSigma, pSpec);
        if(status < 0)
            break;

        switch(dataType)
        {
        case ipp8u:
            switch(channels)
            {
            case 1:  status = ippiFilterBilateralBorder_8u_C1R_L((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBilateralBorder_8u_C3R_L((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 3), pSpec, pTmpBuffer); break;
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

#if IW_ENABLE_THREADING_LAYER
IW_DECL(IppStatus) llwiFilterBilateral_TL(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBilateralSpec_LT *pSpec       = 0;
    IwSize                      specSize    = 0;

    Ipp8u    *pTmpBuffer    = 0;
    IwSize    tmpBufferSize = 0;

    for(;;)
    {
        status = ippiFilterBilateralBorderGetBufferSize_LT(filter, size, radius, dataType, channels, distMethod, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBilateralSpec_LT*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        status = ippiFilterBilateralBorderInit_LT(filter, size, radius, dataType, channels, distMethod, valSquareSigma, posSquareSigma, pSpec);
        if(status < 0)
            break;

        switch(dataType)
        {
        case ipp8u:
            switch(channels)
            {
            case 1:  status = ippiFilterBilateralBorder_8u_C1R_LT((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBilateralBorder_8u_C3R_LT((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 3), pSpec, pTmpBuffer); break;
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
#endif

IW_DECL(IppStatus) llwiFilterBilateral_classic(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBilateralSpec *pSpec       = 0;
    int                      specSize    = 0;

    Ipp8u    *pTmpBuffer    = 0;
    int       tmpBufferSize = 0;

    for(;;)
    {

        status = ippiFilterBilateralBorderGetBufferSize(filter, size, radius, dataType, channels, distMethod, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBilateralSpec*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        status = ippiFilterBilateralBorderInit(filter, size, radius, dataType, channels, distMethod, valSquareSigma, posSquareSigma, pSpec);
        if(status < 0)
            break;

        switch(dataType)
        {
        case ipp32f:
            switch(channels)
            {
            case 1:  status = ippiFilterBilateralBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBilateralBorder_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 3), pSpec, pTmpBuffer); break;
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
