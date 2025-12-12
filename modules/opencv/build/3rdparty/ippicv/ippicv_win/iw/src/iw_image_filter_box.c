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

IW_DECL(IppStatus) llwiFilterBox(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType,
    int channels, IppiSize kernelSize, IwiChDescriptor chDesc, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBox
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterBox(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiSize kernelSize,
    const IwiFilterBoxParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus          status;
    IwiFilterBoxParams auxParams;

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
        iwiFilterBox_SetDefaultParams(&auxParams);

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
                IwiBorderSize  borderSize = iwiSizeToBorderSize(kernelSize);

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

            status = owniLongCompatCheckSize(kernelSize, &_kernelSize);
            if(status < 0)
                return status;

            return llwiFilterBox(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step, _size, pSrcImage->m_dataType,
                pSrcImage->m_channels, _kernelSize, auxParams.chDesc, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterBox(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType,
    int channels, IppiSize kernelSize, IwiChDescriptor chDesc, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus   status;
    OwniChCodes chCode       = owniChDescriptorToCode(chDesc, channels, channels);
    Ipp64f      borderVal[4] = {0};

    Ipp8u   *pTmpBuffer    = 0;
    int      tmpBufferSize = 0;

    for(;;)
    {
        // Initialize Intel IPP functions and check parameters
        status = ippiFilterBoxBorderGetBufferSize(size, kernelSize, dataType, channels, &tmpBufferSize);
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
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        status = ippiFilterBoxBorder_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(8u, 1), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        status = ippiFilterBoxBorder_8u_C3R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(8u, 3), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        status = ippiFilterBoxBorder_8u_C4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(8u, 4), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   status = ippiFilterBoxBorder_8u_AC4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(8u, 3), pTmpBuffer); break;
#endif
            default:            status = ippStsNumChannelsErr; break;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        status = ippiFilterBoxBorder_16u_C1R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16u, 1), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        status = ippiFilterBoxBorder_16u_C3R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16u, 3), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        status = ippiFilterBoxBorder_16u_C4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16u, 4), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   status = ippiFilterBoxBorder_16u_AC4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16u, 3), pTmpBuffer); break;
#endif
            default:            status = ippStsNumChannelsErr; break;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        status = ippiFilterBoxBorder_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16s, 1), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        status = ippiFilterBoxBorder_16s_C3R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16s, 3), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        status = ippiFilterBoxBorder_16s_C4R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16s, 4), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   status = ippiFilterBoxBorder_16s_AC4R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(16s, 3), pTmpBuffer); break;
#endif
            default:            status = ippStsNumChannelsErr; break;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        status = ippiFilterBoxBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(32f, 1), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        status = ippiFilterBoxBorder_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(32f, 3), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        status = ippiFilterBoxBorder_32f_C4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(32f, 4), pTmpBuffer); break;
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   status = ippiFilterBoxBorder_32f_AC4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VALP(32f, 3), pTmpBuffer); break;
#endif
            default:            status = ippStsNumChannelsErr; break;
            }
            break;
#endif
        default: status = ippStsDataTypeErr; break;
        }
        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}

