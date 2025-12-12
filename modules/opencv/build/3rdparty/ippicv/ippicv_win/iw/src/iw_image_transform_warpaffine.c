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

#include "iw/iw_image_transform.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiWarpAffine(const IwiWarpAffineSpec* pSpec, const void *pSrc, int srcStep, void *pDst, int dstStep,
                                    IppiPoint dstRoiOffset, IppiSize dstRoiSize);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiWarpAffine
///////////////////////////////////////////////////////////////////////////// */
struct _IwiWarpAffineSpec
{
    IwiSize                 srcSize;
    IwiSize                 dstSize;
    IppDataType             dataType;
    int                     channels;
    IppiInterpolationType   interpolation;

    IwiWarpAffineParams     params;

    IwiBorderType           border;

    IppiWarpSpec           *pSpec;

    unsigned int initialized;
};

IW_DECL(IppStatus) iwiWarpAffine(const IwiImage *pSrcImage, IwiImage *pDstImage, const double coeffs[2][3], IwTransDirection direction,
    IppiInterpolationType interpolation, const IwiWarpAffineParams *pParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus          status;
    IwiWarpAffineSpec *pSpec  = NULL;

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

    for(;;)
    {
        status = iwiWarpAffine_InitAlloc(&pSpec, pSrcImage->m_size, pDstImage->m_size, pSrcImage->m_dataType, pSrcImage->m_channels, coeffs, direction, interpolation, pParams, border, pBorderVal);
        if(status < 0)
            break;

        status = iwiWarpAffine_Process(pSpec, pSrcImage, pDstImage, pTile);
        break;
    }

    iwiWarpAffine_Free(pSpec);

    return status;
}

IW_DECL(IppStatus) iwiWarpAffine_Free(IwiWarpAffineSpec *pSpec)
{
    if(!pSpec)
        return ippStsNoErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    pSpec->initialized = 0;
    if(pSpec->pSpec)
    {
        OWN_MEM_FREE(pSpec->pSpec);
        pSpec->pSpec = 0;
    }

    OWN_MEM_FREE(pSpec);
    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiWarpAffine_InitAlloc(IwiWarpAffineSpec **ppSpec, IwiSize _srcSize, IwiSize _dstSize, IppDataType dataType, int channels,
    const double coeffs[2][3], IwTransDirection _direction, IppiInterpolationType interpolation, const IwiWarpAffineParams *pParams, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;

    int      specSize = 0;
    Ipp8u*   pInitBuf = 0;
    int      initSize = 0;
    IppiSize srcSize;
    IppiSize dstSize;
    IppiWarpDirection direction = (_direction == iwTransForward)?ippWarpForward:ippWarpBackward;

    IwiWarpAffineSpec spec;

    if(!ppSpec)
        return ippStsNullPtrErr;

    // Long compatibility check
    {
        int typeSize = iwTypeToSize(dataType);

        status = owniLongCompatCheckSize(_srcSize, &srcSize);
        if(status < 0)
            return status;

        status = owniLongCompatCheckSize(_dstSize, &dstSize);
        if(status < 0)
            return status;

        status = ownLongCompatCheckValue(srcSize.width*channels*typeSize, NULL);
        if(status < 0)
            return status;

        status = ownLongCompatCheckValue(dstSize.width*channels*typeSize, NULL);
        if(status < 0)
            return status;
    }

    if(!srcSize.width || !srcSize.height ||
        !dstSize.width || !dstSize.height)
        return ippStsNoOperation;

    OWN_MEM_RESET(&spec);

    // Update spec
    spec.srcSize       = _srcSize;
    spec.dstSize       = _dstSize;
    spec.channels      = channels;
    spec.dataType      = dataType;
    spec.interpolation = interpolation;

    if(pParams)
        spec.params = *pParams;
    else
        iwiWarpAffine_SetDefaultParams(&spec.params);

    spec.border = border;

    for(;;)
    {
        status = ippiWarpAffineGetSize(srcSize, dstSize, dataType, coeffs, interpolation, direction, border, &specSize, &initSize);
        if(status < 0)
            break;

        spec.pSpec = (IppiWarpSpec*)OWN_MEM_ALLOC(specSize);
        if(!spec.pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pInitBuf = (Ipp8u*)OWN_MEM_ALLOC(initSize);
        if(initSize && !pInitBuf)
        {
            status = ippStsNoMemErr;
            break;
        }

        switch(interpolation)
        {
        case ippNearest: status = ippiWarpAffineNearestInit(srcSize, dstSize, dataType, coeffs, direction, channels, border, pBorderVal, spec.params.smoothEdge, spec.pSpec); break;
        case ippLinear:  status = ippiWarpAffineLinearInit(srcSize, dstSize, dataType, coeffs, direction, channels, border, pBorderVal, spec.params.smoothEdge, spec.pSpec); break;
        case ippCubic:   status = ippiWarpAffineCubicInit(srcSize, dstSize, dataType, coeffs, direction, channels, spec.params.cubicBVal, spec.params.cubicCVal, border, pBorderVal, spec.params.smoothEdge, spec.pSpec, pInitBuf); break;
        default:         status = ippStsInterpolationErr; break;
        }
        if(status < 0)
            break;

        break;
    }

    if(pInitBuf)
        OWN_MEM_FREE(pInitBuf);

    if(status < 0)
        return status;

    spec.initialized = OWN_INIT_MAGIC_NUM;
    // Allocate spec structure
    *ppSpec = (IwiWarpAffineSpec*)OWN_MEM_ALLOC(sizeof(IwiWarpAffineSpec));
    if(!*ppSpec)
        return ippStsNoMemErr;
    **ppSpec = spec;

    return status;
}

IW_DECL(IppStatus) iwiWarpAffine_Process(const IwiWarpAffineSpec* pSpec, const IwiImage *pSrcImage, IwiImage *pDstImage, const IwiTile *pTile)
{
    IppStatus status;

    if(!pSpec)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;
    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsInplaceModeNotSupportedErr;

    if(pSpec->channels != pSrcImage->m_channels ||
        pSpec->dataType != pSrcImage->m_dataType ||
        pSpec->srcSize.width != pSrcImage->m_size.width ||
        pSpec->srcSize.height != pSrcImage->m_size.height ||
        pSpec->dstSize.width != pDstImage->m_size.width ||
        pSpec->dstSize.height != pDstImage->m_size.height)
        return ippStsBadArgErr;

    if(pSrcImage->m_dataType != pDstImage->m_dataType ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    {
        const void *pSrc         = pSrcImage->m_ptrConst;
        void       *pDst         = pDstImage->m_ptr;
        IppiPointL  dstRoiOffset = {0, 0};
        IwiSize     dstRoiSize   = pSpec->dstSize;
        if(!dstRoiSize.width || !dstRoiSize.height)
            return ippStsNoOperation;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pSpec->border == ippBorderWrap)
                return ippStsNotSupportedModeErr;

            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &dstRoiSize))
                    return ippStsNoOperation;

                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);

                dstRoiOffset.x = dstRoi.x;
                dstRoiOffset.y = dstRoi.y;
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                return ippStsNotSupportedModeErr;
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize  size;
            IppiPoint offset;

            status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(dstRoiSize, &size);
            if(status < 0)
                return status;

            status = owniLongCompatCheckPoint(dstRoiOffset, &offset);
            if(status < 0)
                return status;

            return llwiWarpAffine(pSpec, pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step, offset, size);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiWarpAffine(const IwiWarpAffineSpec* pSpec, const void *pSrc, int srcStep, void *pDst, int dstStep,
                                    IppiPoint dstRoiOffset, IppiSize dstRoiSize)
{
    IppStatus status;
    Ipp8u    *pTmpBuffer    = 0;
    int       tmpBufferSize = 0;

    for(;;)
    {
        // Get working buffer
        status = ippiWarpGetBufferSize(pSpec->pSpec, dstRoiSize, &tmpBufferSize);
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        // Perform warp
        switch(pSpec->interpolation)
        {
        case ippNearest:
            switch(pSpec->dataType)
            {
            case ipp8u:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineNearest_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineNearest_8u_C3R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineNearest_8u_C4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineNearest_16u_C1R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineNearest_16u_C3R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineNearest_16u_C4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineNearest_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineNearest_16s_C3R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineNearest_16s_C4R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineNearest_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineNearest_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineNearest_32f_C4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp64f:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineNearest_64f_C1R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineNearest_64f_C3R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineNearest_64f_C4R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
            break;
        case ippLinear:
            switch(pSpec->dataType)
            {
            case ipp8u:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineLinear_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineLinear_8u_C3R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineLinear_8u_C4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineLinear_16u_C1R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineLinear_16u_C3R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineLinear_16u_C4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineLinear_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineLinear_16s_C3R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineLinear_16s_C4R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineLinear_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineLinear_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineLinear_32f_C4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp64f:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineLinear_64f_C1R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineLinear_64f_C3R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineLinear_64f_C4R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
            break;
        case ippCubic:
            switch(pSpec->dataType)
            {
            case ipp8u:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineCubic_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineCubic_8u_C3R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineCubic_8u_C4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineCubic_16u_C1R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineCubic_16u_C3R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineCubic_16u_C4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineCubic_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineCubic_16s_C3R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineCubic_16s_C4R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineCubic_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineCubic_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineCubic_32f_C4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp64f:
                switch(pSpec->channels)
                {
                case 1:  status = ippiWarpAffineCubic_64f_C1R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 3:  status = ippiWarpAffineCubic_64f_C3R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                case 4:  status = ippiWarpAffineCubic_64f_C4R((Ipp64f*)pSrc, srcStep, (Ipp64f*)pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pSpec, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
            break;
        default: status = ippStsInterpolationErr; break;
        }

        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
