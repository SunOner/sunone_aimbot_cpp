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

IW_DECL(IppStatus) llwiScale(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                               IppiSize size, int channels, Ipp64f mulVal, Ipp64f addVal, IppHintAlgorithm algoMode);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiScale
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiScale_GetScaleVals(IppDataType srcType, IppDataType dstType, Ipp64f *pMulVal, Ipp64f *pAddVal)
{
    Ipp64f srcRange = 1, dstRange = 1;
    Ipp64f srcMin   = 0, dstMin   = 0;

    if(!pMulVal || !pAddVal)
        return ippStsNullPtrErr;

    if(srcType == dstType)
    {
        *pMulVal = 1;
        *pAddVal = 0;
    }

    if(!iwTypeIsFloat(srcType))
    {
        srcRange = iwTypeGetRange(srcType);
        srcMin   = iwTypeGetMin(srcType);
    }
    if(!iwTypeIsFloat(dstType))
    {
        dstRange = iwTypeGetRange(dstType);
        dstMin   = iwTypeGetMin(dstType);
    }
    if(!srcRange || !dstRange)
        return ippStsDataTypeErr;

    *pMulVal = dstRange/srcRange;
    *pAddVal = dstMin - srcMin*(*pMulVal);

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiScale(const IwiImage *pSrcImage, IwiImage *pDstImage, Ipp64f mulVal, Ipp64f addVal, const IwiScaleParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus      status;
    IwiScaleParams auxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if((pSrcImage->m_ptrConst == pDstImage->m_ptrConst) && (pSrcImage->m_dataType != pDstImage->m_dataType))
        return ippStsInplaceModeNotSupportedErr;

    if(pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiScale_SetDefaultParams(&auxParams);

    if(auxParams.algoMode == ippAlgHintNone)
    {
        int hasScale = ((IPP_ABS(mulVal-1) > IPP_EPS_64F) || (IPP_ABS(addVal) > IPP_EPS_64F));
        if(hasScale && (pDstImage->m_typeSize >= 4 && pDstImage->m_dataType != ipp32f))
            auxParams.algoMode = ippAlgHintAccurate;
        else
            auxParams.algoMode = ippAlgHintFast;
    }

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

            return llwiScale(pSrc, (int)pSrcImage->m_step, pSrcImage->m_dataType, pDst, (int)pDstImage->m_step, pDstImage->m_dataType, _size, pSrcImage->m_channels, mulVal, addVal, auxParams.algoMode);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiScale(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                               IppiSize size, int channels, Ipp64f mulVal, Ipp64f addVal, IppHintAlgorithm algoMode)
{
    size.width = size.width*channels;

    if(pSrc == pDst)
    {
#if IPP_VERSION_COMPLEX >= 20170001
        switch(srcType)
        {
        case ipp8u:     return ippiScaleC_8u_C1IR ((Ipp8u*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        case ipp8s:     return ippiScaleC_8s_C1IR ((Ipp8s*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        case ipp16u:    return ippiScaleC_16u_C1IR((Ipp16u*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        case ipp16s:    return ippiScaleC_16s_C1IR((Ipp16s*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        case ipp32s:    return ippiScaleC_32s_C1IR((Ipp32s*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        case ipp32f:    return ippiScaleC_32f_C1IR((Ipp32f*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        case ipp64f:    return ippiScaleC_64f_C1IR((Ipp64f*)pSrc, srcStep, mulVal, addVal, size, algoMode);
        default: return ippStsDataTypeErr;
        }
#else
        return ippStsInplaceModeNotSupportedErr;
#endif
    }
    else
    {
        switch(srcType)
        {
        case ipp8u:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_8u_C1R   ((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_8u8s_C1R ((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_8u16u_C1R((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_8u16s_C1R((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_8u32s_C1R((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_8u32f_C1R((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_8u64f_C1R((Ipp8u*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        case ipp8s:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_8s8u_C1R ((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_8s_C1R   ((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_8s16u_C1R((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_8s16s_C1R((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_8s32s_C1R((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_8s32f_C1R((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_8s64f_C1R((Ipp8s*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        case ipp16u:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_16u8u_C1R ((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_16u8s_C1R ((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_16u_C1R   ((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_16u16s_C1R((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_16u32s_C1R((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_16u32f_C1R((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_16u64f_C1R((Ipp16u*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        case ipp16s:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_16s8u_C1R ((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_16s8s_C1R ((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_16s16u_C1R((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_16s_C1R   ((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_16s32s_C1R((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_16s32f_C1R((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_16s64f_C1R((Ipp16s*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        case ipp32s:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_32s8u_C1R ((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_32s8s_C1R ((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_32s16u_C1R((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_32s16s_C1R((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_32s_C1R   ((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_32s32f_C1R((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_32s64f_C1R((Ipp32s*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        case ipp32f:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_32f8u_C1R ((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_32f8s_C1R ((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_32f16u_C1R((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_32f16s_C1R((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_32f32s_C1R((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_32f_C1R   ((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_32f64f_C1R((Ipp32f*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        case ipp64f:
            switch(dstType)
            {
            case ipp8u:  return ippiScaleC_64f8u_C1R ((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp8u*)pDst, dstStep, size, algoMode);
            case ipp8s:  return ippiScaleC_64f8s_C1R ((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp8s*)pDst, dstStep, size, algoMode);
            case ipp16u: return ippiScaleC_64f16u_C1R((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp16u*)pDst, dstStep, size, algoMode);
            case ipp16s: return ippiScaleC_64f16s_C1R((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp16s*)pDst, dstStep, size, algoMode);
            case ipp32s: return ippiScaleC_64f32s_C1R((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp32s*)pDst, dstStep, size, algoMode);
            case ipp32f: return ippiScaleC_64f32f_C1R((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp32f*)pDst, dstStep, size, algoMode);
            case ipp64f: return ippiScaleC_64f_C1R   ((Ipp64f*)pSrc, srcStep, mulVal, addVal, (Ipp64f*)pDst, dstStep, size, algoMode);
            default:     return ippStsDataTypeErr;
            }
        default: return ippStsDataTypeErr;
        }
    }
}
