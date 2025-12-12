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

IW_DECL(IppStatus) llwiSet(const double *pValue, void *pDst, int dstStep,
                        IppiSize size, IppDataType dataType, int channels);
IW_DECL(IppStatus) llwiSetUniform(double value, void *pDst, int dstStep,
                                    IppiSize size, IppDataType dataType, int channels);
IW_DECL(IppStatus) llwiSetMask(const double *pValue, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, const Ipp8u *pMask, int maskStep);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSet
///////////////////////////////////////////////////////////////////////////// */
static IppStatus iwiSet_NoMask(const double *pValue, int valuesNum, IwiImage *pDstImage, const IwiTile *pTile)
{
    {
        void*     pDst    = pDstImage->m_ptr;
        IwiSize   size    = pDstImage->m_size;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pDst  = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi dstLim; iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pDst  = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

                size.width  = dstLim.width;
                size.height = dstLim.height;
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppStatus status;
            IppiSize  _size;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            if(valuesNum == 1)
                return llwiSetUniform(*pValue, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels);
            else
            {
                if(valuesNum < pDstImage->m_channels)
                {
                    int    i;
                    Ipp64f val[4];
                    for(i = 0; i < valuesNum; i++)
                        val[i] = pValue[i];
                    for(i = valuesNum; i < pDstImage->m_channels; i++)
                        val[i] = pValue[valuesNum-1];

                    return llwiSet(val, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels);
                }

                return llwiSet(pValue, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels);
            }
        }
    }
}

IW_DECL(IppStatus) iwiSet(const double *pValue, int valuesNum, IwiImage *pDstImage, const IwiImage *pMaskImage, const IwiSetParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    if(!pValue)
        return ippStsNullPtrErr;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(!pMaskImage || !pMaskImage->m_ptrConst)
        return iwiSet_NoMask(pValue, valuesNum, pDstImage, pTile);

    status = owniCheckImageRead(pMaskImage);
    if(status)
        return status;

    if(pDstImage->m_channels > 4)
        return ippStsNumChannelsErr;

    if(pMaskImage->m_dataType != ipp8u ||
        pMaskImage->m_channels != 1)
        return ippStsBadArgErr;

    {
        const void* pMask   = pMaskImage->m_ptrConst;
        void*       pDst    = pDstImage->m_ptr;
        IwiSize     size    = owniGetMinSize(&pDstImage->m_size, &pMaskImage->m_size);

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pDst  = iwiImage_GetPtr(pDstImage,  dstRoi.y, dstRoi.x, 0);
                pMask = iwiImage_GetPtrConst(pMaskImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi dstLim; iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pDst  = iwiImage_GetPtr(pDstImage,  dstLim.y, dstLim.x, 0);
                pMask = iwiImage_GetPtrConst(pMaskImage, dstLim.y, dstLim.x, 0);

                size.width  = dstLim.width;
                size.height = dstLim.height;
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pMaskImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            if(valuesNum < pDstImage->m_channels)
            {
                int    i;
                Ipp64f val[4];
                for(i = 0; i < valuesNum; i++)
                    val[i] = pValue[i];
                for(i = valuesNum; i < pDstImage->m_channels; i++)
                    val[i] = pValue[valuesNum-1];

                return llwiSetMask(val, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels, (const Ipp8u*)pMask, (int)pMaskImage->m_step);
            }

            return llwiSetMask(pValue, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels, (const Ipp8u*)pMask, (int)pMaskImage->m_step);
        }
    }
}

/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiSet(const double *pValue, void *pDst, int dstStep,
                        IppiSize size, IppDataType dataType, int channels)
{
    Ipp64f value[4];

    switch(dataType)
    {
    case ipp8u:
        switch(channels)
        {
        case 1:  return ippiSet_8u_C1R(ownCast_64f8u(*pValue), (Ipp8u*)pDst, dstStep, size);
        case 3:  return ippiSet_8u_C3R(ownCastArray_64f8u(pValue, value, 3), (Ipp8u*)pDst, dstStep, size);
        case 4:  return ippiSet_8u_C4R(ownCastArray_64f8u(pValue, value, 4), (Ipp8u*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp8s:
        switch(channels)
        {
        case 1:  return ippiSet_8u_C1R(ownCast_64f8s(*pValue), (Ipp8u*)pDst, dstStep, size);
        case 3:  return ippiSet_8u_C3R((Ipp8u*)ownCastArray_64f8s(pValue, value, 3), (Ipp8u*)pDst, dstStep, size);
        case 4:  return ippiSet_8u_C4R((Ipp8u*)ownCastArray_64f8s(pValue, value, 4), (Ipp8u*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp16u:
        switch(channels)
        {
        case 1:  return ippiSet_16u_C1R(ownCast_64f16u(*pValue), (Ipp16u*)pDst, dstStep, size);
        case 3:  return ippiSet_16u_C3R(ownCastArray_64f16u(pValue, value, 3), (Ipp16u*)pDst, dstStep, size);
        case 4:  return ippiSet_16u_C4R(ownCastArray_64f16u(pValue, value, 4), (Ipp16u*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp16s:
        switch(channels)
        {
        case 1:  return ippiSet_16u_C1R(ownCast_64f16s(*pValue), (Ipp16u*)pDst, dstStep, size);
        case 3:  return ippiSet_16u_C3R((Ipp16u*)ownCastArray_64f16s(pValue, value, 3), (Ipp16u*)pDst, dstStep, size);
        case 4:  return ippiSet_16u_C4R((Ipp16u*)ownCastArray_64f16s(pValue, value, 4), (Ipp16u*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32u:
        switch(channels)
        {
        case 1:  return ippiSet_32s_C1R(ownCast_64f32u(*pValue), (Ipp32s*)pDst, dstStep, size);
        case 3:  return ippiSet_32s_C3R((Ipp32s*)ownCastArray_64f32u(pValue, value, 3), (Ipp32s*)pDst, dstStep, size);
        case 4:  return ippiSet_32s_C4R((Ipp32s*)ownCastArray_64f32u(pValue, value, 4), (Ipp32s*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32s:
        switch(channels)
        {
        case 1:  return ippiSet_32s_C1R(ownCast_64f32s(*pValue), (Ipp32s*)pDst, dstStep, size);
        case 3:  return ippiSet_32s_C3R((Ipp32s*)ownCastArray_64f32s(pValue, value, 3), (Ipp32s*)pDst, dstStep, size);
        case 4:  return ippiSet_32s_C4R((Ipp32s*)ownCastArray_64f32s(pValue, value, 4), (Ipp32s*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32f:
        switch(channels)
        {
        case 1:  return ippiSet_32f_C1R(ownCast_64f32f(*pValue), (Ipp32f*)pDst, dstStep, size);
        case 3:  return ippiSet_32f_C3R(ownCastArray_64f32f(pValue, value, 3), (Ipp32f*)pDst, dstStep, size);
        case 4:  return ippiSet_32f_C4R(ownCastArray_64f32f(pValue, value, 4), (Ipp32f*)pDst, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    default: return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiSetUniform(double value, void *pDst, int dstStep,
                                    IppiSize size, IppDataType dataType, int channels)
{
    size.width = size.width*channels;

    switch(dataType)
    {
    case ipp8u:  return ippiSet_8u_C1R(ownCast_64f8u(value), (Ipp8u*)pDst, dstStep, size);
    case ipp8s:  return ippiSet_8u_C1R(ownCast_64f8s(value), (Ipp8u*)pDst, dstStep, size);
    case ipp16u: return ippiSet_16u_C1R(ownCast_64f16u(value), (Ipp16u*)pDst, dstStep, size);
    case ipp16s: return ippiSet_16u_C1R(ownCast_64f16s(value), (Ipp16u*)pDst, dstStep, size);
    case ipp32u: return ippiSet_32s_C1R(ownCast_64f32u(value), (Ipp32s*)pDst, dstStep, size);
    case ipp32s: return ippiSet_32s_C1R(ownCast_64f32s(value), (Ipp32s*)pDst, dstStep, size);
    case ipp32f: return ippiSet_32f_C1R(ownCast_64f32f(value), (Ipp32f*)pDst, dstStep, size);
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiSetMask(const double *pValue, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, const Ipp8u *pMask, int maskStep)
{
    Ipp64f value[4];

    switch(dataType)
    {
    case ipp8u:
        switch(channels)
        {
        case 1:  return ippiSet_8u_C1MR(ownCast_64f8u(*pValue), (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_8u_C3MR(ownCastArray_64f8u(pValue, value, 3), (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_8u_C4MR(ownCastArray_64f8u(pValue, value, 4), (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    case ipp8s:
        switch(channels)
        {
        case 1:  return ippiSet_8u_C1MR(ownCast_64f8s(*pValue), (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_8u_C3MR((Ipp8u*)ownCastArray_64f8s(pValue, value, 3), (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_8u_C4MR((Ipp8u*)ownCastArray_64f8s(pValue, value, 4), (Ipp8u*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    case ipp16u:
        switch(channels)
        {
        case 1:  return ippiSet_16u_C1MR(ownCast_64f16u(*pValue), (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_16u_C3MR(ownCastArray_64f16u(pValue, value, 3), (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_16u_C4MR(ownCastArray_64f16u(pValue, value, 4), (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    case ipp16s:
        switch(channels)
        {
        case 1:  return ippiSet_16u_C1MR(ownCast_64f16s(*pValue), (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_16u_C3MR((Ipp16u*)ownCastArray_64f16s(pValue, value, 3), (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_16u_C4MR((Ipp16u*)ownCastArray_64f16s(pValue, value, 4), (Ipp16u*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    case ipp32u:
        switch(channels)
        {
        case 1:  return ippiSet_32s_C1MR(ownCast_64f32u(*pValue), (Ipp32s*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_32s_C3MR((Ipp32s*)ownCastArray_64f32u(pValue, value, 3), (Ipp32s*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_32s_C4MR((Ipp32s*)ownCastArray_64f32u(pValue, value, 4), (Ipp32s*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    case ipp32s:
        switch(channels)
        {
        case 1:  return ippiSet_32s_C1MR(ownCast_64f32s(*pValue), (Ipp32s*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_32s_C3MR(ownCastArray_64f32s(pValue, value, 3), (Ipp32s*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_32s_C4MR(ownCastArray_64f32s(pValue, value, 4), (Ipp32s*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    case ipp32f:
        switch(channels)
        {
        case 1:  return ippiSet_32f_C1MR(ownCast_64f32f(*pValue), (Ipp32f*)pDst, dstStep, size, pMask, maskStep);
        case 3:  return ippiSet_32f_C3MR(ownCastArray_64f32f(pValue, value, 3), (Ipp32f*)pDst, dstStep, size, pMask, maskStep);
        case 4:  return ippiSet_32f_C4MR(ownCastArray_64f32f(pValue, value, 4), (Ipp32f*)pDst, dstStep, size, pMask, maskStep);
        default: return ippStsNumChannelsErr;
        }
    default: return ippStsDataTypeErr;
    }
}
