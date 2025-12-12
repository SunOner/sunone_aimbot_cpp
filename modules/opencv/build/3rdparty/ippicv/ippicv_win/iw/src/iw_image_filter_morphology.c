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
#include "iw/iw_image_op.h"
#include "iw_owni.h"

typedef IppStatus (IPP_STDCALL *IppiFilterMorphology_ptr)(const void* pSrc, IppSizeL srcStep, void* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, const void *borderValue, const void* pMorthSpec, Ipp8u* pBuffer);

typedef IppStatus (IPP_STDCALL *IppiFilterMorphology_GetBufferSize_ptr)(IppiSizeL roiSize, IppiSizeL maskSize, IppDataType depth, int numChannels, IppSizeL* bufferSize);


struct _IwiFilterMorphologySpec
{
    // Init params
    void                                   *pIppSpec;
    IwiFilterMorphologyParams               auxParams;
    IppiFilterMorphology_ptr                ippiFilterMorphology;
    IppiFilterMorphology_GetBufferSize_ptr  ippiFilterMorphology_GetBufferSize;
    OwnCastArray_ptr                        borderCastFun;
    IwiMorphologyType                       morphType;
    IwiSize                                 maskSize;
    IppDataType                             dataType;
    int                                     channels;

    unsigned int initialized;
};
typedef struct _IwiFilterMorphologySpec IwiFilterMorphologySpec;

IW_DECL(IppStatus) iwiFilterMorphology_InitAlloc(IwiFilterMorphologySpec **ppSpec, IwiSize size, IppDataType dataType, int channels,
    IwiMorphologyType morphType, const IwiImage *pMaskImage, const IwiFilterMorphologyParams *pAuxParams, IwiBorderType border);

IW_DECL(IppStatus) iwiFilterMorphology_Free(IwiFilterMorphologySpec *pSpec);

IW_DECL(IppStatus) iwiFilterMorphology_Process(const IwiFilterMorphologySpec *pSpec, const IwiImage *pSrcImage, IwiImage *pDstImage,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile);


IW_DECL(IppStatus) llwiFilterMorphology_InitAlloc(IwiSize size, IppDataType dataType, int channels, IwiMorphologyType morphType, const IwiImage *pMaskImage,
    const IwiFilterMorphologyParams *pAuxParams, IwiBorderType border, IwiFilterMorphologySpec *pSpec);

IW_DECL(void)      llwiFilterMorphology_Free(IwiFilterMorphologySpec *pSpec);

IW_DECL(IppStatus) llwiFilterMorphology_ProcessWrap(const IwiImage *pSrcImage, IwiImage *pDstImage,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile, const IwiFilterMorphologySpec *pSpec);

IW_DECL(IppStatus) llwiFilterMorphology_Process(const void *pSrc, IppSizeL srcStep, void *pDst, IppSizeL dstStep, IppiSizeL roi,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiFilterMorphologySpec *pSpec);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterMorphology
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterMorphology(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiMorphologyType morphType, const IwiImage *pMaskImage,
    const IwiFilterMorphologyParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus               status;
    IwiFilterMorphologySpec   spec;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageRead(pMaskImage);
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

    if((pAuxParams && pAuxParams->iterations == 0) || (pMaskImage->m_size.width == 1 && pMaskImage->m_size.height == 1))
        return iwiCopy(pSrcImage, pDstImage, NULL, NULL, pTile);

    {
        IwiSize size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        status = llwiFilterMorphology_InitAlloc(size, pSrcImage->m_dataType, pSrcImage->m_channels, morphType, pMaskImage, pAuxParams, border, &spec);
        if(status < 0)
            return status;

        status = llwiFilterMorphology_ProcessWrap(pSrcImage, pDstImage, border, pBorderVal, pTile, &spec);
        llwiFilterMorphology_Free(&spec);
    }

    return status;
}

IW_DECL(IppStatus) iwiFilterMorphology_InitAlloc(IwiFilterMorphologySpec **ppSpec, IwiSize size, IppDataType dataType, int channels,
    IwiMorphologyType morphType, const IwiImage *pMaskImage, const IwiFilterMorphologyParams *pAuxParams, IwiBorderType border)
{
    IppStatus               status;
    IwiFilterMorphologySpec   spec;

    status = owniCheckImageRead(pMaskImage);
    if(status)
        return status;

    if(!ppSpec)
        return ippStsNullPtrErr;

    status = llwiFilterMorphology_InitAlloc(size, dataType, channels, morphType, pMaskImage, pAuxParams, border, &spec);
    if(status < 0)
        return status;

    *ppSpec = (IwiFilterMorphologySpec*)OWN_MEM_ALLOC(sizeof(IwiFilterMorphologySpec));
    if(!*ppSpec)
        return ippStsNoMemErr;
    **ppSpec = spec;

    return status;
}

IW_DECL(IppStatus) iwiFilterMorphology_Free(IwiFilterMorphologySpec *pSpec)
{
    if(!pSpec)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    llwiFilterMorphology_Free(pSpec);
    OWN_MEM_FREE(pSpec);

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiFilterMorphology_Process(const IwiFilterMorphologySpec *pSpec, const IwiImage *pSrcImage, IwiImage *pDstImage,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus status;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsInplaceModeNotSupportedErr;

    if(!pSpec)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    if(pSpec->dataType != pSrcImage->m_dataType ||
        pSpec->channels != pSrcImage->m_channels)
        return ippStsBadArgErr;

    if(pSpec->dataType != pDstImage->m_dataType ||
        pSpec->channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(pSpec->auxParams.iterations == 0 || (pSpec->maskSize.width == 1 && pSpec->maskSize.height == 1))
        return iwiCopy(pSrcImage, pDstImage, NULL, NULL, pTile);

    return llwiFilterMorphology_ProcessWrap(pSrcImage, pDstImage, border, pBorderVal, pTile, pSpec);
}

IW_DECL(IppStatus) iwiFilterMorphology_GetBorderSize(IwiMorphologyType morphType, IwiSize maskSize, IwiBorderSize *pBorderSize)
{
    if(!pBorderSize)
        return ippStsNullPtrErr;

    switch(morphType)
    {
    case iwiMorphErode:
    case iwiMorphDilate:
    case iwiMorphGradient:
        *pBorderSize = iwiSizeToBorderSize(maskSize);
        break;
    case iwiMorphOpen:
    case iwiMorphClose:
    case iwiMorphTophat:
    case iwiMorphBlackhat:
        *pBorderSize = iwiSizeToBorderSize(maskSize);
        pBorderSize->left   *= 2;
        pBorderSize->top    *= 2;
        pBorderSize->right  *= 2;
        pBorderSize->bottom *= 2;
        break;
    default:
        return ippStsNotSupportedModeErr;
    }
    return ippStsNoErr;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterMorphology_ProcessWrap(const IwiImage *pSrcImage, IwiImage *pDstImage,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile, const IwiFilterMorphologySpec *pSpec)
{
    IppStatus   status;
    const void *pSrc = pSrcImage->m_ptrConst;
    void       *pDst = pDstImage->m_ptr;
    IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

    if(pTile && pTile->m_initialized != ownTileInitNone)
    {
        IwiImage srcSubImage = *pSrcImage;
        IwiImage dstSubImage = *pDstImage;

        if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
            return ippStsNotSupportedModeErr;

        if(pTile->m_initialized == ownTileInitSimple)
        {
            IwiRoi         dstRoi     = pTile->m_dstRoi;
            IwiBorderSize  borderSize;
            iwiFilterMorphology_GetBorderSize(pSpec->morphType, pSpec->maskSize, &borderSize);

            if(!owniTile_BoundToSize(&dstRoi, &size))
                return ippStsNoOperation;
            owniTile_CorrectBordersOverlap(&dstRoi, &size, &border, &borderSize, &borderSize, &pSrcImage->m_size);
            owniTile_GetTileBorder(&border, &dstRoi, &borderSize, &pSrcImage->m_size);

            iwiImage_RoiSet(&srcSubImage, dstRoi);
            iwiImage_RoiSet(&dstSubImage, dstRoi);
        }
        else if(pTile->m_initialized == ownTileInitPipe)
        {
            iwiImage_RoiSet(&srcSubImage, pTile->m_boundSrcRoi);
            iwiImage_RoiSet(&dstSubImage, pTile->m_boundDstRoi);

            status = owniTilePipeline_ProcBorder(pTile, &srcSubImage, &border, pBorderVal);
            if(status < 0)
                return status;
        }
        else
            return ippStsContextMatchErr;

        return llwiFilterMorphology_ProcessWrap(&srcSubImage, &dstSubImage, border, pBorderVal, NULL, pSpec);
    }

    status = llwiFilterMorphology_Process(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step, size, border, pBorderVal, pSpec);
    return status;
}

IW_DECL(IppStatus) llwiFilterMorphology_Process(const void *pSrc, IppSizeL srcStep, void *pDst, IppSizeL dstStep, IppiSizeL roi,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiFilterMorphologySpec *pSpec)
{
    IppStatus status;
    Ipp64f    borderVal[4];

    Ipp8u   *pTmpBuffer    = 0;
    IwSize   tmpBufferSize = 0;

    if(!pSpec || !pSpec->ippiFilterMorphology)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    for(;;)
    {
        status = pSpec->ippiFilterMorphology_GetBufferSize(roi, pSpec->maskSize, pSpec->dataType, pSpec->channels, &tmpBufferSize);
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        if(OWN_GET_PURE_BORDER(border) == ippBorderConst && pBorderVal)
            pSpec->borderCastFun(pBorderVal, borderVal, pSpec->channels);

        status = pSpec->ippiFilterMorphology(pSrc, srcStep, pDst, dstStep, roi, border, borderVal, pSpec->pIppSpec, pTmpBuffer);
        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}

IW_DECL(IppStatus) llwiFilterMorphology_InitAlloc(IwiSize size, IppDataType dataType, int channels, IwiMorphologyType morphType, const IwiImage *pMaskImage,
    const IwiFilterMorphologyParams *pAuxParams, IwiBorderType border, IwiFilterMorphologySpec *pSpec)
{
    OwniChCodes chCode;

    OWN_MEM_RESET(pSpec);
    (void)border;

    if(pAuxParams)
        pSpec->auxParams = *pAuxParams;
    else
        iwiFilterMorphology_SetDefaultParams(&pSpec->auxParams);

    if(pSpec->auxParams.iterations < 0 || pSpec->auxParams.iterations > 1)
        return ippStsBadArgErr;

    if(pMaskImage->m_channels != 1 || pMaskImage->m_dataType != ipp8u)
        return ippStsBadArgErr;

    chCode = owniChDescriptorToCode(iwiChDesc_None, channels, channels);

    switch(morphType)
    {
    case iwiMorphErode:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiErode_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    case iwiMorphDilate:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiDilate_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    case iwiMorphOpen:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphOpen_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    case iwiMorphClose:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphClose_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    case iwiMorphTophat:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphTophat_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    case iwiMorphBlackhat:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphBlackhat_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    case iwiMorphGradient:
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_8u_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_16u_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_16s_C1R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:    pSpec->ippiFilterMorphology = (IppiFilterMorphology_ptr)ippiMorphGradient_32f_C4R_L; break;
#endif
            default:        return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
        break;

    default: return ippStsNotSupportedModeErr;
    }
    if(!pSpec->ippiFilterMorphology)
        return ippStsBadArgErr;

    switch(dataType)
    {
    case ipp8u:     pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f8u;    break;
    case ipp16u:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f16u;   break;
    case ipp16s:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f16s;   break;
    case ipp32f:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f32f;   break;
    default:        return ippStsDataTypeErr;
    }
    if(!pSpec->borderCastFun)
        return ippStsBadArgErr;

    switch(morphType)
    {
    case iwiMorphErode:     pSpec->ippiFilterMorphology_GetBufferSize = (IppiFilterMorphology_GetBufferSize_ptr)ippiErodeGetBufferSize_L; break;
    case iwiMorphDilate:    pSpec->ippiFilterMorphology_GetBufferSize = (IppiFilterMorphology_GetBufferSize_ptr)ippiDilateGetBufferSize_L; break;
    default:                pSpec->ippiFilterMorphology_GetBufferSize = (IppiFilterMorphology_GetBufferSize_ptr)ippiMorphGetBufferSize_L; break;
    }
    if(!pSpec->ippiFilterMorphology_GetBufferSize)
        return ippStsBadArgErr;

    pSpec->channels    = channels;
    pSpec->dataType    = dataType;
    pSpec->morphType   = morphType;
    pSpec->maskSize    = pMaskImage->m_size;

    {
        IppStatus    status;
        IwSize       specSize  = 0;
        Ipp8u       *pMask     = (Ipp8u*)pMaskImage->m_ptrConst;
        int          maskCopy  = 0;

        if(pSpec->pIppSpec)
            return ippStsContextMatchErr;

        if(pMaskImage->m_step != pMaskImage->m_size.width)
        {
            pMask = ippMalloc_L(pMaskImage->m_size.width*pMaskImage->m_size.height);
            ippiCopy_8u_C1R_L(pMaskImage->m_ptrConst, pMaskImage->m_step, pMask, pMaskImage->m_size.width, pMaskImage->m_size);
            maskCopy = 1;
        }

        for(;;)
        {
            switch(morphType)
            {
            case iwiMorphErode:     status = ippiErodeGetSpecSize_L(size, pSpec->maskSize, &specSize); break;
            case iwiMorphDilate:    status = ippiDilateGetSpecSize_L(size, pSpec->maskSize, &specSize); break;
            default:                status = ippiMorphGetSpecSize_L(size, pSpec->maskSize, dataType, channels, &specSize); break;
            }
            if(status < 0)
                break;

            pSpec->pIppSpec = ownSharedMalloc(specSize);
            if(!pSpec->pIppSpec)
            {
                status = ippStsNoMemErr;
                break;
            }

            switch(morphType)
            {
            case iwiMorphDilate:
            case iwiMorphClose:
            case iwiMorphBlackhat:
            case iwiMorphGradient:
            {
                IppiSize _size;
                _size.width  = (int)pMaskImage->m_size.width;
                _size.height = (int)pMaskImage->m_size.height;

                if(!maskCopy)
                {
                    pMask  = ippMalloc_L(pMaskImage->m_size.width*pMaskImage->m_size.height);
                    status = ippiMirror_8u_C1R(pMaskImage->m_ptrConst, (int)pMaskImage->m_step, pMask, _size.width, _size, ippAxsBoth);
                    if(status < 0)
                        break;
                    maskCopy = 1;
                }
                else
                {
                    status = ippiMirror_8u_C1IR(pMask, _size.width, _size, ippAxsBoth);
                    if(status < 0)
                        break;
                }
            }
            default: break;
            }

            switch(morphType)
            {
            case iwiMorphErode:     status = ippiErodeInit_L (size, pMask, pSpec->maskSize, (IppiMorphStateL*)pSpec->pIppSpec); break;
            case iwiMorphDilate:    status = ippiDilateInit_L(size, pMask, pSpec->maskSize, (IppiMorphStateL*)pSpec->pIppSpec); break;
            default:
                status = ippiMorphInit_L (size, pMask, pSpec->maskSize, dataType, channels, (IppiMorphAdvStateL*)pSpec->pIppSpec);
                if(status < 0)
                    break;
#if IPP_VERSION_COMPLEX >= 20180000
                status = ippiMorphSetMode_L(IPP_MORPH_NO_THRESHOLD, (IppiMorphAdvStateL*)pSpec->pIppSpec); break;
#endif
            }
            break;
        }

        if(pMask && maskCopy)
            ippFree(pMask);
        if(status < 0)
        {
            if(pSpec->pIppSpec)
                OWN_MEM_FREE(pSpec->pIppSpec);

            return status;
        }

        pSpec->initialized = OWN_INIT_MAGIC_NUM;
    }

    return ippStsNoErr;
}

IW_DECL(void) llwiFilterMorphology_Free(IwiFilterMorphologySpec *pSpec)
{
    if(pSpec->pIppSpec)
    {
        OWN_MEM_FREE(pSpec->pIppSpec);
        pSpec->pIppSpec = NULL;
    }

    pSpec->initialized = 0;
}
