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

typedef IppStatus (IPP_STDCALL *IppiFilterGaussian_ptr)(const void* pSrc, IppSizeL srcStep, void* pDst, IppSizeL dstStep,
    IppiSizeL roiSize, IppiBorderType borderType, const void* borderValue, IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer);

typedef struct _IwiFilterGaussianSpecTls
{
    IwiTile         pipe[3];
    IwiImage        inter_split[4];
    IwiImage        inter_proc[4];
    IwiImage       *pInter_split[4];
    IwiImage       *pInter_proc[4];
    IwiSize         size;
    IwiSize         tileSize;
    IwiBorderType   border;

} IwiFilterGaussianSpecTls;

static void IPP_STDCALL tlsDescturctor(void* pParams)
{
    IwiFilterGaussianSpecTls *pSpec = (IwiFilterGaussianSpecTls*)pParams;
    int i;

    iwiTilePipeline_Release(&pSpec->pipe[0]);
    for(i = 0; i < 4; i++)
    {
        if(pSpec->pInter_split[i])
        {
            iwiImage_Release(&pSpec->inter_split[i]);
            iwiImage_Release(&pSpec->inter_proc[i]);
        }
    }
    ippFree(pSpec);
}

struct _IwiFilterGaussianSpec
{
    // Init params
    IppFilterGaussianSpec  *pIppSpec;
    IwiFilterGaussianParams auxParams;
    IppiFilterGaussian_ptr  ippiFilterGaussian;
    OwnCastArray_ptr        borderCastFun;
    IwiSize                 size;
    IppDataType             dataType;
    int                     channels;
    int                     kernelSize;

    // Extended wrappers
    int   extended;
    int   origChannels;
    IwTls tls;

    unsigned int initialized;
};


IW_DECL(IppStatus) llwiFilterGaussian_InitAlloc(IwiSize size, IppDataType dataType, int channels, int kernelSize, float sigma,
    const IwiFilterGaussianParams *pAuxParams, IwiBorderType border, IwiFilterGaussianSpec *pSpec);

IW_DECL(void)      llwiFilterGaussian_Free(IwiFilterGaussianSpec *pSpec);

IW_DECL(IppStatus) llwiFilterGaussian_ProcessWrap(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile, const IwiFilterGaussianSpec *pSpec);

IW_DECL(IppStatus) llwiFilterGaussian_Process(const void *pSrc, IppSizeL srcStep, void *pDst, IppSizeL dstStep, IppiSizeL roi,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiFilterGaussianSpec *pSpec);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterGaussian
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterGaussian(const IwiImage *pSrcImage, IwiImage *pDstImage, int kernelSize, double sigma, const IwiFilterGaussianParams *pAuxParams,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus               status;
    IwiFilterGaussianSpec   spec;

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

    {
        IwiSize size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        status = llwiFilterGaussian_InitAlloc(size, pSrcImage->m_dataType, pSrcImage->m_channels, kernelSize, (float)sigma, pAuxParams, border, &spec);
        if(status < 0)
            return status;

        status = llwiFilterGaussian_ProcessWrap(pSrcImage, pDstImage, border, pBorderVal, pTile, &spec);
        llwiFilterGaussian_Free(&spec);
    }

    return status;
}

IW_DECL(IppStatus) iwiFilterGaussian_InitAlloc(IwiFilterGaussianSpec **ppSpec, IwiSize size, IppDataType dataType, int channels, int kernelSize, double sigma,
    const IwiFilterGaussianParams *pAuxParams, IwiBorderType border)
{
    IppStatus               status;
    IwiFilterGaussianSpec   spec;

    if(!ppSpec)
        return ippStsNullPtrErr;

    status = llwiFilterGaussian_InitAlloc(size, dataType, channels, kernelSize, (float)sigma, pAuxParams, border, &spec);
    if(status < 0)
        return status;

    *ppSpec = (IwiFilterGaussianSpec*)OWN_MEM_ALLOC(sizeof(IwiFilterGaussianSpec));
    if(!*ppSpec)
        return ippStsNoMemErr;
    **ppSpec = spec;

    return status;
}

IW_DECL(IppStatus) iwiFilterGaussian_Free(IwiFilterGaussianSpec *pSpec)
{
    if(!pSpec)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    llwiFilterGaussian_Free(pSpec);
    OWN_MEM_FREE(pSpec);

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiFilterGaussian_Process(const IwiFilterGaussianSpec *pSpec, const IwiImage *pSrcImage, IwiImage *pDstImage, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
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

    if(pSpec->extended)
    {
        if(pSpec->dataType != pSrcImage->m_dataType ||
            pSpec->origChannels != pSrcImage->m_channels)
            return ippStsBadArgErr;

        if(pSpec->dataType != pDstImage->m_dataType ||
            pSpec->origChannels != pDstImage->m_channels)
            return ippStsBadArgErr;
    }
    else
    {
        if(pSpec->dataType != pSrcImage->m_dataType ||
            pSpec->channels != pSrcImage->m_channels)
            return ippStsBadArgErr;

        if(pSpec->dataType != pDstImage->m_dataType ||
            pSpec->channels != pDstImage->m_channels)
            return ippStsBadArgErr;
    }

    return llwiFilterGaussian_ProcessWrap(pSrcImage, pDstImage, border, pBorderVal, pTile, pSpec);
}

/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterGaussian_ProcessWrap(const IwiImage *pSrcImage, IwiImage *pDstImage,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile, const IwiFilterGaussianSpec *pSpec)
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
            IwiBorderSize  borderSize = iwiSizeSymToBorderSize(pSpec->kernelSize);

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

        return llwiFilterGaussian_ProcessWrap(&srcSubImage, &dstSubImage, border, pBorderVal, NULL, pSpec);
    }

    if(pSpec->extended && pSrcImage->m_channels != 1)
    {
        IwiRoi  roi;
        int     ch;
        IwiFilterGaussianSpecTls *pTls = (IwiFilterGaussianSpecTls*)iwTls_Get(&pSpec->tls);
        IwiBorderSize borderSize = iwiSizeSymToBorderSize(pSpec->kernelSize);

        // Init for the current thread
        if(pTls && (pTls->size.width < size.width || pTls->size.height < size.height))
        {
            status = iwTls_Set((IwTls*)&pSpec->tls, NULL);
            if(status < 0)
                return status;
            pTls = NULL;
        }
        if(!pTls)
        {
            IwiSize         splitSize;
            IwiSize         splitProcSize;
            int             activeCh   = 0;

            pTls = (IwiFilterGaussianSpecTls*)ippMalloc_L(sizeof(IwiFilterGaussianSpecTls));
            if(!pTls)
                return ippStsMemAllocErr;
            ippsZero_8u((Ipp8u*)pTls, sizeof(IwiFilterGaussianSpecTls));
            status = iwTls_Set((IwTls*)&pSpec->tls, pTls);
            if(status < 0)
                return status;

            pTls->size = size;
            pTls->pInter_proc[0]  = pTls->pInter_proc[1]  = pTls->pInter_proc[2]  = pTls->pInter_proc[3]  = NULL;
            pTls->pInter_split[0] = pTls->pInter_split[1] = pTls->pInter_split[2] = pTls->pInter_split[3] = NULL;

            for(ch = 0; ch < pSpec->origChannels; ch++)
            {
                if(!pSpec->auxParams.chDesc || OWN_DESC_CHECK_MASK(pSpec->auxParams.chDesc, ch))
                {
                    pTls->pInter_split[ch]  = &pTls->inter_split[ch];
                    pTls->pInter_proc[ch]   = &pTls->inter_proc[ch];

                    iwiImage_Init(&pTls->inter_split[ch]);
                    iwiImage_Init(&pTls->inter_proc[ch]);
                    activeCh++;
                }
                else if(OWN_DESC_CHECK_REPL(pSpec->auxParams.chDesc, ch))
                    pTls->pInter_proc[ch] = pTls->pInter_proc[OWN_DESC_GET_REPL_CH(pSpec->auxParams.chDesc)];
            }

            pTls->tileSize = owniSuggestTileSize_k1(pSrcImage, pSpec->kernelSize, 2+(2*activeCh/4.));

            iwiTilePipeline_Init(&pTls->pipe[2], pTls->tileSize, size, NULL, NULL, NULL);
            iwiTilePipeline_InitChild(&pTls->pipe[1], &pTls->pipe[2], &border, &borderSize, NULL);
            iwiTilePipeline_InitChild(&pTls->pipe[0], &pTls->pipe[1], NULL, NULL, NULL);

            iwiTilePipeline_GetDstBufferSize(&pTls->pipe[0], &splitSize);
            iwiTilePipeline_GetDstBufferSize(&pTls->pipe[1], &splitProcSize);

            for(ch = 0; ch < pSpec->origChannels; ch++)
            {
                if(pTls->pInter_split[ch])
                {
                    iwiImage_Alloc(pTls->pInter_split[ch], splitSize, pSrcImage->m_dataType, 1, NULL);
                    iwiImage_Alloc(pTls->pInter_proc[ch], splitProcSize, pSrcImage->m_dataType, 1, NULL);
                }
            }
        }

        if(border != pTls->border)
        {
            iwiTilePipeline_Init(&pTls->pipe[2], pTls->tileSize, size, NULL, NULL, NULL);
            iwiTilePipeline_InitChild(&pTls->pipe[1], &pTls->pipe[2], &border, &borderSize, NULL);
            iwiTilePipeline_InitChild(&pTls->pipe[0], &pTls->pipe[1], NULL, NULL, NULL);
            pTls->border = border;
        }

        status     = ippStsNoErr;
        roi.width  = pTls->tileSize.width;
        roi.height = pTls->tileSize.height;
        for(roi.y = 0; roi.y < size.height; roi.y += roi.height)
        {
            for(roi.x = 0; roi.x < size.width; roi.x += roi.width)
            {
                iwiTilePipeline_SetRoi(&pTls->pipe[2], roi);

                status = iwiSplitChannels(pSrcImage, pTls->pInter_split, NULL, &pTls->pipe[0]);
                if(status < 0)
                    return status;

                for(ch = 0; ch < pSrcImage->m_channels; ch++)
                {
                    if(pTls->pInter_split[ch])
                    {
                        status = llwiFilterGaussian_ProcessWrap(pTls->pInter_split[ch], pTls->pInter_proc[ch], border, pBorderVal, &pTls->pipe[1], pSpec);
                        if(status < 0)
                            return status;
                    }
                }

                status = iwiMergeChannels((const IwiImage* const*)pTls->pInter_proc, pDstImage, NULL, &pTls->pipe[2]);
                if(status < 0)
                    return status;
            }
        }
    }
    else
    {
        status = llwiFilterGaussian_Process(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step, size, border, pBorderVal, pSpec);
    }
    return status;
}

IW_DECL(IppStatus) llwiFilterGaussian_Process(const void *pSrc, IppSizeL srcStep, void *pDst, IppSizeL dstStep, IppiSizeL roi,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiFilterGaussianSpec *pSpec)
{
    IppStatus status;
    Ipp64f    borderVal[4];

    Ipp8u   *pTmpBuffer    = 0;
    IwSize   tmpBufferSize = 0;

    if(!pSpec || !pSpec->ippiFilterGaussian)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    for(;;)
    {
        status = ippiFilterGaussianGetBufferSize_L(roi, pSpec->kernelSize, pSpec->dataType, border, pSpec->channels, &tmpBufferSize);
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

        status = pSpec->ippiFilterGaussian(pSrc, srcStep, pDst, dstStep, roi, border, borderVal, pSpec->pIppSpec, pTmpBuffer);
        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}

IW_DECL(IppStatus) llwiFilterGaussian_InitAlloc(IwiSize size, IppDataType dataType, int channels, int kernelSize, float sigma,
    const IwiFilterGaussianParams *pAuxParams, IwiBorderType border, IwiFilterGaussianSpec *pSpec)
{
    OwniChCodes chCode;

    OWN_MEM_RESET(pSpec);

    if(pAuxParams)
        pSpec->auxParams = *pAuxParams;
    else
        iwiFilterGaussian_SetDefaultParams(&pSpec->auxParams);

    if(channels == 4)
    {
        pSpec->origChannels = channels;
        pSpec->extended     = 1;
        channels            = 1;
    }

    chCode = owniChDescriptorToCode(pSpec->auxParams.chDesc, channels, channels);

    switch(dataType)
    {
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:
        pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f8u;
        switch(chCode)
        {
#if IW_ENABLE_CHANNELS_C1
        case owniC1:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
        case owniC3:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_8u_C3R_L; break;
#endif
        default:        return ippStsNumChannelsErr;
        }
        break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u:
        pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f16u;
        switch(chCode)
        {
#if IW_ENABLE_CHANNELS_C1
        case owniC1:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
        case owniC3:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_16u_C3R_L; break;
#endif
        default:        return ippStsNumChannelsErr;
        }
        break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s:
        pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f16s;
        switch(chCode)
        {
#if IW_ENABLE_CHANNELS_C1
        case owniC1:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
        case owniC3:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_16s_C3R_L; break;
#endif
        default:        return ippStsNumChannelsErr;
        }
        break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f:
        pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f32f;
        switch(chCode)
        {
#if IW_ENABLE_CHANNELS_C1
        case owniC1:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
        case owniC3:    pSpec->ippiFilterGaussian = (IppiFilterGaussian_ptr)ippiFilterGaussian_32f_C3R_L; break;
#endif
        default:        return ippStsNumChannelsErr;
        }
        break;
#endif
    default: return ippStsDataTypeErr;
    }
    if(!pSpec->borderCastFun || !pSpec->ippiFilterGaussian)
        return ippStsBadArgErr;

    pSpec->size        = size;
    pSpec->channels    = channels;
    pSpec->dataType    = dataType;
    pSpec->kernelSize  = kernelSize;

    {
        IppStatus status;
        IwSize    specSize = 0;
        Ipp8u    *pInitBuf = NULL;
        IwSize    initSize = 0;

        if(pSpec->pIppSpec)
            return ippStsContextMatchErr;

        for(;;)
        {
            // Initialize Intel IPP functions and check parameters
            status = ippiFilterGaussianGetSpecSize_L(pSpec->kernelSize, pSpec->dataType, pSpec->channels, &specSize, &initSize);
            if(status < 0)
                break;

            pSpec->pIppSpec = (IppFilterGaussianSpec*)OWN_MEM_ALLOC(specSize);
            if(!pSpec->pIppSpec)
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

            status = ippiFilterGaussianInit_L(pSpec->size, pSpec->kernelSize, sigma, border, pSpec->dataType, pSpec->channels, pSpec->pIppSpec, pInitBuf);
            if(status < 0)
                break;

            if(pSpec->extended)
            {
                status = iwTls_Init(&pSpec->tls, tlsDescturctor);
                if(status < 0)
                    break;
            }

            break;
        }

        if(pInitBuf)
            OWN_MEM_FREE(pInitBuf);
        if(status < 0)
        {
            if(pSpec->pIppSpec)
                OWN_MEM_FREE(pSpec->pIppSpec);

            return status;
        }
    }

    pSpec->initialized = OWN_INIT_MAGIC_NUM;
    return ippStsNoErr;
}

IW_DECL(void) llwiFilterGaussian_Free(IwiFilterGaussianSpec *pSpec)
{
    if(pSpec->pIppSpec)
    {
        OWN_MEM_FREE(pSpec->pIppSpec);
        pSpec->pIppSpec = NULL;
    }

    if(pSpec->extended)
        iwTls_Release(&pSpec->tls);

    pSpec->initialized = 0;
}
