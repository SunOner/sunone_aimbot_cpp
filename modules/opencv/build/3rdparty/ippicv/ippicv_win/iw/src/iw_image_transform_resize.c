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

typedef IppStatus (IPP_STDCALL *IppiResizeBorder_ptr)(const void* pSrc, IppSizeL srcStep, void* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const void* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer);
typedef IppStatus (IPP_STDCALL *IppiResize_ptr)(const void* pSrc, IppSizeL srcStep, void* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer);

typedef IppStatus (IPP_STDCALL *IppiResizeBorder_TL_ptr)(const void* pSrc, IppSizeL srcStep, void* pDst, IppSizeL dstStep, IppiBorderType border, const void* pBorderValue, const IppiResizeSpec_LT* pSpec, Ipp8u* pBuffer);
typedef IppStatus (IPP_STDCALL *IppiResize_TL_ptr)(const void* pSrc, IppSizeL srcStep, void* pDst, IppSizeL dstStep, const IppiResizeSpec_LT* pSpec, Ipp8u* pBuffer);


struct _IwiResizeSpec
{
#if IW_ENABLE_THREADING_LAYER
    IppiResizeSpec_LT      *pIppSpec_TL;
    IppiResize_TL_ptr       ippiResize_TL;
    IppiResizeBorder_TL_ptr ippiResizeBorder_TL;
#endif
    IppiResizeSpec         *pIppSpec;
    IwiResizeParams         auxParams;
    IppiResize_ptr          ippiResize;
    IppiResizeBorder_ptr    ippiResizeBorder;
    OwnCastArray_ptr        borderCastFun;
    IwiSize                 srcSize;
    IwiSize                 dstSize;
    int                     channels;
    IppDataType             dataType;
    IppiInterpolationType   interpolation;


    unsigned int initialized;
};

IW_DECL(IppStatus) llwiResize_InitAlloc(IwiSize srcSize, IwiSize dstSize, IppDataType dataType, int channels,
    IppiInterpolationType interpolation, const IwiResizeParams *pAuxParams, IwiBorderType border, IwiResizeSpec *pSpec);

IW_DECL(void)      llwiResize_Free(IwiResizeSpec *pSpec);

IW_DECL(IppStatus) llwiResize_ProcessWrap(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiPoint dstRoiOffset, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile, const IwiResizeSpec *pSpec);

IW_DECL(IppStatus) llwiResize_Process(const void *pSrc, IppSizeL srcStep, void *pDst, IppSizeL dstStep, IppiPointL dstRoiOffset, IppiSizeL dstRoiSize,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiResizeSpec *pSpec);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiResize
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiResize(const IwiImage *pSrcImage, IwiImage *pDstImage, IppiInterpolationType interpolation,
    const IwiResizeParams *pParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus       status;
    IwiResizeSpec   spec;
    IwiPoint        dstRoiOffset = {0, 0};

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
        status = llwiResize_InitAlloc(pSrcImage->m_size, pDstImage->m_size, pSrcImage->m_dataType, pSrcImage->m_channels, interpolation, pParams, border, &spec);
        if(status < 0)
            return status;

        status = llwiResize_ProcessWrap(pSrcImage, pDstImage, dstRoiOffset, border, pBorderVal, pTile, &spec);
        llwiResize_Free(&spec);
    }

    return status;
}

IW_DECL(IppStatus) iwiResize_Free(IwiResizeSpec *pSpec)
{
    if(!pSpec)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    llwiResize_Free(pSpec);
    OWN_MEM_FREE(pSpec);

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiResize_InitAlloc(IwiResizeSpec **ppSpec, IwiSize srcSize, IwiSize dstSize, IppDataType dataType, int channels,
    IppiInterpolationType interpolation, const IwiResizeParams *pAuxParams, IwiBorderType border)
{
    IppStatus       status;
    IwiResizeSpec   spec;

    if(!ppSpec)
        return ippStsNullPtrErr;

    status = llwiResize_InitAlloc(srcSize, dstSize, dataType, channels, interpolation, pAuxParams, border, &spec);
    if(status < 0)
        return status;

    *ppSpec = (IwiResizeSpec*)OWN_MEM_ALLOC(sizeof(IwiResizeSpec));
    if(!*ppSpec)
        return ippStsNoMemErr;
    **ppSpec = spec;

    return status;
}

IW_DECL(IppStatus) iwiResize_Process(const IwiResizeSpec *pSpec, const IwiImage *pSrcImage, IwiImage *pDstImage, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus status;
    IwiPoint  dstRoiOffset = {0, 0};

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

    return llwiResize_ProcessWrap(pSrcImage, pDstImage, dstRoiOffset, border, pBorderVal, pTile, pSpec);
}

IW_DECL(IppStatus) iwiResize_GetSrcRoi(const IwiResizeSpec *pSpec, IwiRoi dstRoi, IwiRoi *pSrcRoi)
{
    IppStatus status;

    IppiPointL  srcRoiOffset = {0, 0};
    IwiSize     srcRoiSize   = {0, 0};
    IppiPointL  dstRoiOffset;
    IwiSize     dstRoiSize;

    if(!pSpec || !pSrcRoi)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

#if IW_ENABLE_THREADING_LAYER
    if(pSpec->pIppSpec_TL)
        return ippStsNotSupportedModeErr;
#endif

    //OWN_ROI_FIT(pSpec->dstSize, dstRoi);

    dstRoiOffset.x    = dstRoi.x;
    dstRoiOffset.y    = dstRoi.y;
    dstRoiSize.width  = dstRoi.width;
    dstRoiSize.height = dstRoi.height;

    status = ippiResizeGetSrcRoi_L(pSpec->pIppSpec, dstRoiOffset, dstRoiSize, &srcRoiOffset, &srcRoiSize);
    if(status < 0)
        return status;

    pSrcRoi->x = srcRoiOffset.x;
    pSrcRoi->y = srcRoiOffset.y;
    pSrcRoi->width  = srcRoiSize.width;
    pSrcRoi->height = srcRoiSize.height;

    return status;
}

IW_DECL(IppStatus) iwiResize_GetBorderSize(const IwiResizeSpec *pSpec, IwiBorderSize *pBorderSize)
{
    IppStatus      status;
    IppiBorderSize borderSize;

    if(!pSpec || !pBorderSize)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

#if IW_ENABLE_THREADING_LAYER
    if(pSpec->pIppSpec_TL)
        status = ippiResizeGetBorderSize_LT(pSpec->pIppSpec_TL, &borderSize);
    else
#endif
        status = ippiResizeGetBorderSize_L(pSpec->pIppSpec, &borderSize);
    if(status < 0)
        return status;

    pBorderSize->left   = borderSize.borderLeft;
    pBorderSize->top    = borderSize.borderTop;
    pBorderSize->right  = borderSize.borderRight;
    pBorderSize->bottom = borderSize.borderBottom;

    return ippStsNoErr;
}


#if IW_ENABLE_iwiResize_LinearAA || IW_ENABLE_iwiResize_CubicAA || IW_ENABLE_iwiResize_LanczosAA
#define IW_ENABLE_iwiResize_Antialiasing 1
#else
#define IW_ENABLE_iwiResize_Antialiasing 0
#endif

/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiResize_ProcessWrap(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiPoint dstRoiOffset, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile, const IwiResizeSpec *pSpec)
{
    IppStatus   status;
    const void *pSrc       = pSrcImage->m_ptrConst;
    void       *pDst       = pDstImage->m_ptr;
    IwiSize     dstRoiSize = pDstImage->m_size;

#if IW_ENABLE_THREADING_LAYER
    if(pSpec->pIppSpec_TL)
        pTile = NULL;
#endif
    if(pTile && pTile->m_initialized != ownTileInitNone)
    {
        IwiImage srcSubImage = *pSrcImage;
        IwiImage dstSubImage = *pDstImage;

        if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
            return ippStsNotSupportedModeErr;

        if(pTile->m_initialized == ownTileInitSimple)
        {
            IwiRoi    dstRoi = pTile->m_dstRoi;
            IwiRoi    srcRoi;

            if(!owniTile_BoundToSize(&dstRoi, &dstRoiSize))
                return ippStsNoOperation;

            status = iwiResize_GetSrcRoi(pSpec, dstRoi, &srcRoi);
            if(status < 0)
                return status;

            iwiImage_RoiSet(&srcSubImage, srcRoi);
            iwiImage_RoiSet(&dstSubImage, dstRoi);

            dstRoiOffset.x = dstRoi.x;
            dstRoiOffset.y = dstRoi.y;
        }
        else if(pTile->m_initialized == ownTileInitPipe)
        {
            return ippStsNotSupportedModeErr;
        }
        else
            return ippStsContextMatchErr;

        return llwiResize_ProcessWrap(&srcSubImage, &dstSubImage, dstRoiOffset, border, pBorderVal, NULL, pSpec);
    }

    status = llwiResize_Process(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step, dstRoiOffset, dstRoiSize, border, pBorderVal, pSpec);
    return status;
}

IW_DECL(IppStatus) llwiResize_Process(const void *pSrc, IppSizeL srcStep, void *pDst, IppSizeL dstStep, IppiPointL dstRoiOffset, IppiSizeL dstRoiSize,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiResizeSpec *pSpec)
{
    IppStatus status;
    Ipp64f    borderVal[4];

    Ipp8u   *pTmpBuffer    = 0;
    IwSize   tmpBufferSize = 0;

    if(!pSpec)
        return ippStsNullPtrErr;
    if(pSpec->initialized != OWN_INIT_MAGIC_NUM)
        return ippStsContextMatchErr;

    for(;;)
    {
#if IW_ENABLE_THREADING_LAYER
        if(pSpec->pIppSpec_TL && (pSpec->ippiResize_TL || pSpec->ippiResizeBorder_TL))
        {
            status = ippiResizeGetBufferSize_LT(pSpec->pIppSpec_TL, &tmpBufferSize);
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

            if(pSpec->ippiResize)
                status = pSpec->ippiResize_TL(pSrc, srcStep, pDst, dstStep, pSpec->pIppSpec_TL, pTmpBuffer);
            else
                status = pSpec->ippiResizeBorder_TL(pSrc, srcStep, pDst, dstStep, border, borderVal, pSpec->pIppSpec_TL, pTmpBuffer);
            break;
        }
        else
#endif
        {
            status = ippiResizeGetBufferSize_L(pSpec->pIppSpec, dstRoiSize, pSpec->channels, &tmpBufferSize);
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

            if(pSpec->ippiResize)
                status = pSpec->ippiResize(pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec->pIppSpec, pTmpBuffer);
            else
                status = pSpec->ippiResizeBorder(pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, border, borderVal, pSpec->pIppSpec, pTmpBuffer);
            break;
        }
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}

IW_DECL(IppStatus) llwiResize_InitAlloc(IwiSize srcSize, IwiSize dstSize, IppDataType dataType, int channels,
    IppiInterpolationType interpolation, const IwiResizeParams *pAuxParams, IwiBorderType border, IwiResizeSpec *pSpec)
{
    if(!srcSize.width || !srcSize.height ||
        !dstSize.width || !dstSize.height)
        return ippStsNoOperation;

    OWN_MEM_RESET(pSpec);
    (void)border;

    if(pAuxParams)
        pSpec->auxParams = *pAuxParams;
    else
        iwiResize_SetDefaultParams(&pSpec->auxParams);

    if(pSpec->interpolation == ippNearest ||
        pSpec->interpolation == ippSuper)
        pSpec->auxParams.antialiasing = 0;


#if IW_ENABLE_iwiResize_Antialiasing
    if(pSpec->auxParams.antialiasing)
    {
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_8u_C4R_L; break;
#endif
            default:    return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_16u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_16u_C4R_L; break;
#endif
            default: return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_16s_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_16s_C4R_L; break;
#endif
            default: return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeAntialiasing_32f_C4R_L; break;
#endif
            default: return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
    }
    else
#endif
    {
        switch(interpolation)
        {
#if IW_ENABLE_iwiResize_Nearest
        case ippNearest:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_8u_C4R_L; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_16u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_16u_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_16s_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_16s_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize = (IppiResize_ptr)ippiResizeNearest_32f_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Linear
        case ippLinear:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_8u_C4R_L; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_16u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_16u_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_16s_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_16s_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_32f_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_64F
            case ipp64f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_64f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_64f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLinear_64f_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Cubic
        case ippCubic:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_8u_C4R_L; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_16u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_16u_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_16s_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_16s_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeCubic_32f_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Lanczos
        case ippLanczos:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_8u_C4R_L; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_16u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_16u_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_16s_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_16s_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder = (IppiResizeBorder_ptr)ippiResizeLanczos_32f_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Super
        case ippSuper:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_8u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_8u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_8u_C4R_L; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_16u_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_16u_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_16u_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_16s_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_16s_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_16s_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_32f_C1R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_32f_C3R_L; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize = (IppiResize_ptr)ippiResizeSuper_32f_C4R_L; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
        default: return ippStsInterpolationErr;
        }
    }

#if IW_ENABLE_THREADING_LAYER
#if IW_ENABLE_iwiResize_Antialiasing
    if(pSpec->auxParams.antialiasing)
    {
        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_8u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_8u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_8u_C4R_LT; break;
#endif
            default:    return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
        case ipp16u:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_16u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_16u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_16u_C4R_LT; break;
#endif
            default: return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
        case ipp16s:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_16s_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_16s_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_16s_C4R_LT; break;
#endif
            default: return ippStsNumChannelsErr;
            }
            break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
        case ipp32f:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_32f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
            case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_32f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
            case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeAntialiasing_32f_C4R_LT; break;
#endif
            default: return ippStsNumChannelsErr;
            }
            break;
#endif
        default: return ippStsDataTypeErr;
        }
    }
    else
#endif
    {
        switch(interpolation)
        {
#if IW_ENABLE_iwiResize_Nearest
        case ippNearest:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_8u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_8u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_8u_C4R_LT; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_16u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_16u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_16u_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_16s_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_16s_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_16s_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_32f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_32f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeNearest_32f_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Linear
        case ippLinear:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_8u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_8u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_8u_C4R_LT; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_16u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_16u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_16u_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_16s_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_16s_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_16s_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_32f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_32f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_32f_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_64F
            case ipp64f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_64f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_64f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLinear_64f_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Cubic
        case ippCubic:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_8u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_8u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_8u_C4R_LT; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_16u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_16u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_16u_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_16s_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_16s_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_16s_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_32f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_32f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeCubic_32f_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Lanczos
        case ippLanczos:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_8u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_8u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_8u_C4R_LT; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_16u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_16u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_16u_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_16s_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_16s_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_16s_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_32f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_32f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResizeBorder_TL = (IppiResizeBorder_TL_ptr)ippiResizeLanczos_32f_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
#if IW_ENABLE_iwiResize_Super
        case ippSuper:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:     pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_8u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:     pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_8u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:     pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_8u_C4R_LT; break;
#endif
                default:    return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1:  pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_16u_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3:  pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_16u_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4:  pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_16u_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_16s_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_16s_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_16s_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f:
                switch(channels)
                {
#if IW_ENABLE_CHANNELS_C1
                case 1: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_32f_C1R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C3
                case 3: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_32f_C3R_LT; break;
#endif
#if IW_ENABLE_CHANNELS_C4
                case 4: pSpec->ippiResize_TL = (IppiResize_TL_ptr)ippiResizeSuper_32f_C4R_LT; break;
#endif
                default: return ippStsNumChannelsErr;
                }
                break;
#endif
            default: return ippStsDataTypeErr;
            }
            break;
#endif
        default: return ippStsInterpolationErr;
        }
    }
#endif
    if(!pSpec->ippiResizeBorder && !pSpec->ippiResize)
        return ippStsBadArgErr;

    switch(dataType)
    {
    case ipp8u:     pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f8u;    break;
    case ipp16u:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f16u;   break;
    case ipp16s:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f16s;   break;
    case ipp32f:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f32f;   break;
    case ipp64f:    pSpec->borderCastFun = (OwnCastArray_ptr)ownCastArray_64f64f;   break;
    default:        return ippStsDataTypeErr;
    }
    if(!pSpec->borderCastFun)
        return ippStsBadArgErr;

    pSpec->srcSize       = srcSize;
    pSpec->dstSize       = dstSize;
    pSpec->channels      = channels;
    pSpec->dataType      = dataType;
    pSpec->interpolation = interpolation;

    {
        IppStatus status;
        IwSize    specSize = 0;
        Ipp8u    *pInitBuf = NULL;
        IwSize    initSize = 0;

        for(;;)
        {
#if IW_ENABLE_THREADING_LAYER
            if(iwGetThreadsNum() > 1)
            {
                if(pSpec->pIppSpec_TL)
                    return ippStsContextMatchErr;

                status = ippiResizeGetSize_LT(pSpec->srcSize, pSpec->dstSize, pSpec->dataType, pSpec->interpolation, pSpec->auxParams.antialiasing, &specSize, &initSize);
                if(status < 0)
                    break;

                pSpec->pIppSpec_TL = (IppiResizeSpec_LT*)OWN_MEM_ALLOC(specSize);
                if(!pSpec->pIppSpec_TL)
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

                if(pSpec->auxParams.antialiasing)
                {
                    switch(interpolation)
                    {
#if IW_ENABLE_iwiResize_LinearAA
                    case ippLinear:  status = ippiResizeAntialiasingLinearInit_LT(srcSize, dstSize, dataType, channels, pSpec->pIppSpec_TL, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_CubicAA
                    case ippCubic:   status = ippiResizeAntialiasingCubicInit_LT(srcSize, dstSize, dataType, channels, pSpec->auxParams.cubicBVal, pSpec->auxParams.cubicCVal, pSpec->pIppSpec_TL, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_LanczosAA
                    case ippLanczos: status = ippiResizeAntialiasingLanczosInit_LT(srcSize, dstSize, dataType, channels, pSpec->auxParams.lanczosLobes, pSpec->pIppSpec_TL, pInitBuf); break;
#endif
                    default:         status = ippStsInterpolationErr; break;
                    }
                }
                else
                {
                    switch(interpolation)
                    {
#if IW_ENABLE_iwiResize_Nearest
                    case ippNearest: status = ippiResizeNearestInit_LT(srcSize, dstSize, dataType, channels, pSpec->pIppSpec_TL); break;
#endif
#if IW_ENABLE_iwiResize_Linear
                    case ippLinear:  status = ippiResizeLinearInit_LT(srcSize, dstSize, dataType, channels, pSpec->pIppSpec_TL); break;
#endif
#if IW_ENABLE_iwiResize_Cubic
                    case ippCubic:   status = ippiResizeCubicInit_LT(srcSize, dstSize, dataType, channels, pSpec->auxParams.cubicBVal, pSpec->auxParams.cubicCVal, pSpec->pIppSpec_TL, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_Lanczos
                    case ippLanczos: status = ippiResizeLanczosInit_LT(srcSize, dstSize, dataType, channels, pSpec->auxParams.lanczosLobes, pSpec->pIppSpec_TL, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_Super
                    case ippSuper:   status = ippiResizeSuperInit_LT(srcSize, dstSize, dataType, channels, pSpec->pIppSpec_TL); break;
#endif
                    default:         status = ippStsInterpolationErr; break;
                    }
                }
                break;
            }
            else
#endif
            {
                if(pSpec->pIppSpec)
                    return ippStsContextMatchErr;

                status = ippiResizeGetSize_L(srcSize, dstSize, dataType, interpolation, pSpec->auxParams.antialiasing, &specSize, &initSize);
                if(status < 0)
                    break;

                pSpec->pIppSpec = (IppiResizeSpec*)OWN_MEM_ALLOC(specSize);
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

                if(pSpec->auxParams.antialiasing)
                {
                    switch(interpolation)
                    {
#if IW_ENABLE_iwiResize_LinearAA
                    case ippLinear:  status = ippiResizeAntialiasingLinearInit_L(srcSize, dstSize, dataType, pSpec->pIppSpec, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_CubicAA
                    case ippCubic:   status = ippiResizeAntialiasingCubicInit_L(srcSize, dstSize, dataType, pSpec->auxParams.cubicBVal, pSpec->auxParams.cubicCVal, pSpec->pIppSpec, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_LanczosAA
                    case ippLanczos: status = ippiResizeAntialiasingLanczosInit_L(srcSize, dstSize, dataType, pSpec->auxParams.lanczosLobes, pSpec->pIppSpec, pInitBuf); break;
#endif
                    default:         status = ippStsInterpolationErr; break;
                    }
                }
                else
                {
                    switch(interpolation)
                    {
#if IW_ENABLE_iwiResize_Nearest
                    case ippNearest: status = ippiResizeNearestInit_L(srcSize, dstSize, dataType, pSpec->pIppSpec); break;
#endif
#if IW_ENABLE_iwiResize_Linear
                    case ippLinear:  status = ippiResizeLinearInit_L(srcSize, dstSize, dataType, pSpec->pIppSpec); break;
#endif
#if IW_ENABLE_iwiResize_Cubic
                    case ippCubic:   status = ippiResizeCubicInit_L(srcSize, dstSize, dataType, pSpec->auxParams.cubicBVal, pSpec->auxParams.cubicCVal, pSpec->pIppSpec, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_Lanczos
                    case ippLanczos: status = ippiResizeLanczosInit_L(srcSize, dstSize, dataType, pSpec->auxParams.lanczosLobes, pSpec->pIppSpec, pInitBuf); break;
#endif
#if IW_ENABLE_iwiResize_Super
                    case ippSuper:   status = ippiResizeSuperInit_L(srcSize, dstSize, dataType, pSpec->pIppSpec); break;
#endif
                    default:         status = ippStsInterpolationErr; break;
                    }
                }
                break;
            }
        }

        if(pInitBuf)
            OWN_MEM_FREE(pInitBuf);
        if(status < 0)
        {
            if(pSpec->pIppSpec)
                OWN_MEM_FREE(pSpec->pIppSpec);
#if IW_ENABLE_THREADING_LAYER
            if(pSpec->pIppSpec_TL)
                OWN_MEM_FREE(pSpec->pIppSpec_TL);
#endif

            return status;
        }

        pSpec->initialized = OWN_INIT_MAGIC_NUM;
    }

    return ippStsNoErr;
}

IW_DECL(void) llwiResize_Free(IwiResizeSpec *pSpec)
{
    if(pSpec->pIppSpec)
    {
        OWN_MEM_FREE(pSpec->pIppSpec);
        pSpec->pIppSpec = 0;
    }
#if IW_ENABLE_THREADING_LAYER
    if(pSpec->pIppSpec_TL)
    {
        OWN_MEM_FREE(pSpec->pIppSpec_TL);
        pSpec->pIppSpec_TL = 0;
    }
#endif

    pSpec->initialized = 0;
}
