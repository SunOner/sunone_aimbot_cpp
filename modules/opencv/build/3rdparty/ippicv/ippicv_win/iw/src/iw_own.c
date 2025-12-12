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

#include "iw_own.h"

#if defined _WIN32
#include <windows.h>
#include <intrin.h>
#else
#if IW_ENABLE_TLS
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1 /* for PTHREAD_MUTEX_RECURSIVE */
#endif
#ifndef __USE_UNIX98
#define __USE_UNIX98 1 /* for PTHREAD_MUTEX_RECURSIVE, on SLES11.1 with gcc 4.3.4 wherein pthread.h missing dependency on __USE_XOPEN2K8 */
#endif
#include <pthread.h>
#endif
#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#endif

#if IW_ENABLE_THREADING_LAYER
#include "omp.h"
#endif

#define OWN_ENABLE_BUFFER_POOL 0

/* /////////////////////////////////////////////////////////////////////////////
//                   Global initialization state
///////////////////////////////////////////////////////////////////////////// */
#define OWN_STATE_NOT_INITIALIZED 0
#define OWN_STATE_INITIALIZATION  1
#define OWN_STATE_INITIALIZED     2

static int* ownGlobalInitState(void)
{
    static int state = 0;
    return &state;
}
static int ownGlobalGetInitState(void)
{
    return *ownGlobalInitState();
}
static void ownGlobalSetInitState(int state)
{
    *ownGlobalInitState() = state;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   ownAlignedMalloc
///////////////////////////////////////////////////////////////////////////// */
void* IPP_STDCALL ownAlignedMalloc(size_t iSize, size_t iAlign)
{
#if defined _WIN32
    return _aligned_malloc(iSize, iAlign);
#elif defined __APPLE__
    if(iAlign <= 1)
        return malloc(iSize);
    else
    {
        void *pBuffer  = malloc(iSize + (iAlign - 1) + sizeof(void*));
        char *pABuffer = ((char*)pBuffer) + sizeof(void*);

        pABuffer += (iAlign - (((size_t)pABuffer) & (iAlign - 1)));

        ((void**)pABuffer)[-1] = pBuffer;
        return pABuffer;
    }
#else
    return memalign(iAlign, iSize);
#endif
}
void IPP_STDCALL ownAlignedFree(void* pBuffer)
{
#if defined _WIN32
    _aligned_free(pBuffer);
#elif defined __APPLE__
    free(((void**)pBuffer)[-1]);
#else
    free(pBuffer);
#endif
}

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnMutex
///////////////////////////////////////////////////////////////////////////// */
#if IW_ENABLE_TLS
typedef struct _OwnMutex
{
#if defined _WIN32
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t  mutex;
#endif
} OwnMutex;

static int ownMutex_Init(OwnMutex *pMutex)
{
#if defined _WIN32
#if _WIN32_WINNT >= 0x0600
    if(InitializeCriticalSectionEx(&pMutex->mutex, 0, 0) == 0)
        return OWN_STATUS_FAIL;
    return OWN_STATUS_OK;
#else
    __try
    {
        InitializeCriticalSection(&pMutex->mutex);
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
        return OWN_STATUS_FAIL;
    }
    return OWN_STATUS_OK;
#endif
#else
    pthread_mutexattr_t mutAttib;
    if(pthread_mutexattr_init(&mutAttib) != 0)
        return OWN_STATUS_FAIL;
    if(pthread_mutexattr_settype(&mutAttib, PTHREAD_MUTEX_RECURSIVE) != 0)
    {
        pthread_mutexattr_destroy(&mutAttib);
        return OWN_STATUS_FAIL;
    }
    if(pthread_mutex_init(&pMutex->mutex, &mutAttib) != 0)
    {
        pthread_mutexattr_destroy(&mutAttib);
        return OWN_STATUS_FAIL;
    }
    if(pthread_mutexattr_destroy(&mutAttib) != 0)
    {
        pthread_mutex_destroy(&pMutex->mutex);
        return OWN_STATUS_FAIL;
    }
    return OWN_STATUS_OK;
#endif
}

static int ownMutex_Release(OwnMutex *pMutex)
{
#if defined _WIN32
    __try
    {
        DeleteCriticalSection(&pMutex->mutex);
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
        return OWN_STATUS_FAIL;
    }
    return OWN_STATUS_OK;
#else
    if(pthread_mutex_destroy(&pMutex->mutex) != 0)
        return OWN_STATUS_FAIL;
    return OWN_STATUS_OK;
#endif
}

static int ownMutex_Lock(OwnMutex *pMutex)
{
#if defined _WIN32
    __try
    {
        EnterCriticalSection(&pMutex->mutex);
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
        return OWN_STATUS_FAIL;
    }
    return OWN_STATUS_OK;
#else
    if(pthread_mutex_lock(&pMutex->mutex) != 0)
        return OWN_STATUS_FAIL;
    return OWN_STATUS_OK;
#endif
}

static int ownMutex_Unlock(OwnMutex *pMutex)
{
#if defined _WIN32
    __try
    {
        LeaveCriticalSection(&pMutex->mutex);
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
        return OWN_STATUS_FAIL;
    }
    return OWN_STATUS_OK;
#else
    if(pthread_mutex_unlock(&pMutex->mutex) != 0)
        return OWN_STATUS_FAIL;
    return OWN_STATUS_OK;
#endif
}
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnVector - C Vector
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(void) ownVector_Reserve(OwnVector *pVector, size_t reserveSize)
{
    if(reserveSize*pVector->m_elemSize > pVector->m_bufferLen)
    {
        if(pVector->m_pBuffer)
        {
            size_t newBufferLen = reserveSize*pVector->m_elemSize;
            Ipp8u *pNewBuffer   = (Ipp8u*)OWN_SAFE_MALLOC(newBufferLen);

            OWN_SAFE_COPY(pVector->m_pBuffer, pNewBuffer, pVector->m_bufferLen);
            OWN_SAFE_RESET(pNewBuffer + pVector->m_bufferLen, (newBufferLen-pVector->m_bufferLen));
            OWN_SAFE_FREE(pVector->m_pBuffer);

            pVector->m_pBuffer   = pNewBuffer;
            pVector->m_bufferLen = newBufferLen;
        }
        else
        {
            pVector->m_bufferLen = reserveSize*pVector->m_elemSize;
            pVector->m_pBuffer   = (Ipp8u*)OWN_SAFE_MALLOC(pVector->m_bufferLen);
            if(!pVector->m_pBuffer)
            {
                pVector->m_pBuffer   = NULL;
                pVector->m_bufferLen = 0;
                return;
            }
            OWN_SAFE_RESET(pVector->m_pBuffer, pVector->m_bufferLen);
        }
    }
}

IW_DECL(void) ownVector_Init(OwnVector *pVector, size_t elemSize, size_t reserve)
{
    OWN_SAFE_RESET(pVector, sizeof(*pVector));

    pVector->m_elemSize = elemSize;
    ownVector_Reserve(pVector, reserve);
}

IW_DECL(void) ownVector_Release(OwnVector *pVector)
{
    if(pVector->m_pBuffer)
    {
        OWN_SAFE_FREE(pVector->m_pBuffer);
        pVector->m_pBuffer = 0;
    }
    pVector->m_elemSize = pVector->m_size = pVector->m_bufferLen = 0;
}

IW_DECL(void) ownVector_Resize(OwnVector *pVector, size_t newSize)
{
    ownVector_Reserve(pVector, newSize);
    pVector->m_size = newSize;
}

IW_DECL(void) ownVector_PushBack(OwnVector *pVector, void *pData)
{
    ownVector_Resize(pVector, pVector->m_size + 1);
    OWN_SAFE_COPY(pData, pVector->m_pBuffer + pVector->m_elemSize*(pVector->m_size-1), pVector->m_elemSize);
}

IW_DECL(void) ownVector_PopBack(OwnVector *pVector, void *pData)
{
    if(pVector->m_size && pData)
    {
        OWN_SAFE_COPY(pVector->m_pBuffer + pVector->m_elemSize*(pVector->m_size-1), pData, pVector->m_elemSize);
        ownVector_Resize(pVector, pVector->m_size - 1);
    }
}

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnTlsCore - TLS system abstraction
///////////////////////////////////////////////////////////////////////////// */
#if IW_ENABLE_TLS
typedef struct _OwnTlsCore
{
#if defined _WIN32
    DWORD tlsKey;
#else
    pthread_key_t tlsKey;
#endif
} OwnTlsCore;

static int ownTlsCore_Init(OwnTlsCore *pTls)
{
    OWN_SAFE_RESET(pTls, sizeof(*pTls));
#if defined _WIN32
    pTls->tlsKey = TlsAlloc();
    if(pTls->tlsKey == TLS_OUT_OF_INDEXES)
        return OWN_STATUS_FAIL;
#else
    if(pthread_key_create(&pTls->tlsKey, NULL) != 0)
        return OWN_STATUS_FAIL;
#endif
    return OWN_STATUS_OK;
}

static int ownTlsCore_Release(OwnTlsCore *pTls)
{
#if defined _WIN32
    if(TlsFree(pTls->tlsKey) == 0)
        return OWN_STATUS_FAIL;
#else
    if(pthread_key_delete(pTls->tlsKey) != 0)
        return OWN_STATUS_FAIL;
#endif
    return OWN_STATUS_OK;
}

static void* ownTlsCore_GetData(OwnTlsCore *pTls)
{
#if defined _WIN32
    return TlsGetValue(pTls->tlsKey);
#else
    return pthread_getspecific(pTls->tlsKey);
#endif
}

static int ownTlsCore_SetData(OwnTlsCore *pTls, void *pData)
{
#if defined _WIN32
    if(TlsSetValue(pTls->tlsKey, pData) == 0)
        return OWN_STATUS_FAIL;
#else
    if(pthread_setspecific(pTls->tlsKey, pData) != 0)
        return OWN_STATUS_FAIL;
#endif
    return OWN_STATUS_OK;
}
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnTlsStorage - TLS data storage interface
///////////////////////////////////////////////////////////////////////////// */
#if IW_ENABLE_TLS
typedef struct _OwnTlsStorage
{
    OwnVector    statesVector;
    OwnVector    threadsVector;

    OwnTlsCore   tls;
    OwnMutex     mutex;
} OwnTlsStorage;

typedef struct _OwnTlsStorageTD
{
    OwnVector  dataVector;  /* Thread data array */
    size_t    index;       /* Index of the thread */
} OwnTlsStorageTD;

static int ownTlsStorage_Init(OwnTlsStorage *pTls)
{
    OWN_SAFE_RESET(pTls, sizeof(*pTls));

    if(ownMutex_Init(&pTls->mutex) < 0)
        return OWN_STATUS_FAIL;
    if(ownTlsCore_Init(&pTls->tls) < 0)
    {
        ownMutex_Release(&pTls->mutex);
        return OWN_STATUS_FAIL;
    }

    ownVector_Init(&pTls->statesVector, sizeof(int), 4);
    ownVector_Init(&pTls->threadsVector, sizeof(OwnTlsStorageTD*), 16);

    return OWN_STATUS_OK;
}

static int ownTlsStorage_Release(OwnTlsStorage *pTls)
{
    size_t i, j;

    OwnTlsStorageTD *pThreads;

    for(i = 0; i < pTls->threadsVector.m_size; i++)
    {
        pThreads = ((OwnTlsStorageTD**)pTls->threadsVector.m_pBuffer)[i];
        if(pThreads)
        {
            for(j = 0; j < pThreads->dataVector.m_size; j++)
            {
                /* Check that all data is destroyed. Data pointers must be deallocated externally*/
                if(((void**)pThreads->dataVector.m_pBuffer)[j])
                    return OWN_STATUS_FAIL;
            }
            ownVector_Release(&pThreads->dataVector);
            OWN_SAFE_FREE(pThreads);
        }
    }

    ownVector_Release(&pTls->statesVector);
    ownVector_Release(&pTls->threadsVector);

    if(ownTlsCore_Release(&pTls->tls) < 0)
        return OWN_STATUS_FAIL;
    if(ownMutex_Release(&pTls->mutex) < 0)
        return OWN_STATUS_FAIL;

    return OWN_STATUS_OK;
}

static size_t ownTlsStorage_ReserveDataIndex(OwnTlsStorage *pTls)
{
    size_t slot;
    if(ownMutex_Lock(&pTls->mutex) != OWN_STATUS_OK)
        return OWN_IDX_INVALID;

    // Find unused slots
    for(slot = 0; slot < pTls->statesVector.m_size; slot++)
    {
        int *pStates = (int*)pTls->statesVector.m_pBuffer;
        if(!pStates[slot])
        {
            pStates[slot] = 1;
            ownMutex_Unlock(&pTls->mutex);
            return slot;
        }
    }

    // Create new slot
    slot = pTls->statesVector.m_size;
    ownVector_Resize(&pTls->statesVector, pTls->statesVector.m_size+1);
    ((int*)pTls->statesVector.m_pBuffer)[slot] = 1;
    ownMutex_Unlock(&pTls->mutex);
    return slot;
}

static int ownTlsStorage_DataVector(OwnTlsStorage *pTls, size_t dataIdx, OwnVector *pDataVector, int bClear)
{
    size_t    i;
    OwnVector *pThreadDataVector;
    void     *pThreadData;

    ownMutex_Lock(&pTls->mutex);
    if(pTls->statesVector.m_size <= dataIdx)
    {
        ownMutex_Unlock(&pTls->mutex);
        return OWN_STATUS_FAIL;
    }

    for(i = 0; i < pTls->threadsVector.m_size; i++)
    {
        pThreadDataVector = &((OwnTlsStorageTD**)pTls->threadsVector.m_pBuffer)[i]->dataVector;
        pThreadData       = ((void**)pThreadDataVector->m_pBuffer)[dataIdx];
        if(pThreadDataVector->m_size > dataIdx && pThreadData)
        {
            if(pDataVector)
                ownVector_PushBack(pDataVector, &pThreadData);
            if(bClear)
                ((void**)pThreadDataVector->m_pBuffer)[dataIdx] = 0;
        }
    }

    if(bClear)
        ((int*)pTls->statesVector.m_pBuffer)[dataIdx] = 0;
    ownMutex_Unlock(&pTls->mutex);

    return OWN_STATUS_OK;
}

static int ownTlsStorage_GetDataVector(OwnTlsStorage *pTls, size_t dataIdx, OwnVector *pDataVector)
{
    if(!pDataVector)
        return OWN_STATUS_FAIL;

    return ownTlsStorage_DataVector(pTls, dataIdx, pDataVector, 0);
}

static int ownTlsStorage_ResetData(OwnTlsStorage *pTls, size_t dataIdx)
{
    return ownTlsStorage_DataVector(pTls, dataIdx, 0, 1);
}

static void* ownTlsStorage_GetData(OwnTlsStorage *pTls, size_t dataIdx)
{
    OwnTlsStorageTD *pThreadData;
    if(pTls->statesVector.m_size <= dataIdx)
        return 0;

    pThreadData = (OwnTlsStorageTD*)ownTlsCore_GetData(&pTls->tls);
    if(pThreadData && pThreadData->dataVector.m_size > dataIdx)
        return ((void**)(pThreadData->dataVector.m_pBuffer))[dataIdx];

    return 0;
}

static int ownTlsStorage_SetData(OwnTlsStorage *pTls, size_t dataIdx, void* pData)
{
    OwnTlsStorageTD *pThreadData = 0;
    if(pTls->statesVector.m_size <= dataIdx && !pData)
        return OWN_STATUS_FAIL;

    pThreadData = (OwnTlsStorageTD*)ownTlsCore_GetData(&pTls->tls);
    if(!pThreadData)
    {
        pThreadData = (OwnTlsStorageTD*)OWN_SAFE_MALLOC(sizeof(OwnTlsStorageTD));
        if(!pThreadData)
            return OWN_STATUS_FAIL;

        ownVector_Init(&pThreadData->dataVector, sizeof(void*), 32);
        ownTlsCore_SetData(&pTls->tls, pThreadData);
        {
            ownMutex_Lock(&pTls->mutex);
            pThreadData->index = pTls->threadsVector.m_size;
            ownVector_PushBack(&pTls->threadsVector, &pThreadData);
            ownMutex_Unlock(&pTls->mutex);
        }
    }

    if(dataIdx >= pThreadData->dataVector.m_size)
    {
        void *null = NULL;
        ownMutex_Lock(&pTls->mutex);
        while(dataIdx >= pThreadData->dataVector.m_size)
            ownVector_PushBack(&pThreadData->dataVector, &null);
        ownMutex_Unlock(&pTls->mutex);
    }
    ((void**)(pThreadData->dataVector.m_pBuffer))[dataIdx] = pData;

    return OWN_STATUS_OK;
}

static OwnTlsStorage* ownGlobalTlsStorage(int bRelease)
{
    static OwnTlsStorage *pStorage = NULL;
    if(!pStorage && ownGlobalGetInitState() == OWN_STATE_INITIALIZATION)
    {
        pStorage = (OwnTlsStorage*)OWN_SAFE_MALLOC(sizeof(OwnTlsStorage));
        if(!pStorage)
            return NULL;

        ownTlsStorage_Init(pStorage);
    }
    else if(bRelease && pStorage)
    {
        ownTlsStorage_Release(pStorage);
        OWN_SAFE_FREE(pStorage);
        pStorage = 0;
    }
    return pStorage;
}

static OwnTlsStorage* ownGlobalGetTlsStorage(void)
{
    return ownGlobalTlsStorage(0);
}
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   IwTls - TLS data storage interface
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwTls_Init(IwTls *pTls, IwTlsDestructor destructor)
{
#if IW_ENABLE_TLS
    OwnTlsStorage *pStorage;

    if(!pTls || !destructor)
        return ippStsNullPtrErr;

    pTls->m_idx         = OWN_IDX_INVALID;
    pTls->m_desctuctor  = destructor;
    pTls->m_pTlsStorage = NULL;

    pStorage = ownGlobalGetTlsStorage();
    if(!pStorage) // No global storage available, create new storage
    {
        pStorage = (OwnTlsStorage*)OWN_SAFE_MALLOC(sizeof(OwnTlsStorage));
        if(!pStorage)
            return ippStsMemAllocErr;

        if(ownTlsStorage_Init(pStorage) < OWN_STATUS_OK)
        {
            OWN_SAFE_FREE(pStorage);
            return ippStsErr;
        }

        pTls->m_idx = ownTlsStorage_ReserveDataIndex(pStorage);
        if(pTls->m_idx == OWN_IDX_INVALID)
        {
            ownTlsStorage_Release(pStorage);
            OWN_SAFE_FREE(pStorage);
            return ippStsErr;
        }
        pTls->m_pTlsStorage = pStorage;
    }
    else
    {
        pTls->m_idx = ownTlsStorage_ReserveDataIndex(pStorage);
        if(pTls->m_idx == OWN_IDX_INVALID)
            return ippStsErr;
    }

    return ippStsNoErr;
#else
    if(!pTls || !destructor)
        return ippStsNullPtrErr;

    pTls->m_idx         = OWN_IDX_INVALID;
    pTls->m_desctuctor  = destructor;
    pTls->m_pTlsStorage = NULL;

    return ippStsUnknownFeature;
#endif
}

IW_DECL(IppStatus) iwTls_Set(IwTls *pTls, void *pData)
{
#if IW_ENABLE_TLS
    OwnTlsStorage *pStorage;
    void          *pOldData;

    if(!pTls)
        return ippStsNullPtrErr;
    if(pTls->m_idx == OWN_IDX_INVALID)
        return ippStsErr;

    pStorage = ownGlobalGetTlsStorage();
    if(!pStorage)
    {
        pStorage = (OwnTlsStorage*)pTls->m_pTlsStorage;
        if(!pStorage)
            return ippStsErr;
    }

    pOldData = ownTlsStorage_GetData(pStorage, pTls->m_idx);
    if(pOldData != pData)
    {
        if(pTls->m_desctuctor && pOldData)
            pTls->m_desctuctor(pOldData);

        if(ownTlsStorage_SetData(pStorage, pTls->m_idx, pData) != OWN_STATUS_OK)
            return ippStsErr;
    }
    return ippStsNoErr;
#else
    if(!pTls)
        return ippStsNullPtrErr;

    pTls->m_pTlsStorage = pData;

    return ippStsUnknownFeature;
#endif
}

IW_DECL(void*) iwTls_Get(const IwTls *pTls)
{
#if IW_ENABLE_TLS
    OwnTlsStorage *pStorage;

    if(!pTls)
        return NULL;
    if(pTls->m_idx == OWN_IDX_INVALID)
        return NULL;

    pStorage = ownGlobalGetTlsStorage();
    if(!pStorage)
    {
        pStorage = (OwnTlsStorage*)pTls->m_pTlsStorage;
        if(!pStorage)
            return NULL;
    }

    return ownTlsStorage_GetData(pStorage, pTls->m_idx);
#else
    if(!pTls)
        return NULL;

    return pTls->m_pTlsStorage;
#endif
}

IW_DECL(IppStatus) iwTls_ReleaseData(IwTls *pTls)
{
#if IW_ENABLE_TLS
    OwnTlsStorage *pStorage;

    if(!pTls)
        return ippStsNullPtrErr;

    pStorage = ownGlobalGetTlsStorage();
    if(!pStorage)
    {
        pStorage = (OwnTlsStorage*)pTls->m_pTlsStorage;
        if(!pStorage)
            return ippStsErr;
    }

    if(pTls->m_idx != OWN_IDX_INVALID)
    {
        if(pTls->m_desctuctor)
        {
            size_t   i;
            OwnVector vData;
            ownVector_Init(&vData, sizeof(void*), 16);

            if(ownTlsStorage_GetDataVector(pStorage, pTls->m_idx, &vData) != OWN_STATUS_OK)
            {
                ownVector_Release(&vData);
                return ippStsErr;
            }

            for(i = 0; i < vData.m_size; i++)
            {
                pTls->m_desctuctor(((void**)vData.m_pBuffer)[i]);
            }
            ownVector_Release(&vData);
        }

        if(ownTlsStorage_ResetData(pStorage, pTls->m_idx) != OWN_STATUS_OK)
            return ippStsErr;
    }

    return ippStsNoErr;
#else
    if(!pTls)
        return ippStsNullPtrErr;

    pTls->m_desctuctor(pTls->m_pTlsStorage);

    return ippStsUnknownFeature;
#endif
}

IW_DECL(IppStatus) iwTls_Release(IwTls *pTls)
{
#if IW_ENABLE_TLS
    OwnTlsStorage *pStorage;

    if(!pTls)
        return ippStsNullPtrErr;

    pStorage = ownGlobalGetTlsStorage();
    if(!pStorage)
    {
        pStorage = (OwnTlsStorage*)pTls->m_pTlsStorage;
        if(!pStorage)
            return ippStsNoErr;
    }

    if(pTls->m_idx != OWN_IDX_INVALID)
    {
        iwTls_ReleaseData(pTls);
        pTls->m_idx = OWN_IDX_INVALID;
    }

    if(pTls->m_pTlsStorage)
    {
        ownTlsStorage_Release(pStorage);
        OWN_SAFE_FREE(pStorage);
        pTls->m_pTlsStorage = NULL;
    }
    return ippStsNoErr;
#else
    if(!pTls)
        return ippStsNullPtrErr;

    pTls->m_desctuctor(pTls->m_pTlsStorage);

    return ippStsUnknownFeature;
#endif
}

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnBufferPool - Memory pool manager
///////////////////////////////////////////////////////////////////////////// */
#if OWN_ENABLE_BUFFER_POOL

#define OWN_BUFFER_POOL_INIT_SIZE         131072  /* 128KB initial chunk size */
#define OWN_BUFFER_POOL_MAX_RETAIN_CYCLES 4       /* Maximum number of release cycles before buffer will be actually removed */

typedef struct _OwnBufferPoolEntry
{
    void   *ptr;
    size_t  size;
    int     locked;
    int     cycle;
} OwnBufferPoolEntry;

typedef struct _OwnBufferPool
{
    OwnVector poolVector;
    size_t   memoryTotal;
    int      chunks;
    int      chunksActive;
    int      allocations;
    int      releases;
    int      cycles;
} OwnBufferPool;

static void ownBufferPool_Init(OwnBufferPool *pPool)
{
    OWN_SAFE_RESET(pPool, sizeof(*pPool));

    pPool->memoryTotal  = 0;
    pPool->chunks       = 0;
    pPool->chunksActive = 0;
    pPool->allocations  = 0;
    pPool->releases     = 0;
    pPool->cycles       = 0;
    ownVector_Init(&pPool->poolVector, sizeof(OwnBufferPoolEntry), 8);
}

static int ownBufferPool_CleanUp(OwnBufferPool *pPool, int bForce)
{
    OwnBufferPoolEntry *pEntry;
    int status = OWN_STATUS_FAIL;
    size_t i;

    for(i = 0; i < pPool->poolVector.m_size; i++)
    {
        pEntry = &((OwnBufferPoolEntry*)pPool->poolVector.m_pBuffer)[i];

        if(pEntry->ptr && (!pEntry->locked || bForce))
        {
            pPool->chunks--;
            pPool->releases++;
            pPool->memoryTotal -= pEntry->size;

            OWN_SAFE_FREE(pEntry->ptr);
            pEntry->size  = 0;
            pEntry->ptr   = 0;
            pEntry->cycle = 0;
        }
    }
    return status;
}

static void ownBufferPool_Release(OwnBufferPool *pPool)
{
    if(!pPool)
        return;

    ownBufferPool_CleanUp(pPool, 1);
    ownVector_Release(&pPool->poolVector);
}

static void* ownBufferPool_GetBuffer(OwnBufferPool *pPool, size_t size)
{
    OwnBufferPoolEntry *pEntry;
    size_t i = 0;
    int idx = -1;

    if(!size)
        return 0;

    for(i = 0; i < pPool->poolVector.m_size; i++)
    {
        pEntry = &((OwnBufferPoolEntry*)pPool->poolVector.m_pBuffer)[i];
        if(pEntry->ptr && pEntry->size >= size && !pEntry->locked)
        {
            pPool->chunksActive++;
            pEntry->locked = 1;
            return pEntry->ptr;
        }
    }

    /* Find empty index */
    for(i = 0; i < pPool->poolVector.m_size; i++)
    {
        if(!((OwnBufferPoolEntry*)pPool->poolVector.m_pBuffer)[i].ptr)
            idx = (int)i;
    }
    if(idx < 0)
    {
        idx = (int)pPool->poolVector.m_size;
        ownVector_Resize(&pPool->poolVector, pPool->poolVector.m_size+1);
    }

    /* Create new buffer */
    pEntry = &((OwnBufferPoolEntry*)pPool->poolVector.m_pBuffer)[idx];
    pEntry->size = size;
    if(pEntry->size < OWN_BUFFER_POOL_INIT_SIZE)
        pEntry->size = OWN_BUFFER_POOL_INIT_SIZE;

    pEntry->ptr    = OWN_SAFE_MALLOC(pEntry->size);
    pEntry->locked = 1;
    pPool->memoryTotal += pEntry->size;
    pPool->allocations++;
    pPool->chunks++;
    pPool->chunksActive++;

    return pEntry->ptr;
}

static int ownBufferPool_ReleaseBuffer(OwnBufferPool *pPool, void *pBuffer, int bHard)
{
    OwnBufferPoolEntry *pEntry;
    int status = OWN_STATUS_FAIL;
    size_t i;

    if(!pBuffer)
        return OWN_STATUS_FAIL;

    for(i = 0; i < pPool->poolVector.m_size; i++)
    {
        pEntry = &((OwnBufferPoolEntry*)pPool->poolVector.m_pBuffer)[i];

        if(pEntry->ptr == pBuffer && pEntry->locked)
        {
            pPool->chunksActive--;
            pEntry->locked = 0;
            if(bHard)
            {
                pPool->chunks--;
                pPool->releases++;
                pPool->memoryTotal -= pEntry->size;

                OWN_SAFE_FREE(pEntry->ptr);
                pEntry->ptr   = 0;
                pEntry->size  = 0;
                pEntry->cycle = 0;
            }
            else
                pEntry->cycle = pPool->cycles;
            pPool->cycles++;
            return OWN_STATUS_OK;
        }
        else if(pEntry->ptr && !pEntry->locked && (pPool->cycles - pEntry->cycle) > OWN_BUFFER_POOL_MAX_RETAIN_CYCLES)
        {
            pPool->chunks--;
            pPool->releases++;
            pPool->memoryTotal -= pEntry->size;

            OWN_SAFE_FREE(pEntry->ptr);
            pEntry->ptr   = 0;
            pEntry->size  = 0;
            pEntry->cycle = 0;
        }
    }
    return status;
}

static void ownGlobalBufferRelease(size_t idx, int full)
{
    if(idx == OWN_IDX_INVALID)
        return;

    {
        size_t         i;
        OwnVector       dataVector;
        OwnTlsStorage *pTls  = ownGlobalGetTlsStorage(); /* Get global TLS state */
        if(!pTls)
            return;

        ownVector_Init(&dataVector, sizeof(void*), 32);
        if(ownTlsStorage_GetDataVector(pTls, idx, &dataVector) != OWN_STATUS_OK)
            return;
        for(i = 0; i < dataVector.m_size; i++)
        {
            OwnBufferPool *pPool = ((OwnBufferPool**)dataVector.m_pBuffer)[i];
            if(pPool)
            {
                ownBufferPool_CleanUp(pPool, full); /* Release unused buffer pool memory */
                if(full)
                {
                    ownVector_Release(&pPool->poolVector);   /* Release buffers vector */
                    OWN_SAFE_FREE(pPool);                   /* Release object itself */
                }
            }
        }
        if(full)
            ownTlsStorage_ResetData(pTls, idx);
        ownVector_Release(&dataVector);
    }
}

static size_t ownGlobalBufferPoolIdx(int bRelease)
{
    static size_t idx = OWN_IDX_INVALID;
    if(idx == OWN_IDX_INVALID && ownGlobalGetInitState() == OWN_STATE_INITIALIZATION)
    {
        OwnTlsStorage *pTls  = ownGlobalGetTlsStorage(); /* Get global TLS state */
        if(!pTls)
            return idx; /* Global TLS was not initialized */

        idx = ownTlsStorage_ReserveDataIndex(pTls);
    }
    else if(bRelease && idx != OWN_IDX_INVALID)
    {
        ownGlobalBufferRelease(idx, 1);
        idx = OWN_IDX_INVALID;
    }
    return idx;
}

static OwnBufferPool* ownGlobalGetBufferPool(void)
{
    size_t         idx   = ownGlobalBufferPoolIdx(0);
    OwnBufferPool *pPool = 0;
    OwnTlsStorage *pTls  = ownGlobalGetTlsStorage(); /* Get global TLS state */
    if(!pTls)
        return 0; /* Global TLS was not initialized */

    if(idx == OWN_IDX_INVALID)
        return 0;

    pPool = (OwnBufferPool*)ownTlsStorage_GetData(pTls, idx);
    if(!pPool)
    {
        pPool = (OwnBufferPool*)OWN_SAFE_MALLOC(sizeof(OwnBufferPool));
        if(!pPool)
            return NULL;

        ownBufferPool_Init(pPool);
        ownTlsStorage_SetData(pTls, idx, pPool);
    }
    return pPool;
}
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   OwnBufferPool - External memory interface
///////////////////////////////////////////////////////////////////////////// */
void* IPP_STDCALL ownSharedMalloc(IwSize size)
{
#if OWN_ENABLE_BUFFER_POOL
    OwnBufferPool *pPool = ownGlobalGetBufferPool(); /* Get global buffer pool state */
    if(pPool)
    {
        void *pBuffer = ownBufferPool_GetBuffer(pPool, size);
        if(pBuffer)
            return pBuffer;
    }
#endif
    /* Uninitialized or error in buffer pool, use simple malloc */
    return ippMalloc_L(size);
}

void IPP_STDCALL ownSharedFree(void* pBuffer)
{
#if OWN_ENABLE_BUFFER_POOL
    OwnBufferPool *pPool = ownGlobalGetBufferPool(); /* Get global buffer pool state */
    if(pPool)
    {
        if(ownBufferPool_ReleaseBuffer(pPool, pBuffer, 0) >= 0)
            return;
    }
#endif

    /* Uninitialized or error in buffer pool, use simple free */
    ippFree(pBuffer);
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IW library-scope objects initialization
///////////////////////////////////////////////////////////////////////////// */
#if OWN_ENABLE_BUFFER_POOL
IW_DECL(void) iwInit()
{
    int state = ownGlobalGetInitState();
    if(state == OWN_STATE_NOT_INITIALIZED)
    {
        ownGlobalSetInitState(OWN_STATE_INITIALIZATION);

        ownGlobalGetTlsStorage();   /* Initialize tls static object  */
        ownGlobalGetBufferPool();   /* initialize global buffer pool */

        ownGlobalSetInitState(OWN_STATE_INITIALIZED);
    }
}
IW_DECL(void) iwCleanup()
{
    ownGlobalBufferRelease(ownGlobalBufferPoolIdx(0), 0);
}
IW_DECL(void) iwRelease()
{
    int state = ownGlobalGetInitState();
    if(state == OWN_STATE_INITIALIZED)
    {
        ownGlobalBufferPoolIdx(1);
        ownGlobalTlsStorage(1);

        ownGlobalSetInitState(OWN_STATE_NOT_INITIALIZED);
    }
}
#endif

IW_DECL(void) iwSetThreadsNum(int threads)
{
#if IW_ENABLE_THREADING_LAYER
    ippSetNumThreads_LT(threads);
#else
    (void)threads;
#endif
}
IW_DECL(int)  iwGetThreadsNum()
{
#if IW_ENABLE_THREADING_LAYER
    int threads;
    ippGetNumThreads_LT(&threads);
    return threads;
#else
    return 0;
#endif
}
IW_DECL(int)  iwGetThreadsNumDefault()
{
#if IW_ENABLE_THREADING_LAYER
    return IPP_MIN(omp_get_num_procs(), omp_get_max_threads());
#else
    return 0;
#endif
}

#if OWN_ENABLE_BUFFER_POOL
IW_DECL(void) iwGetDebugInfo(IwStateDebugInfo *pInfo)
{
    OwnTlsStorage *pTls  = NULL;
    OwnBufferPool *pPool = NULL;

    pTls  = ownGlobalGetTlsStorage();
    if(pTls)
    {
        pInfo->m_tlsInitialized    = 1;
        pInfo->m_tlsDataIndexesMax = pTls->statesVector.m_size;
        pInfo->m_tlsThreadsMax     = pTls->threadsVector.m_size;
    }
    else
        pInfo->m_tlsInitialized = 0;

    pPool = ownGlobalGetBufferPool(); /* Get global buffer pool state */
    if(pPool)
    {
        size_t i;
        OwnBufferPoolEntry *pEntry;

        pInfo->m_poolInitialized  = 1;
        pInfo->m_poolMemoryTotal  = pPool->memoryTotal;
        pInfo->m_poolChunks       = pPool->chunks;
        pInfo->m_poolChunksLocked = pPool->chunksActive;
        pInfo->m_poolAllocations  = pPool->allocations;
        pInfo->m_poolReleases     = pPool->releases;
        pInfo->m_poolEntries      = pPool->poolVector.m_size;
        for(i = 0; i < IPP_MIN(pPool->poolVector.m_size, 16); i++)
        {
            pEntry = &((OwnBufferPoolEntry*)pPool->poolVector.m_pBuffer)[i];
            pInfo->m_poolEntrySizes[i] = pEntry->size;
        }
    }
    else
        pInfo->m_poolInitialized = 0;
}
#endif
