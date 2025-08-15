// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include <string.h>
#include <stdio.h>
#include <vector>
#include "defs.h"

#ifdef _WIN32

	#include <Windows.h>
	#include <process.h>
	#include <intrin.h>

	#define CSHANDLER		CRITICAL_SECTION
	#define INIT_CS(cs)     InitializeCriticalSection((cs))
	#define DELETE_CS(cs)   DeleteCriticalSection((cs))
	#define LOCK_CS(cs)     EnterCriticalSection((cs))
	#define TRY_LOCK_CS(cs)
	#define UNLOCK_CS(cs)   LeaveCriticalSection((cs))

	#define HHANDLER		HANDLE

#else
	#include <math.h>
	#include <pthread.h>
	#include <unistd.h>
	#include <x86intrin.h>
	#define DWORD           u32
	#define CSHANDLER		pthread_mutex_t
	#define INIT_CS(cs)		{pthread_mutexattr_t attr; pthread_mutexattr_init(&attr); pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE); pthread_mutex_init((cs), &attr);}
	#define DELETE_CS(cs)	pthread_mutex_destroy((cs))
	#define LOCK_CS(cs)		pthread_mutex_lock((cs))
	#define UNLOCK_CS(cs)	pthread_mutex_unlock((cs))
	#define HHANDLER		pthread_t
 
	u64 GetTickCount64();
	static void Sleep(int x) { usleep(x * 1000); }      
    void _BitScanReverse64(u32* index, u64 msk);
    void _BitScanForward64(u32* index, u64 msk);       
    typedef __uint128_t uint128_t;
    u64 _umul128(u64 m1, u64 m2, u64* hi);
    u64 __shiftright128 (u64 LowPart, u64 HighPart, u8 Shift);
    u64 __shiftleft128 (u64 LowPart, u64 HighPart, u8 Shift);
#endif

class CriticalSection
{
private:
	CSHANDLER cs_body;
public:
	CriticalSection() { INIT_CS(&cs_body); };
	~CriticalSection() { DELETE_CS(&cs_body); };

	void Enter() { LOCK_CS(&cs_body); };
	void Leave() { UNLOCK_CS(&cs_body); };
};

#pragma pack(push, 1)
struct TListRec
{
	u16 cnt;
	u16 capacity;
	u32* data;
};
#pragma pack(pop)

// tames file header
#define TAMES_MAGIC "PMAP"
#define TAMES_VERSION 1
#define TAMES_FLAG_LE 0x0001
#define TAMES_FLAG_BASE128 0x0002
#define TAMES_RANGE_SHIFT 8

#pragma pack(push, 1)
struct TamesHeader
{
        char magic[4];
        u8   version;
        u8   stride;
        u16  flags;
        u64  rec_cnt;
};
#pragma pack(pop)

class MemPool
{
private:
	std::vector <void*> pages;
	u32 pnt;
public:
	MemPool();
	~MemPool();
	void Clear();
	inline void* AllocRec(u32* cmp_ptr);
	inline void* GetRecPtr(u32 cmp_ptr);
};

class TFastBase
{
private:
        MemPool mps[256];
        TListRec lists[256][256][256];
        int lower_bound(TListRec* list, int mps_ind, u8* data);
        int lower_bound_mapped(TListRec* list, u8* data);

#ifdef _WIN32
        HHANDLER hFile;
        HHANDLER hMap;
#else
        int fd;
#endif
        u8* mapped_ptr;
        size_t mapped_size;
        bool mapped_mode;
public:
        TamesHeader Header;

        TFastBase();
        ~TFastBase();
        void Clear();
        u8* AddDataBlock(u8* data, int pos = -1);
        u8* FindDataBlock(u8* data);
        u8* FindOrAddDataBlock(u8* data);
        u64 GetBlockCnt();
        bool LoadFromFile(char* fn);
        bool SaveToFile(char* fn);
        bool LoadFromFileBase128(char* fn); // legacy Base128 support; cannot be memory-mapped and requires full in-memory decoding
        bool SaveToFileBase128(char* fn);   // legacy Base128 support, prefer binary pmap
        bool OpenMapped(char* fn);
        void CloseMapped();
        u8* FindDataBlockMapped(u8* data);
        bool IsMapped();
};

// Streaming writer for tames records
struct TamesRecordWriter;
TamesRecordWriter* TamesRecordWriterOpen(const char* path, bool base128, size_t rec_size, u64 prealloc_recs = 0);
bool TamesRecordWriterWrite(TamesRecordWriter* wr, const u8* data);
void TamesRecordWriterClose(TamesRecordWriter* wr);

bool IsFileExist(char* fn);
