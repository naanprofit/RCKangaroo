// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include "utils.h"
#include <wchar.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#ifdef _WIN32

#else

void _BitScanReverse64(u32* index, u64 msk) 
{
    *index = 63 - __builtin_clzll(msk); 
}

void _BitScanForward64(u32* index, u64 msk) 
{
    *index = __builtin_ffsll(msk) - 1; 
}

u64 _umul128(u64 m1, u64 m2, u64* hi) 
{ 
    uint128_t ab = (uint128_t)m1 * m2; *hi = (u64)(ab >> 64); return (u64)ab; 
}

u64 __shiftright128 (u64 LowPart, u64 HighPart, u8 Shift)
{
   u64 ret;
   __asm__ ("shrd {%[Shift],%[HighPart],%[LowPart]|%[LowPart], %[HighPart], %[Shift]}" 
      : [ret] "=r" (ret)
      : [LowPart] "0" (LowPart), [HighPart] "r" (HighPart), [Shift] "Jc" (Shift)
      : "cc");
   return ret;
}

u64 __shiftleft128 (u64 LowPart, u64 HighPart, u8 Shift)
{
   u64 ret;
   __asm__ ("shld {%[Shift],%[LowPart],%[HighPart]|%[HighPart], %[LowPart], %[Shift]}" 
      : [ret] "=r" (ret)
      : [LowPart] "r" (LowPart), [HighPart] "0" (HighPart), [Shift] "Jc" (Shift)
      : "cc");
   return ret;
}   

u64 GetTickCount64()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return (u64)(ts.tv_nsec / 1000000) + ((u64)ts.tv_sec * 1000ull);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DB_REC_LEN			32
#define DB_FIND_LEN			9
#define DB_MIN_GROW_CNT		2

//we need advanced memory management to reduce memory fragmentation
//everything will be stable up to about 8TB RAM

#define MEM_PAGE_SIZE		(128 * 1024)
#define RECS_IN_PAGE		(MEM_PAGE_SIZE / DB_REC_LEN)
#define MAX_PAGES_CNT		(0xFFFFFFFF / RECS_IN_PAGE)

MemPool::MemPool()
{
	pnt = 0;
}

MemPool::~MemPool()
{
	Clear();
}

void MemPool::Clear()
{
	int cnt = (int)pages.size();
	for (int i = 0; i < cnt; i++)
		free(pages[i]);
	pages.clear();
	pnt = 0;
}

void* MemPool::AllocRec(u32* cmp_ptr)
{
	void* mem;
	if (pages.empty() || (pnt + DB_REC_LEN > MEM_PAGE_SIZE))
	{
		if (pages.size() >= MAX_PAGES_CNT)
			return NULL; //overflow
		pages.push_back(malloc(MEM_PAGE_SIZE));
		pnt = 0;
	}
	u32 page_ind = (u32)pages.size() - 1;
	mem = (u8*)pages[page_ind] + pnt;
	*cmp_ptr = (page_ind * RECS_IN_PAGE) | (pnt / DB_REC_LEN);
	pnt += DB_REC_LEN;
	return mem;
}

void* MemPool::GetRecPtr(u32 cmp_ptr)
{
	u32 page_ind = cmp_ptr / RECS_IN_PAGE;
	u32 rec_ind = cmp_ptr % RECS_IN_PAGE;
	return (u8*)pages[page_ind] + DB_REC_LEN * rec_ind;
}

TFastBase::TFastBase()
{
        memset(lists, 0, sizeof(lists));
        memset(Header, 0, sizeof(Header));
#ifdef _WIN32
        hFile = NULL;
        hMap = NULL;
#else
        fd = -1;
#endif
        mapped_ptr = NULL;
        mapped_size = 0;
        mapped_mode = false;
}

TFastBase::~TFastBase()
{
        CloseMapped();
        Clear();
}

void TFastBase::Clear()
{
        for (int i = 0; i < 256; i++)
        {
                for (int j = 0; j < 256; j++)
                        for (int k = 0; k < 256; k++)
                        {
                                if (!mapped_mode && lists[i][j][k].data)
                                        free(lists[i][j][k].data);
                                lists[i][j][k].data = NULL;
                                lists[i][j][k].capacity = 0;
                                lists[i][j][k].cnt = 0;
                        }
                if (!mapped_mode)
                        mps[i].Clear();
        }
}

u64 TFastBase::GetBlockCnt()
{
	u64 blockCount = 0;
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			for (int k = 0; k < 256; k++)
			blockCount += lists[i][j][k].cnt;
	return blockCount;
}

// http://en.cppreference.com/w/cpp/algorithm/lower_bound
int TFastBase::lower_bound(TListRec* list, int mps_ind, u8* data)
{
	int count = list->cnt;
	int it, first, step;
	first = 0;
	while (count > 0)
	{
		it = first;
		step = count / 2;   
		it += step;
		void* ptr = mps[mps_ind].GetRecPtr(list->data[it]);
		if (memcmp(ptr, data, DB_FIND_LEN) < 0)
		{
			first = ++it;
			count -= step + 1;
		}
		else
			count = step;
	}
	return first;
}
 
u8* TFastBase::AddDataBlock(u8* data, int pos)
{
	TListRec* list = &lists[data[0]][data[1]][data[2]];
	if (list->cnt >= list->capacity)
	{
		u32 grow = list->capacity / 2;
		if (grow < DB_MIN_GROW_CNT)
			grow = DB_MIN_GROW_CNT;
		u32 newcap = list->capacity + grow;
		if (newcap > 0xFFFF)
			newcap = 0xFFFF;
		if (newcap <= list->capacity)
			return NULL; //failed
		list->data = (u32*)realloc(list->data, newcap * sizeof(u32));
		list->capacity = newcap;
	}
	int first = (pos < 0) ? lower_bound(list, data[0], data + 3) : pos;
	memmove(list->data + first + 1, list->data + first, (list->cnt - first) * sizeof(u32));
	u32 cmp_ptr;
	void* ptr = mps[data[0]].AllocRec(&cmp_ptr);
	list->data[first] = cmp_ptr;
	memcpy(ptr, data + 3, DB_REC_LEN);
	list->cnt++;
	return (u8*)ptr;
}

u8* TFastBase::FindDataBlock(u8* data)
{
	bool res = false;
	TListRec* list = &lists[data[0]][data[1]][data[2]];
	int first = lower_bound(list, data[0], data + 3);
	if (first == list->cnt)
		return NULL;
	void* ptr = mps[data[0]].GetRecPtr(list->data[first]);
	if (memcmp(ptr, data + 3, DB_FIND_LEN))
		return NULL;
	return (u8*)ptr;
}

u8* TFastBase::FindOrAddDataBlock(u8* data)
{
	void* ptr;
	TListRec* list = &lists[data[0]][data[1]][data[2]];
	int first = lower_bound(list, data[0], data + 3);
	if (first == list->cnt)
		goto label_not_found;
	ptr = mps[data[0]].GetRecPtr(list->data[first]);
	if (memcmp(ptr, data + 3, DB_FIND_LEN))
		goto label_not_found;
	return (u8*)ptr;
label_not_found:
	AddDataBlock(data, first);
	return NULL;
}

//slow but I hope you are not going to create huge DB with this proof-of-concept software
bool TFastBase::LoadFromFile(char* fn)
{
	Clear();
	FILE* fp = fopen(fn, "rb");
	if (!fp)
		return false;
	if (fread(Header, 1, sizeof(Header), fp) != sizeof(Header))
	{
		fclose(fp);
		return false;
	}
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			for (int k = 0; k < 256; k++)
			{
				TListRec* list = &lists[i][j][k];
				fread(&list->cnt, 1, 2, fp);
				if (list->cnt)
				{
					u32 grow = list->cnt / 2;
					if (grow < DB_MIN_GROW_CNT)
						grow = DB_MIN_GROW_CNT;
					u32 newcap = list->cnt + grow;
					if (newcap > 0xFFFF)
						newcap = 0xFFFF;
					list->data = (u32*)realloc(list->data, newcap * sizeof(u32));
					list->capacity = newcap;

					for (int m = 0; m < list->cnt; m++)
					{
						u32 cmp_ptr;
						void* ptr = mps[i].AllocRec(&cmp_ptr);
						list->data[m] = cmp_ptr;
						if (fread(ptr, 1, DB_REC_LEN, fp) != DB_REC_LEN)
						{
							fclose(fp);
							return false;
						}
					}
				}
			}
	fclose(fp);
	return true;
}

bool TFastBase::SaveToFile(char* fn)
{
	FILE* fp = fopen(fn, "wb");
	if (!fp)
		return false;
	if (fwrite(Header, 1, sizeof(Header), fp) != sizeof(Header))
	{
		fclose(fp);
		return false;
	}
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			for (int k = 0; k < 256; k++)
			{
				TListRec* list = &lists[i][j][k];
				fwrite(&list->cnt, 1, 2, fp);
				for (int m = 0; m < list->cnt; m++)
				{
					void* ptr = mps[i].GetRecPtr(list->data[m]);
					if (fwrite(ptr, 1, DB_REC_LEN, fp) != DB_REC_LEN)
					{
						fclose(fp);
						return false;
					}
				}
			}
	fclose(fp);
	return true;
}

bool IsFileExist(char* fn)
{
	FILE* fp = fopen(fn, "rb");
	if (!fp)
		return false;
	fclose(fp);
	return true;
}

int TFastBase::lower_bound_mapped(TListRec* list, u8* data)
{
        int count = list->cnt;
        int it, first, step;
        first = 0;
        while (count > 0)
        {
                it = first;
                step = count / 2;
                it += step;
                void* ptr = (u8*)list->data + DB_REC_LEN * it;
                if (memcmp(ptr, data, DB_FIND_LEN) < 0)
                {
                        first = ++it;
                        count -= step + 1;
                }
                else
                        count = step;
        }
        return first;
}

static bool write_base128(FILE* fp, const u8* data, size_t len)
{
        int bits = 0;
        unsigned int acc = 0;
        for (size_t i = 0; i < len; ++i)
        {
                acc = (acc << 8) | data[i];
                bits += 8;
                while (bits >= 7)
                {
                        u8 out = (acc >> (bits - 7)) & 0x7F;
                        if (fputc(out, fp) == EOF)
                                return false;
                        bits -= 7;
                }
        }
        if (bits)
        {
                u8 out = (acc << (7 - bits)) & 0x7F;
                if (fputc(out, fp) == EOF)
                        return false;
        }
        return true;
}

static bool read_base128(FILE* fp, u8* data, size_t len)
{
        int bits = 0;
        unsigned int acc = 0;
        size_t pos = 0;
        while (pos < len)
        {
                int ch = fgetc(fp);
                if (ch == EOF)
                        return false;
                acc = (acc << 7) | (ch & 0x7F);
                bits += 7;
                if (bits >= 8)
                {
                        bits -= 8;
                        data[pos++] = (acc >> bits) & 0xFF;
                }
        }
        return true;
}

bool TFastBase::SaveToFileBase128(char* fn)
{
        FILE* fp = fopen(fn, "wb");
        if (!fp)
                return false;
        if (!write_base128(fp, Header, sizeof(Header)))
        {
                fclose(fp);
                return false;
        }
        for (int i = 0; i < 256; i++)
                for (int j = 0; j < 256; j++)
                        for (int k = 0; k < 256; k++)
                        {
                                TListRec* list = &lists[i][j][k];
                                u8 cnt_buf[2];
                                cnt_buf[0] = list->cnt & 0xFF;
                                cnt_buf[1] = (list->cnt >> 8) & 0xFF;
                                if (!write_base128(fp, cnt_buf, 2))
                                {
                                        fclose(fp);
                                        return false;
                                }
                                for (int m = 0; m < list->cnt; m++)
                                {
                                        void* ptr = mps[i].GetRecPtr(list->data[m]);
                                        if (!write_base128(fp, (u8*)ptr, DB_REC_LEN))
                                        {
                                                fclose(fp);
                                                return false;
                                        }
                                }
                        }
        fclose(fp);
        return true;
}

bool TFastBase::LoadFromFileBase128(char* fn)
{
        Clear();
        FILE* fp = fopen(fn, "rb");
        if (!fp)
                return false;
        if (!read_base128(fp, Header, sizeof(Header)))
        {
                fclose(fp);
                return false;
        }
        for (int i = 0; i < 256; i++)
                for (int j = 0; j < 256; j++)
                        for (int k = 0; k < 256; k++)
                        {
                                TListRec* list = &lists[i][j][k];
                                u8 cnt_buf[2];
                                if (!read_base128(fp, cnt_buf, 2))
                                {
                                        fclose(fp);
                                        return false;
                                }
                                list->cnt = cnt_buf[0] | (cnt_buf[1] << 8);
                                if (list->cnt)
                                {
                                        u32 grow = list->cnt / 2;
                                        if (grow < DB_MIN_GROW_CNT)
                                                grow = DB_MIN_GROW_CNT;
                                        u32 newcap = list->cnt + grow;
                                        if (newcap > 0xFFFF)
                                                newcap = 0xFFFF;
                                        list->data = (u32*)realloc(list->data, newcap * sizeof(u32));
                                        list->capacity = newcap;
                                        for (int m = 0; m < list->cnt; m++)
                                        {
                                                u8 buf[DB_REC_LEN];
                                                if (!read_base128(fp, buf, DB_REC_LEN))
                                                {
                                                        fclose(fp);
                                                        return false;
                                                }
                                                u32 cmp_ptr;
                                                void* ptr = mps[i].AllocRec(&cmp_ptr);
                                                list->data[m] = cmp_ptr;
                                                memcpy(ptr, buf, DB_REC_LEN);
                                        }
                                }
                        }
        fclose(fp);
        return true;
}

bool TFastBase::OpenMapped(char* fn)
{
        if (mapped_mode)
                CloseMapped();
        Clear();
#ifdef _WIN32
        hFile = CreateFileA(fn, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE)
                return false;
        mapped_size = GetFileSize(hFile, NULL);
        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap)
        {
                CloseHandle(hFile);
                hFile = NULL;
                return false;
        }
        mapped_ptr = (u8*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!mapped_ptr)
        {
                CloseHandle(hMap);
                CloseHandle(hFile);
                hMap = NULL;
                hFile = NULL;
                return false;
        }
#else
        fd = open(fn, O_RDONLY);
        if (fd < 0)
                return false;
        struct stat st;
        if (fstat(fd, &st) < 0)
        {
                close(fd);
                fd = -1;
                return false;
        }
        mapped_size = st.st_size;
        mapped_ptr = (u8*)mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_ptr == MAP_FAILED)
        {
                close(fd);
                fd = -1;
                return false;
        }
#endif
        mapped_mode = true;
        u8* p = mapped_ptr;
        memcpy(Header, p, sizeof(Header));
        p += sizeof(Header);
        for (int i = 0; i < 256; i++)
                for (int j = 0; j < 256; j++)
                        for (int k = 0; k < 256; k++)
                        {
                                u16 cnt = p[0] | (p[1] << 8);
                                lists[i][j][k].cnt = cnt;
                                lists[i][j][k].capacity = 0;
                                lists[i][j][k].data = (u32*)(p + 2);
                                p += 2 + cnt * DB_REC_LEN;
                        }
        return true;
}

void TFastBase::CloseMapped()
{
        if (!mapped_mode)
                return;
        Clear();
#ifdef _WIN32
        if (mapped_ptr)
                UnmapViewOfFile(mapped_ptr);
        if (hMap)
                CloseHandle(hMap);
        if (hFile)
                CloseHandle(hFile);
        hMap = NULL;
        hFile = NULL;
#else
        if (mapped_ptr)
                munmap(mapped_ptr, mapped_size);
        if (fd >= 0)
                close(fd);
        fd = -1;
#endif
        mapped_ptr = NULL;
        mapped_size = 0;
        mapped_mode = false;
}

u8* TFastBase::FindDataBlockMapped(u8* data)
{
        TListRec* list = &lists[data[0]][data[1]][data[2]];
        int first = lower_bound_mapped(list, data + 3);
        if (first == list->cnt)
                return NULL;
        u8* ptr = (u8*)list->data + DB_REC_LEN * first;
        if (memcmp(ptr, data + 3, DB_FIND_LEN))
                return NULL;
        return ptr;
}

bool TFastBase::IsMapped()
{
        return mapped_mode;
}