// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#include "Bloom.h"


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;
bool gTamesBase128; // legacy Base128 tames format
bool gMultiDP = true;
int gDpCoarseOffset = 0;
int gBloomMBits = 24;
int gBloomK = 3;
int gPhiFold = 1; //phi folding mode
BloomFilter gBloom;
TamesRecordWriter* gTamesWriter = NULL;

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

//	gcnt = 1; //dbg
	if (!gcnt)
		return;

        int drv, rt;
        cudaRuntimeGetVersion(&rt);
        cudaDriverGetVersion(&drv);
        char drvver[100];
        sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

        printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);

        double sys_gb = 0.0;
#ifdef _WIN32
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        if (GlobalMemoryStatusEx(&statex))
                sys_gb = (double)statex.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
#else
        struct sysinfo info;
        if (sysinfo(&info) == 0)
                sys_gb = (double)info.totalram * info.mem_unit / (1024.0 * 1024.0 * 1024.0);
#endif
        printf("System memory: %.2f GB\n", sys_gb);

        cudaError_t cudaStatus;
        for (int i = 0; i < gcnt; i++)
        {
                cudaStatus = cudaSetDevice(i);
                if (cudaStatus != cudaSuccess)
                {
                        printf("cudaSetDevice for gpu %d failed!\r\n", i);
                        continue;
		}

		if (!gGPUs_Mask[i])
			continue;

                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                double gpu_gb = ((double)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
                printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, gpu_gb, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
                printf("Unified memory (GPU + system): %.2f GB\n", gpu_gb + sys_gb);
		
		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
            EcPoint P = ec.MultiplyG_GLV(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
            P = ec.MultiplyG_GLV(gPrivKey);
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
            EcPoint P = ec.MultiplyG_GLV(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
            P = ec.MultiplyG_GLV(gPrivKey);
		return P.IsEqual(pnt);
	}
}


void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

        for (int i = 0; i < cnt; i++)
        {
                u8* p = pPntList2 + i * GPU_DP_SIZE;

                bool bloom_hit = false;
                if (gMultiDP)
                {
                        u64 x_key = *(u64*)p;
                        u64 coarse = x_key >> gDpCoarseOffset;
                        bloom_hit = gBloom.Test(coarse);
                        gBloom.Add(coarse);
                        u64 fine_mask = gDpCoarseOffset ? ((1ull << gDpCoarseOffset) - 1) : 0;
                        if ((x_key & fine_mask) != 0)
                                continue;
                }

                DBRec nrec;
                memcpy(nrec.x, p, 12);
                memcpy(nrec.d, p + 32, 22);
                u8 type_byte = gGenMode ? TAME : p[56];
                u8 nrec_k = type_byte >> 2;
                u8 nrec_type = type_byte & 3;
                nrec.type = type_byte;

                if (gGenMode)
                {
                        if (gTamesWriter)
                                TamesRecordWriterWrite(gTamesWriter, (u8*)&nrec);
                        continue;
                }

                DBRec* pref = NULL;
                if (!gMultiDP)
                {
                        if (db.IsMapped())
                                pref = (DBRec*)db.FindDataBlockMapped((u8*)&nrec);
                        else
                                pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
                        if (!pref)
                                continue;
                }
                else
                {
                        if (bloom_hit)
                        {
                                if (db.IsMapped())
                                        pref = (DBRec*)db.FindDataBlockMapped((u8*)&nrec);
                                else
                                {
                                        pref = (DBRec*)db.FindDataBlock((u8*)&nrec);
                                        if (!pref)
                                        {
                                                db.AddDataBlock((u8*)&nrec);
                                                continue;
                                        }
                                }
                        }
                        else
                        {
                                if (!db.IsMapped())
                                        db.AddDataBlock((u8*)&nrec);
                                continue;
                        }
                }

                DBRec tmp_pref;
                memcpy(&tmp_pref, &nrec, 3);
                memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
                pref = &tmp_pref;
                u8 pref_k = pref->type >> 2;
                u8 pref_type = pref->type & 3;

                if ((pref_type == nrec_type) && (pref_k == nrec_k))
                {
                        if (pref_type == TAME)
                                continue;

                        if (*(u64*)pref->d == *(u64*)nrec.d)
                                continue;
                }

                EcInt w, t;
                int TameType, WildType;
                if (pref_type != TAME)
                {
                        memcpy(w.data, pref->d, sizeof(pref->d));
                        if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                        memcpy(t.data, nrec.d, sizeof(nrec.d));
                        if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                        TameType = nrec_type;
                        WildType = pref_type;
                }
                else
                {
                        memcpy(w.data, nrec.d, sizeof(nrec.d));
                        if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                        memcpy(t.data, pref->d, sizeof(pref->d));
                        if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                        TameType = TAME;
                        WildType = nrec_type;
                }

                if (pref_k == 1) w.MulLambdaN();
                else if (pref_k == 2) w.MulLambda2N();
                if (nrec_k == 1) t.MulLambdaN();
                else if (nrec_k == 2) t.MulLambda2N();

                bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
                if (!res)
                {
                        bool w12 = ((pref_type == WILD1) && (nrec_type == WILD2)) || ((pref_type == WILD2) && (nrec_type == WILD1));
                        if (w12)
                                ;
                        else
                        {
                                printf("Collision Error\r\n");
                                gTotalErrors++;
                        }
                        continue;
                }
                gSolved = true;
                break;
        }
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val);

#if 0
bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 14) || (DP > 60)) 
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
        double MaxTotalOps = 0.0;
        if (gMax > 0)
        {
                MaxTotalOps = gMax * ops;
                double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
                ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
                ram_max /= (1024 * 1024 * 1024); //GB
                printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
                if (MaxTotalOps < ops * 0.1)
                        printf("WARNING: MaxTotalOps is set very low and the search may stop before finding the key\r\n");
        }

        u64 total_kangs = GpuKangs[0]->CalcKangCnt();
        for (int i = 1; i < GpuCnt; i++)
                total_kangs += GpuKangs[i]->CalcKangCnt();
        double path_single_kang = ops / total_kangs;
        double DPs_per_kang = path_single_kang / dp_val;
        printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

       bool tamesRangeMismatch = false;
       if (!gGenMode && gTamesFileName[0])
       {
               printf("load tames...\r\n");
               bool ok = false;
               if (gTamesBase128)
               {
                       ok = db.LoadFromFileBase128(gTamesFileName);
                       if (ok)
                               printf("Base128 tames cannot be memory-mapped and require full in-memory decoding.\r\n");
                       else
                               printf("Base128 tames loading failed\r\n");
               }
               else
               {
                       ok = db.OpenMapped(gTamesFileName);
                       if (!ok)
                       {
                               printf("memory-mapped tames failed, loading into RAM...\r\n");
                               ok = db.LoadFromFile(gTamesFileName);
                               if (!ok)
                                       printf("binary tames loading failed\r\n");
                       }
               }
               if (ok)
               {
                       bool fileBase128 = (db.Header.flags & TAMES_FLAG_BASE128) != 0;
                       if (fileBase128 != gTamesBase128)
                       {
                               printf("tames format mismatch\r\n");
                               db.Clear();
                               ok = false;
                       }
                       else if ((db.Header.flags >> TAMES_RANGE_SHIFT) != gRange)
                       {
                               printf("loaded tames have different range, they cannot be used, clear\r\n");
                               db.Clear();
                               tamesRangeMismatch = true;
                               printf("WARNING: tames cleared due to range mismatch, continuing without precomputed tames\r\n");
                       }
                       else
                               printf("tames loaded\r\n");
               }
               if (!ok)
               {
                       printf("tames loading failed\r\n");
                       printf("WARNING: tames loading failed, continuing without precomputed tames\r\n");
               }
       }

        if (gGenMode && gTamesFileName[0])
        {
                gTamesWriter = TamesRecordWriterOpen(gTamesFileName, gTamesBase128, sizeof(DBRec));
                if (!gTamesWriter)
                {
                        printf("tames writer open failed\r\n");
                        return false;
                }
        }

        SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;
//prepare jumps
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(Range / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
            EcJumps1[i].p = ec.MultiplyG_GLV(EcJumps1[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
            EcJumps2[i].p = ec.MultiplyG_GLV(EcJumps2[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
            EcJumps3[i].p = ec.MultiplyG_GLV(EcJumps3[i].dist);
	}
	SetRndSeed(GetTickCount64());

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
    Pnt_HalfRange = ec.MultiplyG_GLV(Int_HalfRange);
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	Int_TameOffset.Set(1);
	Int_TameOffset.ShiftLeft(Range - 1);
	EcInt tt;
	tt.Set(1);
	tt.ShiftLeft(Range - 5); //half of tame range width
	Int_TameOffset.Sub(tt);
	gPntToSolve = PntToSolve;

//prepare GPUs
        int gpuDP = gMultiDP ? (DP - gDpCoarseOffset) : DP;
        for (int i = 0; i < GpuCnt; i++)
                if (!GpuKangs[i]->Prepare(PntToSolve, Range, gpuDP, EcJumps1, EcJumps2, EcJumps3, gPhiFold))
                {
                        GpuKangs[i]->Failed = true;
                        printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
                }

	u64 tm0 = GetTickCount64();
	printf("GPUs started...\r\n");

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT];
#else
	pthread_t thr_handles[MAX_GPU_CNT];
#endif

	u32 ThreadID;
	gSolved = false;
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(10);
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);
			tm_stats = GetTickCount64();
		}

                if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
                {
                        gIsOpsLimit = true;
                        printf("Operations limit reached: %llu/%.0f ops. Tames range mismatch: %s\r\n", PntTotalOps, MaxTotalOps, tamesRangeMismatch ? "yes" : "no");
                        break;
                }
	}

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(10);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}

        if (gIsOpsLimit)
        {
                printf("Operations limit reached: %llu/%.0f ops. Tames range mismatch: %s. Search aborted before finding the key\r\n", PntTotalOps, MaxTotalOps, tamesRangeMismatch ? "yes" : "no");
                if (gGenMode && gTamesWriter)
                {
                        TamesRecordWriterClose(gTamesWriter);
                        gTamesWriter = NULL;
                        printf("tames saved\r\n");
                }
                db.Clear();
                return false;
        }

        double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
        printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
        if (gGenMode && gTamesWriter)
        {
                TamesRecordWriterClose(gTamesWriter);
                gTamesWriter = NULL;
        }
        db.Clear();
        *pk_res = gPrivKey;
        return true;
}
#endif

#if 0
bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 14) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
                if (strcmp(argument, "-tames") == 0)
                {
                        strcpy(gTamesFileName, argv[ci]);
                        ci++;
                }
                else
                if (strcmp(argument, "-max") == 0)
                {
                        double val = atof(argv[ci]);
                        ci++;
                        if (val < 0.001)
                        {
                                printf("error: invalid value for -max option\r\n");
                                return false;
                        }
                        gMax = val;
                }
                else
                if (strcmp(argument, "-base128") == 0)
                {
                        gTamesBase128 = true; // use legacy Base128 tames format
                }
                else if (strcmp(argument, "--phi-fold") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --phi-fold option\r\n"); return false; }
                        gPhiFold = atoi(argv[ci]);
                        if (gPhiFold < 0) gPhiFold = 0;
                        if (gPhiFold > 2) gPhiFold = 2;
                        ci++;
                }
                else if (strcmp(argument, "--multi-dp") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --multi-dp option\r\n"); return false; }
                        gMultiDP = atoi(argv[ci]) != 0; ci++;
                }
                else if (strcmp(argument, "--dp-coarse-offset") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --dp-coarse-offset option\r\n"); return false; }
                        gDpCoarseOffset = atoi(argv[ci]); ci++;
                }
                else if (strcmp(argument, "--bloom-mbits") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --bloom-mbits option\r\n"); return false; }
                        gBloomMBits = atoi(argv[ci]); ci++;
                }
                else if (strcmp(argument, "--bloom-k") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --bloom-k option\r\n"); return false; }
                        gBloomK = atoi(argv[ci]); ci++;
                }
                else
                {
                        printf("error: unknown option %s\r\n", argument);
                        return false;
                }
        }
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	printf("********************************************************************************\r\n");
	printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
	printf("DEBUG MODE\r\n\r\n");
#endif

	InitEc();
	gDP = 0;
        gRange = 0;
        gStartSet = false;
        gTamesFileName[0] = 0;
        gTamesBase128 = false;
        gMax = 0.0;
        gGenMode = false;
        gIsOpsLimit = false;
        gPhiFold = 1;
        memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
        if (!ParseCommandLine(argc, argv))
                return 0;

        if (gMultiDP)
                gBloom.Init(gBloomMBits, gBloomK);

	InitGpus();

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

        cudaMallocManaged((void**)&pPntList, MAX_CNT_LIST * GPU_DP_SIZE);
        cudaMallocManaged((void**)&pPntList2, MAX_CNT_LIST * GPU_DP_SIZE);
        TotalOps = 0;
        TotalSolved = 0;
        gTotalErrors = 0;
        IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
                    PntOfs = ec.MultiplyG_GLV(gStart);
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}

		char sx[100], sy[100];
		gPubKey.x.GetHexStr(sx);
		gPubKey.y.GetHexStr(sy);
		printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
		gStart.GetHexStr(sx);
		printf("Offset: %s\r\n", sx);

                if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
                {
                        if (gIsOpsLimit)
                                printf("Search stopped: operations cap hit before locating the key\r\n");
                        else
                                printf("Key not found\r\n");
                        goto label_end;
                }
		pk_found.AddModP(gStart);
            EcPoint tmp = ec.MultiplyG_GLV(pk_found);
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		//solve points, show K
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;

			//generate random pk
			pk.RndBits(gRange);
                    PntToSolve = ec.MultiplyG_GLV(pk);

                        if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
                        {
                                if (gIsOpsLimit)
                                        printf("Benchmark stopped: operations cap hit before locating the key\r\n");
                                else
                                        printf("Benchmark stopped: key not found\r\n");
                                break;
                        }
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
			}
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
        db.CloseMapped();
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
        cudaFree(pPntList2);
        cudaFree(pPntList);
}
#endif

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
	if (speed)
		exp_sec = (u64)((exp_ops / 1000000) / speed); //in sec
	u64 exp_days = exp_sec / (3600 * 24);
	int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

	u64 sec = (GetTickCount64() - tm_start) / 1000;
	u64 days = sec / (3600 * 24);
	int hours = (int)(sec - days * (3600 * 24)) / 3600;
	int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;
	 
	printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm/%llud:%02dh:%02dm\r\n", gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "), speed, gTotalErrors, db.GetBlockCnt()/1000, est_dps_cnt/1000, days, hours, min, exp_days, exp_hours, exp_min);
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 14) || (DP > 60)) 
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
        double MaxTotalOps = 0.0;
        if (gMax > 0)
        {
                MaxTotalOps = gMax * ops;
                double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
                ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
                ram_max /= (1024 * 1024 * 1024); //GB
                printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
                if (MaxTotalOps < ops * 0.1)
                        printf("WARNING: MaxTotalOps is set very low and the search may stop before finding the key\r\n");
        }

        u64 total_kangs = GpuKangs[0]->CalcKangCnt();
        for (int i = 1; i < GpuCnt; i++)
                total_kangs += GpuKangs[i]->CalcKangCnt();
        double path_single_kang = ops / total_kangs;
        double DPs_per_kang = path_single_kang / dp_val;
        printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

       bool tamesRangeMismatch = false;
       if (!gGenMode && gTamesFileName[0])
       {
               printf("load tames...\r\n");
               bool ok = false;
               if (gTamesBase128)
               {
                       ok = db.LoadFromFileBase128(gTamesFileName);
                       if (ok)
                               printf("Base128 tames cannot be memory-mapped and require full in-memory decoding.\r\n");
                       else
                               printf("Base128 tames loading failed\r\n");
               }
               else
               {
                       ok = db.OpenMapped(gTamesFileName);
                       if (!ok)
                       {
                               printf("memory-mapped tames failed, loading into RAM...\r\n");
                               ok = db.LoadFromFile(gTamesFileName);
                               if (!ok)
                                       printf("binary tames loading failed\r\n");
                       }
               }
               if (ok)
               {
                       bool fileBase128 = (db.Header.flags & TAMES_FLAG_BASE128) != 0;
                       if (fileBase128 != gTamesBase128)
                       {
                               printf("tames format mismatch\r\n");
                               db.Clear();
                               ok = false;
                       }
                       else if ((db.Header.flags >> TAMES_RANGE_SHIFT) != gRange)
                       {
                               printf("loaded tames have different range, they cannot be used, clear\r\n");
                               db.Clear();
                               tamesRangeMismatch = true;
                               printf("WARNING: tames cleared due to range mismatch, continuing without precomputed tames\r\n");
                       }
                       else
                               printf("tames loaded\r\n");
               }
               if (!ok)
               {
                       printf("tames loading failed\r\n");
                       printf("WARNING: tames loading failed, continuing without precomputed tames\r\n");
               }
       }

	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;
//prepare jumps
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(Range / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
            EcJumps1[i].p = ec.MultiplyG_GLV(EcJumps1[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
            EcJumps2[i].p = ec.MultiplyG_GLV(EcJumps2[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
            EcJumps3[i].p = ec.MultiplyG_GLV(EcJumps3[i].dist);
	}
	SetRndSeed(GetTickCount64());

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
    Pnt_HalfRange = ec.MultiplyG_GLV(Int_HalfRange);
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	Int_TameOffset.Set(1);
	Int_TameOffset.ShiftLeft(Range - 1);
	EcInt tt;
	tt.Set(1);
	tt.ShiftLeft(Range - 5); //half of tame range width
	Int_TameOffset.Sub(tt);
	gPntToSolve = PntToSolve;

//prepare GPUs
        for (int i = 0; i < GpuCnt; i++)
                if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3, gPhiFold))
                {
                        GpuKangs[i]->Failed = true;
                        printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
                }

	u64 tm0 = GetTickCount64();
	printf("GPUs started...\r\n");

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT];
#else
	pthread_t thr_handles[MAX_GPU_CNT];
#endif

	u32 ThreadID;
	gSolved = false;
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(10);
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);
			tm_stats = GetTickCount64();
		}

                if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
                {
                        gIsOpsLimit = true;
                        printf("Operations limit reached: %llu/%.0f ops. Tames range mismatch: %s\r\n", PntTotalOps, MaxTotalOps, tamesRangeMismatch ? "yes" : "no");
                        break;
                }
	}

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(10);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}

        if (gIsOpsLimit)
        {
                printf("Operations limit reached: %llu/%.0f ops. Tames range mismatch: %s. Search aborted before finding the key\r\n", PntTotalOps, MaxTotalOps, tamesRangeMismatch ? "yes" : "no");
                if (gGenMode)
                {
                        printf("saving tames...\r\n");
                       db.Header.flags = (u16)(TAMES_FLAG_LE | (gRange << TAMES_RANGE_SHIFT) | (gTamesBase128 ? TAMES_FLAG_BASE128 : 0));
                       bool ok;
                       if (gTamesBase128)
                               ok = db.SaveToFileBase128(gTamesFileName);
                       else
                                ok = db.SaveToFile(gTamesFileName);
                        if (ok)
                                printf("tames saved\r\n");
                        else
                                printf("tames saving failed\r\n");
                }
                db.Clear();
                return false;
        }

	double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
	printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
	db.Clear();
	*pk_res = gPrivKey;
	return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 14) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
                if (strcmp(argument, "-tames") == 0)
                {
                        strcpy(gTamesFileName, argv[ci]);
                        ci++;
                }
                else
                if (strcmp(argument, "-max") == 0)
                {
                        double val = atof(argv[ci]);
                        ci++;
                        if (val < 0.001)
                        {
                                printf("error: invalid value for -max option\r\n");
                                return false;
                        }
                        gMax = val;
                }
                else
                if (strcmp(argument, "-base128") == 0)
                {
                        gTamesBase128 = true; // use legacy Base128 tames format
                }
                else if (strcmp(argument, "--phi-fold") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --phi-fold option\r\n"); return false; }
                        gPhiFold = atoi(argv[ci]);
                        if (gPhiFold < 0) gPhiFold = 0;
                        if (gPhiFold > 2) gPhiFold = 2;
                        ci++;
                }
                else if (strcmp(argument, "--multi-dp") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --multi-dp option\r\n"); return false; }
                        gMultiDP = atoi(argv[ci]) != 0; ci++;
                }
                else if (strcmp(argument, "--dp-coarse-offset") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --dp-coarse-offset option\r\n"); return false; }
                        gDpCoarseOffset = atoi(argv[ci]); ci++;
                }
                else if (strcmp(argument, "--bloom-mbits") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --bloom-mbits option\r\n"); return false; }
                        gBloomMBits = atoi(argv[ci]); ci++;
                }
                else if (strcmp(argument, "--bloom-k") == 0)
                {
                        if (ci >= argc) { printf("error: missed value after --bloom-k option\r\n"); return false; }
                        gBloomK = atoi(argv[ci]); ci++;
                }
                else
                {
                        printf("error: unknown option %s\r\n", argument);
                        return false;
                }
        }
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	printf("********************************************************************************\r\n");
	printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
	printf("DEBUG MODE\r\n\r\n");
#endif

	InitEc();
	gDP = 0;
        gRange = 0;
        gStartSet = false;
        gTamesFileName[0] = 0;
        gTamesBase128 = false;
        gMax = 0.0;
        gGenMode = false;
        gIsOpsLimit = false;
        gPhiFold = 1;
        memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
        if (!ParseCommandLine(argc, argv))
                return 0;

        if (gMultiDP)
                gBloom.Init(gBloomMBits, gBloomK);

        InitGpus();

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

        cudaMallocManaged((void**)&pPntList, MAX_CNT_LIST * GPU_DP_SIZE);
        cudaMallocManaged((void**)&pPntList2, MAX_CNT_LIST * GPU_DP_SIZE);
        TotalOps = 0;
        TotalSolved = 0;
        gTotalErrors = 0;
        IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
                    PntOfs = ec.MultiplyG_GLV(gStart);
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}

		char sx[100], sy[100];
		gPubKey.x.GetHexStr(sx);
		gPubKey.y.GetHexStr(sy);
		printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
		gStart.GetHexStr(sx);
		printf("Offset: %s\r\n", sx);

                if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
                {
                        if (gIsOpsLimit)
                                printf("Search stopped: operations cap hit before locating the key\r\n");
                        else
                                printf("Key not found\r\n");
                        goto label_end;
                }
		pk_found.AddModP(gStart);
            EcPoint tmp = ec.MultiplyG_GLV(pk_found);
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		//solve points, show K
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;

			//generate random pk
			pk.RndBits(gRange);
                    PntToSolve = ec.MultiplyG_GLV(pk);

                        if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
                        {
                                if (gIsOpsLimit)
                                        printf("Benchmark stopped: operations cap hit before locating the key\r\n");
                                else
                                        printf("Benchmark stopped: key not found\r\n");
                                break;
                        }
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
			}
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
        db.CloseMapped();
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
        cudaFree(pPntList2);
        cudaFree(pPntList);
}

