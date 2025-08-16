// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include "defs.h"
#include "Ec.h"
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include "utils.h"

// https://en.bitcoin.it/wiki/Secp256k1
EcInt g_P; //FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
EcPoint g_G; //Generator point
EcInt g_Lambda; //GLV eigenvalue
EcInt g_Beta;   //cube root of unity
EcInt g_Lambda2; //lambda squared mod n

#define P_REV	0x00000001000003D1

#ifdef DEBUG_MODE
u8* GTable = NULL; //16x16-bit table
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool parse_u8(const char* s, u8* res)
{
	char cl = toupper(s[1]);
	char ch = toupper(s[0]);
	if (((cl < '0') || (cl > '9')) && ((cl < 'A') || (cl > 'F')))
		return false;
	if (((ch < '0') || (ch > '9')) && ((ch < 'A') || (ch > 'F')))
		return false;
	u8 l = ((cl >= '0') && (cl <= '9')) ? (cl - '0') : (cl - 'A' + 10);
	u8 h = ((ch >= '0') && (ch <= '9')) ? (ch - '0') : (ch - 'A' + 10);
	*res = l + (h << 4);
	return true;
}

bool EcPoint::IsEqual(EcPoint& pnt)
{
	return this->x.IsEqual(pnt.x) && this->y.IsEqual(pnt.y);
}

void EcPoint::LoadFromBuffer64(u8* buffer)
{
	memcpy(x.data, buffer, 32);
	x.data[4] = 0;
	memcpy(y.data, buffer + 32, 32);
	y.data[4] = 0;
}

void EcPoint::SaveToBuffer64(u8* buffer)
{
	memcpy(buffer, x.data, 32);
	memcpy(buffer + 32, y.data, 32);
}

bool EcPoint::SetHexStr(const char* str)
{
	EcPoint res;
	int len = (int)strlen(str);
	if (len < 66)
		return false;
	u8 type, b;
	if (!parse_u8(str, &type))
		return false;
	if ((type < 2) || (type > 4))
		return false;
	if (((type == 2) || (type == 3)) && (len != 66))
		return false;
	if ((type == 4) && (len != 130))
		return false;

	if (len == 66) //compressed
	{
		str += 2;
		for (int i = 0; i < 32; i++)
		{
			if (!parse_u8(str + 2 * i, &b))
				return false;
			((u8*)res.x.data)[31 - i] = b;
		}
		res.y = Ec::CalcY(res.x, type == 2);
		if (!Ec::IsValidPoint(res))
			return false;		
		*this = res;
		return true;
	}
	//uncompressed
	str += 2;
	for (int i = 0; i < 32; i++)
	{
		if (!parse_u8(str + 2 * i, &b))
			return false;
		((u8*)res.x.data)[31 - i] = b;

		if (!parse_u8(str + 2 * i + 64, &b))
			return false;
		((u8*)res.y.data)[31 - i] = b;
	}
	if (!Ec::IsValidPoint(res))
		return false;
	*this = res;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// https://en.bitcoin.it/wiki/Secp256k1
void InitEc()
{
        g_P.SetHexStr("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"); //Fp
        g_G.x.SetHexStr("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"); //G.x
        g_G.y.SetHexStr("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"); //G.y
        g_Lambda.SetHexStr("5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
        g_Beta.SetHexStr("7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE");
        g_Lambda2 = g_Lambda;
        g_Lambda2.MulModN(g_Lambda);
#ifdef DEBUG_MODE
        GTable = (u8*)malloc(16 * 256 * 256 * 64);
	EcPoint pnt = g_G;
	for (int i = 0; i < 16; i++)
	{
		pnt.SaveToBuffer64(GTable + (i * 256 * 256) * 64);
		EcPoint tmp = pnt;
		pnt = Ec::DoublePoint(pnt);
		for (int j = 1; j < 256 * 256 - 1; j++)
		{
			pnt.SaveToBuffer64(GTable + (i * 256 * 256 + j) * 64);
			pnt = Ec::AddPoints(pnt, tmp);
		}
	}
#endif
};

void DeInitEc()
{
#ifdef DEBUG_MODE
	if (GTable)
		free(GTable);
#endif
}

// https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_addition
EcPoint Ec::AddPoints(EcPoint& pnt1, EcPoint& pnt2)
{
        EcPoint res;
        EcInt dx, dy, lambda, lambda2;

        // Handle addition with the point at infinity represented as (0,0).
        if (pnt1.x.IsZero() && pnt1.y.IsZero())
                return pnt2;
        if (pnt2.x.IsZero() && pnt2.y.IsZero())
                return pnt1;

	dx = pnt2.x;
	dx.SubModP(pnt1.x);
	dx.InvModP();

	dy = pnt2.y;
	dy.SubModP(pnt1.y);

	lambda = dy;
	lambda.MulModP(dx);
	lambda2 = lambda;
	lambda2.MulModP(lambda);

	res.x = lambda2;
	res.x.SubModP(pnt1.x);
	res.x.SubModP(pnt2.x);

	res.y = pnt2.x;
	res.y.SubModP(res.x);
	res.y.MulModP(lambda);
	res.y.SubModP(pnt2.y);
	return res;
}

// https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_doubling
EcPoint Ec::DoublePoint(EcPoint& pnt)
{
	EcPoint res;
	EcInt t1, t2, lambda, lambda2;

	t1 = pnt.y;
	t1.AddModP(pnt.y);
	t1.InvModP();

	t2 = pnt.x;
	t2.MulModP(pnt.x);
	lambda = t2;
	lambda.AddModP(t2);
	lambda.AddModP(t2);
	lambda.MulModP(t1);
	lambda2 = lambda;
	lambda2.MulModP(lambda);

	res.x = lambda2;
	res.x.SubModP(pnt.x);
	res.x.SubModP(pnt.x);

	res.y = pnt.x;
	res.y.SubModP(res.x);
	res.y.MulModP(lambda);
	res.y.SubModP(pnt.y);
	return res;
}

//k up to 256 bits
EcPoint Ec::MultiplyG(EcInt& k)
{
	EcPoint res;
	EcPoint t = g_G;
	bool first = true;
	int n = 3;
	while ((n >= 0) && !k.data[n])
		n--;
	if (n < 0)
		return res; //error
	int index;                     
	_BitScanReverse64((DWORD*)&index, k.data[n]);
	for (int i = 0; i <= 64 * n + index; i++)
	{
		u8 v = (k.data[i / 64] >> (i % 64)) & 1;	
		if (v)
		{
			if (first)
			{
				first = false;
				res = t;
			}
			else
				res = Ec::AddPoints(res, t);
		}
		t = Ec::DoublePoint(t);
	}
	return res;
}

// GLV-based multiplication using endomorphism
EcPoint Ec::MultiplyG_GLV(EcInt& k)
{
        using boost::multiprecision::cpp_int;

        static const cpp_int order("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        static const cpp_int lambda("0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
        static const cpp_int g1("0x3086D221A7D46BCDE86C90E49284EB153DAA8A1471E8CA7FE893209A45DBB031");
        static const cpp_int g2("0xE4437ED6010E88286F547FA90ABFE4C4221208AC9DF506C61571B4AE8AC47F71");
        // Precomputed constants from the secp256k1 endomorphism lattice.
        // `minus_b1` is actually the second basis vector b2, while `minus_b2`
        // is -b1 mod n. These names are kept for backward compatibility.
        static const cpp_int minus_b1("0xE4437ED6010E88286F547FA90ABFE4C3");
        static const cpp_int minus_b2("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE8A280AC50774346DD765CDA83DB1562C");

        auto to_cpp = [](const EcInt& a) {
                cpp_int r = 0;
                for (int i = 3; i >= 0; --i)
                {
                        r <<= 64;
                        r += a.data[i];
                }
                return r;
        };
        auto from_cpp = [](const cpp_int& x) {
                EcInt r;
                cpp_int t = x;
                for (int i = 0; i < 4; ++i)
                {
                        r.data[i] = (u64)(t & 0xFFFFFFFFFFFFFFFFULL);
                        t >>= 64;
                }
                r.data[4] = 0;
                return r;
        };

        cpp_int K = to_cpp(k);
        cpp_int c1 = (K * g1 + (cpp_int(1) << 383)) >> 384;
        cpp_int c2 = (K * g2 + (cpp_int(1) << 383)) >> 384;
        cpp_int r2 = c1 * minus_b1 + c2 * minus_b2;
        cpp_int r1 = K - r2 * lambda;

        bool neg1 = r1 < 0;
        bool neg2 = r2 < 0;
        if (neg1) r1 = -r1;
        if (neg2) r2 = -r2;
        r1 %= order;
        r2 %= order;

        EcInt k1 = from_cpp(r1);
        EcInt k2 = from_cpp(r2);

        EcPoint p1 = MultiplyG(k1);
        if (neg1 && !p1.y.IsZero()) p1.y.NegModP();
        EcPoint p2 = MultiplyG(k2);
        if (neg2 && !p2.y.IsZero()) p2.y.NegModP();
        p2.x.MulModP(g_Beta);

        return AddPoints(p1, p2);
}

#ifdef DEBUG_MODE
//uses gTable (16x16-bit) to speedup calculation
EcPoint Ec::MultiplyG_Fast(EcInt& k)
{
	int i;
	u16 b;
	EcPoint pnt, res;
	for (i = 0; i < 16; i++)
	{
		b = k.GetU16(i);
		if (b)
			break;
	}
	if (i >= 16)
		return pnt;
	if (i < 16)
	{
		res.LoadFromBuffer64(GTable + (256 * 256 * i + (b - 1)) * 64);
		i++;
	}
	while (i < 16)
	{
		b = k.GetU16(i);
		if (b)
		{
			pnt.LoadFromBuffer64(GTable + (256 * 256 * i + (b - 1)) * 64);
			res = AddPoints(res, pnt);
		}
		i++;
	}
	return res;
}
#endif

EcInt Ec::CalcY(EcInt& x, bool is_even)
{
	EcInt res;
	EcInt tmp;
	tmp.Set(7);
	res = x;
	res.MulModP(x);
	res.MulModP(x);
	res.AddModP(tmp);
	res.SqrtModP();
	if ((res.data[0] & 1) == is_even)
		res.NegModP();
	return res;
}

bool Ec::IsValidPoint(EcPoint& pnt)
{
	EcInt x, y, seven;
	seven.Set(7);
	x = pnt.x;
	x.MulModP(pnt.x);
	x.MulModP(pnt.x);
	x.AddModP(seven);
	y = pnt.y;
	y.MulModP(pnt.y);
	return x.IsEqual(y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Mul256_by_64(u64* input, u64 multiplier, u64* result)
{
	u64 h1, h2;
	result[0] = _umul128(input[0], multiplier, &h1);
	u8 carry = _addcarry_u64(0, _umul128(input[1], multiplier, &h2), h1, result + 1);
	carry = _addcarry_u64(carry, _umul128(input[2], multiplier, &h1), h2, result + 2);
	carry = _addcarry_u64(carry, _umul128(input[3], multiplier, &h2), h1, result + 3);
	_addcarry_u64(carry, 0, h2, result + 4);
}

void Mul320_by_64(u64* input, u64 multiplier, u64* result)
{
	u64 h1, h2;
	result[0] = _umul128(input[0], multiplier, &h1);
	u8 carry = _addcarry_u64(0, _umul128(input[1], multiplier, &h2), h1, result + 1);
	carry = _addcarry_u64(carry, _umul128(input[2], multiplier, &h1), h2, result + 2);
	carry = _addcarry_u64(carry, _umul128(input[3], multiplier, &h2), h1, result + 3);
	_addcarry_u64(carry, _umul128(input[4], multiplier, &h1), h2, result + 4);
}

void Add320_to_256(u64* in_out, u64* val)
{
	u8 c = _addcarry_u64(0, in_out[0], val[0], in_out);
	c = _addcarry_u64(c, in_out[1], val[1], in_out + 1);
	c = _addcarry_u64(c, in_out[2], val[2], in_out + 2);
	c = _addcarry_u64(c, in_out[3], val[3], in_out + 3);
	_addcarry_u64(c, 0, val[4], in_out + 4);
}

EcInt::EcInt()
{
	SetZero();
}

void EcInt::Assign(EcInt& val)
{
	memcpy(data, val.data, sizeof(data));
}

void EcInt::Set(u64 val)
{
	SetZero();
	data[0] = val;
}

void EcInt::SetZero()
{
	memset(data, 0, sizeof(data));
}

bool EcInt::SetHexStr(const char* str)
{
	SetZero();
	int len = (int)strlen(str);
	if (len > 64)
		return false;
	char s[64];
	memset(s, '0', 64);
	memcpy(s + 64 - len, str, len);
	for (int i = 0; i < 32; i++)
	{
		int n = 62 - 2 * i;
		u8 b;
		if (!parse_u8(s + n, &b))
			return false;
		((u8*)data)[i] = b;
	}
	return true;
}

void EcInt::GetHexStr(char* str)
{
	for (int i = 0; i < 32; i++)
		sprintf(str + 2 * i, "%02X", ((u8*)data)[31 - i]);
	str[64] = 0;
}

u16 EcInt::GetU16(int index)
{
	return (u16)(data[index / 4] >> (16 * (index % 4)));
}

//returns carry
bool EcInt::Add(EcInt& val)
{
	u8 c = _addcarry_u64(0, data[0], val.data[0], data + 0);
	c = _addcarry_u64(c, data[1], val.data[1], data + 1);
	c = _addcarry_u64(c, data[2], val.data[2], data + 2);
	c = _addcarry_u64(c, data[3], val.data[3], data + 3);
	return _addcarry_u64(c, data[4], val.data[4], data + 4) != 0;
}

//returns carry
bool EcInt::Sub(EcInt& val)
{
	u8 c = _subborrow_u64(0, data[0], val.data[0], data + 0);
	c = _subborrow_u64(c, data[1], val.data[1], data + 1);
	c = _subborrow_u64(c, data[2], val.data[2], data + 2);
	c = _subborrow_u64(c, data[3], val.data[3], data + 3);
	return _subborrow_u64(c, data[4], val.data[4], data + 4) != 0;
}

void EcInt::Neg()
{
	u8 c = _subborrow_u64(0, 0, data[0], data + 0);
	c = _subborrow_u64(c, 0, data[1], data + 1);
	c = _subborrow_u64(c, 0, data[2], data + 2);
	c = _subborrow_u64(c, 0, data[3], data + 3);
	_subborrow_u64(c, 0, data[4], data + 4);
}

void EcInt::Neg256()
{
	u8 c = _subborrow_u64(0, 0, data[0], data + 0);
	c = _subborrow_u64(c, 0, data[1], data + 1);
	c = _subborrow_u64(c, 0, data[2], data + 2);
	c = _subborrow_u64(c, 0, data[3], data + 3);
	data[4] = 0;
}

bool EcInt::IsLessThanU(EcInt& val)
{
	int i = 4;
	while (i >= 0)
	{
		if (data[i] != val.data[i])
			break;
		i--;
	}
	if (i < 0)
		return false;
	return data[i] < val.data[i];
}

bool EcInt::IsLessThanI(EcInt& val)
{
	if ((data[4] >> 63) && !(val.data[4] >> 63))
		return true;
	if (!(data[4] >> 63) && (val.data[4] >> 63))
		return false;

	int i = 4;
	while (i >= 0)
	{
		if (data[i] != val.data[i])
			break;
		i--;
	}
	if (i < 0)
		return false;
	return data[i] < val.data[i];
}

bool EcInt::IsEqual(EcInt& val)
{
	return memcmp(val.data, this->data, 40) == 0;
}

bool EcInt::IsZero()
{
	return ((data[0] == 0) && (data[1] == 0) && (data[2] == 0) && (data[3] == 0) && (data[4] == 0));
}

void EcInt::AddModP(EcInt& val)
{
	Add(val);
	if (!IsLessThanU(g_P)) 
		Sub(g_P);
}

void EcInt::SubModP(EcInt& val)
{
	if (Sub(val))
		Add(g_P);
}

//assume value < P
void EcInt::NegModP()
{
	Neg();
	Add(g_P);
}

void EcInt::ShiftRight(int nbits)
{
	int offset = nbits / 64;
	if (offset)
	{
		for (int i = 0; i < 5 - offset; i++)
			data[i] = data[i + offset];
		for (int i = 5 - offset; i < 5; i++)
			data[i] = 0;
		nbits -= 64 * offset;
	}
	data[0] = __shiftright128(data[0], data[1], nbits);
	data[1] = __shiftright128(data[1], data[2], nbits);
	data[2] = __shiftright128(data[2], data[3], nbits);
	data[3] = __shiftright128(data[3], data[4], nbits);
	data[4] = ((i64)data[4]) >> nbits;
}

void EcInt::ShiftLeft(int nbits)
{
	int offset = nbits / 64;
	if (offset)
	{
		for (int i = 4; i >= offset; i--)
			data[i] = data[i - offset];
		for (int i = offset - 1; i >= 0; i--)
			data[i] = 0;
		nbits -= 64 * offset;
	}
	data[4] = __shiftleft128(data[3], data[4], nbits);
	data[3] = __shiftleft128(data[2], data[3], nbits);
	data[2] = __shiftleft128(data[1], data[2], nbits);
	data[1] = __shiftleft128(data[0], data[1], nbits);
	data[0] = data[0] << nbits;
}

void EcInt::MulModP(EcInt& val)
{
	u64 buff[8], tmp[5], h;
	//calc 512 bits
	Mul256_by_64(val.data, data[0], buff);
	Mul256_by_64(val.data, data[1], tmp);
	Add320_to_256(buff + 1, tmp);
	Mul256_by_64(val.data, data[2], tmp);
	Add320_to_256(buff + 2, tmp);
	Mul256_by_64(val.data, data[3], tmp);
	Add320_to_256(buff + 3, tmp);
	//fast mod P
	Mul256_by_64(buff + 4, P_REV, tmp);
	u8 c = _addcarry_u64(0, buff[0], tmp[0], buff);
	c = _addcarry_u64(c, buff[1], tmp[1], buff + 1);
	c = _addcarry_u64(c, buff[2], tmp[2], buff + 2);
	tmp[4] += _addcarry_u64(c, buff[3], tmp[3], buff + 3);
	c = _addcarry_u64(0, buff[0], _umul128(tmp[4], P_REV, &h), data);
	c = _addcarry_u64(c, buff[1], h, data + 1);
	c = _addcarry_u64(c, 0, buff[2], data + 2);
	data[4] = _addcarry_u64(c, buff[3], 0, data + 3);
        while (data[4])
                Sub(g_P);
}

void EcInt::MulModN(const EcInt& val)
{
        using boost::multiprecision::cpp_int;
        static const cpp_int order("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        cpp_int a = 0, b = 0;
        for (int i = 3; i >= 0; --i)
        {
                a <<= 64; a += data[i];
                b <<= 64; b += val.data[i];
        }
        cpp_int r = (a * b) % order;
        for (int i = 0; i < 4; ++i)
        {
                data[i] = (u64)(r & 0xFFFFFFFFFFFFFFFFULL);
                r >>= 64;
        }
        data[4] = 0;
}

void EcInt::MulLambdaN()
{
        MulModN(g_Lambda);
}

void EcInt::MulLambda2N()
{
        MulModN(g_Lambda2);
}

void EcInt::Mul_u64(EcInt& val, u64 multiplier)
{
	Assign(val);
	Mul320_by_64(data, (u64)multiplier, data);
}

void EcInt::Mul_i64(EcInt& val, i64 multiplier)
{
	Assign(val);
	if (multiplier < 0)
	{
		Neg();
		multiplier = -multiplier;
	}
	Mul320_by_64(data, (u64)multiplier, data);
}

#define APPLY_DIV_SHIFT() kbnt -= index; val >>= index; matrix[0] <<= index; matrix[1] <<= index; 
	
// https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494
//a bit tricky
void DIV_62(i64& kbnt, i64 modp, i64 val, i64* matrix)
{
	int index, cnt;
	_BitScanForward64((DWORD*)&index, val | 0x4000000000000000);
	APPLY_DIV_SHIFT();
	cnt = 62 - index;
	while (cnt > 0)
	{
		if (kbnt < 0)
		{
			kbnt = -kbnt;
			i64 tmp = -modp; modp = val; val = tmp;
			tmp = -matrix[0]; matrix[0] = matrix[2]; matrix[2] = tmp;
			tmp = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tmp;
		}
		int thr = cnt;
		if ((kbnt + 1) < cnt)
			thr = (int)(kbnt + 1);
		i64 mul = (-modp * val) & ((UINT64_MAX >> (64 - thr)) & 0x07);
		val += (modp * mul);
		matrix[2] += (matrix[0] * mul);
		matrix[3] += (matrix[1] * mul);
		_BitScanForward64((DWORD*)&index, val | (1ull << cnt));
		APPLY_DIV_SHIFT();
		cnt -= index;
	}
}

void EcInt::InvModP()
{
	i64 matrix[4];
	EcInt result, a, tmp, tmp2;
	EcInt modp, val;
	i64 kbnt = -1;
	matrix[1] = matrix[2] = 0;
	matrix[0] = matrix[3] = 1;	
	DIV_62(kbnt, g_P.data[0], data[0], matrix);
	modp.Mul_i64(g_P, matrix[0]);
	tmp.Mul_i64(*this, matrix[1]);
	modp.Add(tmp);
	modp.ShiftRight(62);
	val.Mul_i64(g_P, matrix[2]);
	tmp.Mul_i64(*this, matrix[3]);
	val.Add(tmp);
	val.ShiftRight(62);
	if (matrix[1] >= 0)
		result.Set(matrix[1]);
	else
	{
		result.Set(-matrix[1]);
		result.Neg();
	}
	if (matrix[3] >= 0)
		a.Set(matrix[3]);
	else
	{ 
		a.Set(-matrix[3]);
		a.Neg();
	}
	Mul320_by_64(g_P.data, (result.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp.data);
	result.Add(tmp);
	result.ShiftRight(62);
	Mul320_by_64(g_P.data, (a.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp.data);
	a.Add(tmp);
	a.ShiftRight(62);
	
	while (val.data[0] || val.data[1] || val.data[2] || val.data[3])
	{
		matrix[1] = matrix[2] = 0;
		matrix[0] = matrix[3] = 1;	
		DIV_62(kbnt, modp.data[0], val.data[0], matrix);
		tmp.Mul_i64(modp, matrix[0]);
		tmp2.Mul_i64(val, matrix[1]);
		tmp.Add(tmp2);
		tmp2.Mul_i64(val, matrix[3]);
		val.Mul_i64(modp, matrix[2]);
		val.Add(tmp2);
		val.ShiftRight(62);
		modp = tmp;
		modp.ShiftRight(62);
		tmp.Mul_i64(result, matrix[0]);
		tmp2.Mul_i64(a, matrix[1]);
		tmp.Add(tmp2);
		tmp2.Mul_i64(a, matrix[3]);
		a.Mul_i64(result, matrix[2]);
		a.Add(tmp2);
		Mul320_by_64(g_P.data, (a.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp2.data);
		a.Add(tmp2);
		a.ShiftRight(62);	
		Mul320_by_64(g_P.data, (tmp.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp2.data);
		result = tmp;
		result.Add(tmp2);
		result.ShiftRight(62);
	}
	Assign(result);
	if (modp.data[4] >> 63)
	{
		Neg();
		modp.Neg();	
	}

	if (modp.data[0] == 1) 
	{
		if (data[4] >> 63)
			Add(g_P);
		if (data[4] >> 63)
			Add(g_P);
		if (!IsLessThanU(g_P))
			Sub(g_P);
		if (!IsLessThanU(g_P))
			Sub(g_P);
	}
	else
		SetZero(); //error
}

// x = a^ { (p + 1) / 4 } mod p
void EcInt::SqrtModP()
{
	EcInt one, res;
	one.Set(1);
	EcInt exp = g_P;
	exp.Add(one);
	exp.ShiftRight(2);
	res.Set(1);
	EcInt cur = *this;
	while (!exp.IsZero())
	{
		if (exp.data[0] & 1)
			res.MulModP(cur);
		EcInt tmp = cur;
		tmp.MulModP(cur);
		cur = tmp;
		exp.ShiftRight(1);
	}
	*this = res;
}

std::mt19937_64 rng;
CriticalSection cs_rnd;

void SetRndSeed(u64 seed)
{
	rng.seed(seed);
}

void EcInt::RndBits(int nbits)
{
	SetZero();
	if (nbits > 256)
		nbits = 256;
	cs_rnd.Enter();
	for (int i = 0; i < (nbits + 63) / 64; i++)
		data[i] = rng();
	cs_rnd.Leave();
	data[nbits / 64] &= (1ull << (nbits % 64)) - 1;
}

//up to 256 bits only
void EcInt::RndMax(EcInt& max)
{
	SetZero();
	int n = 3;
	while ((n >= 0) && !max.data[n])
		n--;
	if (n < 0)
		return;
	u64 val = max.data[n];
	int k = 0;
	while ((val & 0x8000000000000000) == 0)
	{
		val <<= 1;
		k++;
	}
	int bits = 64 * n + (64 - k);
	RndBits(bits);
	while (!IsLessThanU(max)) // :)
		RndBits(bits);
}





