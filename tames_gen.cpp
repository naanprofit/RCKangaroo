// Utility to build a tames file from base128 encoded input
// Each line of the input should contain a base128 representation of
// a 35-byte record (3-byte prefix + 32 bytes of data).

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include "utils.h"

static bool base128_decode(const std::string &s, uint8_t *out, size_t outlen)
{
    std::vector<uint8_t> tmp(outlen, 0);
    for (unsigned char ch : s)
    {
        if (ch > 127)
            return false;
        unsigned int carry = ch;
        for (int i = (int)outlen - 1; i >= 0; --i)
        {
            unsigned int v = tmp[i] * 128u + carry;
            tmp[i] = v & 0xFFu;
            carry = v >> 8;
        }
    }
    std::memcpy(out, tmp.data(), outlen);
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: tames_gen <input_base128.txt> <output_tames.dat>\n";
        return 1;
    }

    std::ifstream fin(argv[1]);
    if (!fin.is_open())
    {
        std::cerr << "Cannot open input file\n";
        return 1;
    }

    TFastBase db;
    std::string line;
    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;
        uint8_t buf[35];
        std::memset(buf, 0, sizeof(buf));
        if (!base128_decode(line, buf, sizeof(buf)))
        {
            std::cerr << "Invalid base128 line, skipping\n";
            continue;
        }
        db.AddDataBlock(buf, -1);
    }

    fin.close();

    if (db.SaveToFile(argv[2]))
    {
        std::cout << "Tames file saved to " << argv[2] << "\n";
        return 0;
    }
    std::cerr << "Failed to save tames file\n";
    return 1;
}

