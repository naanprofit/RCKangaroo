// Simple utility to generate a tames file
// This tool creates random tames records for testing purposes.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include "utils.h"

#define DB_REC_LEN 32

int main(int argc, char* argv[])
{
        int range = 0;
        bool base128 = false;
        const char* out_file = NULL;
        uint64_t count = 0;

        for (int i = 1; i < argc; i++)
        {
                char* argument = argv[i];
                if (strcmp(argument, "-range") == 0)
                {
                        if (++i >= argc)
                        {
                                printf("Usage: %s -range <bits(32-170)> [-base128] <output_file> <count>\n", argv[0]);
                                return 1;
                        }
                        range = atoi(argv[i]);
                }
                else if (strcmp(argument, "-base128") == 0)
                {
                        base128 = true;
                }
                else if (!out_file)
                {
                        out_file = argument;
                }
                else if (count == 0)
                {
                        errno = 0;
                        count = strtoull(argument, NULL, 10);
                        if (errno == ERANGE)
                        {
                                printf("Count exceeds supported limits\n");
                                return 1;
                        }
                }
                else
                {
                        printf("Usage: %s -range <bits(32-170)> [-base128] <output_file> <count>\n", argv[0]);
                        return 1;
                }
        }

        if ((range < 32) || (range > 170) || !out_file || count == 0)
        {
                printf("Usage: %s -range <bits(32-170)> [-base128] <output_file> <count>\n", argv[0]);
                return 1;
        }

        TFastBase* db = new TFastBase();
        for (uint64_t i = 0; i < count; i++)
        {
                u8 rec35[3 + DB_REC_LEN];
                for (int j = 0; j < 3 + DB_REC_LEN; j++)
                        rec35[j] = rand() & 0xFF;
                db->AddDataBlock(rec35);
        }
        db->Header.flags = (range << TAMES_RANGE_SHIFT);
        bool ok = base128 ? db->SaveToFileBase128((char*)out_file) : db->SaveToFile((char*)out_file);
        if (!ok)
        {
                printf("Failed to save tames file\n");
                delete db;
                return 1;
        }
        printf("Generated %llu tames to %s\n", (unsigned long long)count, out_file);
        delete db;
        return 0;
}
