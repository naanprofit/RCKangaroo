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
        int ci = 1;
        while (ci < argc && argv[ci][0] == '-')
        {
                char* argument = argv[ci];
                ci++;
                if (strcmp(argument, "-range") == 0)
                {
                        if (ci >= argc)
                        {
                                printf("Usage: %s -range <bits(32-170)> [-base128] <output_file> <count>\n", argv[0]);
                                return 1;
                        }
                        range = atoi(argv[ci]);
                        ci++;
                }
                else if (strcmp(argument, "-base128") == 0)
                {
                        base128 = true;
                }
                else
                {
                        printf("Usage: %s -range <bits(32-170)> [-base128] <output_file> <count>\n", argv[0]);
                        return 1;
                }
        }

        if ((range < 32) || (range > 170) || (argc - ci < 2))
        {
                printf("Usage: %s -range <bits(32-170)> [-base128] <output_file> <count>\n", argv[0]);
                return 1;
        }

        char* out_file = argv[ci];
        errno = 0;
        uint64_t count = strtoull(argv[ci + 1], NULL, 10);
        if (errno == ERANGE)
        {
                printf("Count exceeds supported limits\n");
                return 1;
        }
        TFastBase* db = new TFastBase();
        db->Header[0] = range;

        for (uint64_t i = 0; i < count; i++)
        {
                u8 data[3 + DB_REC_LEN];
                for (int j = 0; j < 3 + DB_REC_LEN; j++)
                        data[j] = rand() & 0xFF;
                db->AddDataBlock(data);
        }

        bool ok = base128 ? db->SaveToFileBase128(out_file) : db->SaveToFile(out_file);
        if (ok)
        {
                printf("Generated %llu tames to %s\n", (unsigned long long)count, out_file);
                delete db;
                return 0;
        }

        printf("Failed to save tames\n");
        delete db;
        return 1;
}
