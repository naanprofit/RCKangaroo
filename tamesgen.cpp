// Simple utility to generate a base128 encoded tames file
// This tool creates random tames records for testing purposes.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define DB_REC_LEN 32

int main(int argc, char* argv[])
{
        int range = 0;
        int ci = 1;
        while (ci < argc && argv[ci][0] == '-')
        {
                char* argument = argv[ci];
                ci++;
                if (strcmp(argument, "-range") == 0)
                {
                        if (ci >= argc)
                        {
                                printf("Usage: %s -range <bits(32-170)> <output_file> <count>\n", argv[0]);
                                return 1;
                        }
                        range = atoi(argv[ci]);
                        ci++;
                }
                else
                {
                        printf("Usage: %s -range <bits(32-170)> <output_file> <count>\n", argv[0]);
                        return 1;
                }
        }

        if ((range < 32) || (range > 170) || (argc - ci < 2))
        {
                printf("Usage: %s -range <bits(32-170)> <output_file> <count>\n", argv[0]);
                return 1;
        }

        char* out_file = argv[ci];
        int count = atoi(argv[ci + 1]);
        TFastBase* db = new TFastBase();
        db->Header[0] = range;

        for (int i = 0; i < count; i++)
        {
                u8 data[3 + DB_REC_LEN];
                for (int j = 0; j < 3 + DB_REC_LEN; j++)
                        data[j] = rand() & 0xFF;
                db->AddDataBlock(data);
        }

        if (db->SaveToFileBase128(out_file))
        {
                printf("Generated %d tames to %s\n", count, out_file);
                delete db;
                return 0;
        }

        printf("Failed to save tames\n");
        delete db;
        return 1;
}

