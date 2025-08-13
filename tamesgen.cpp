// Simple utility to generate a base128 encoded tames file
// This tool creates random tames records for testing purposes.

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define DB_REC_LEN 32

int main(int argc, char* argv[])
{
        if (argc < 3)
        {
                printf("Usage: %s <output_file> <count>\n", argv[0]);
                return 1;
        }

        int count = atoi(argv[2]);
        TFastBase db;

        for (int i = 0; i < count; i++)
        {
                u8 data[3 + DB_REC_LEN];
                for (int j = 0; j < 3 + DB_REC_LEN; j++)
                        data[j] = rand() & 0xFF;
                db.AddDataBlock(data);
        }

        if (db.SaveToFileBase128(argv[1]))
        {
                printf("Generated %d tames to %s\n", count, argv[1]);
                return 0;
        }

        printf("Failed to save tames\n");
        return 1;
}

