(c) 2024, RetiredCoder (RC)

RCKangaroo is free and open-source (GPLv3).
This software demonstrates efficient GPU implementation of SOTA Kangaroo method for solving ECDLP. 
It's part #3 of my research, you can find more details here: https://github.com/RetiredC

Discussion thread: https://bitcointalk.org/index.php?topic=5517607

<b>Features:</b>

- Lowest K=1.15, it means 1.8 times less required operations compared to classic method with K=2.1, also it means that you need 1.8 times less memory to store DPs.
- Fast, about 8GKeys/s on RTX 4090, 4GKeys/s on RTX 3090.
- Keeps DP overhead as small as possible.
- Supports ranges up to 170 bits.
- Both Windows and Linux are supported.

<b>Limitations:</b>

- No advanced features like networking, saving/loading DPs, etc.

<b>Command line parameters:</b>

<b>-gpu</b>		which GPUs are used, for example, "035" means that GPUs #0, #3 and #5 are used. If not specified, all available GPUs are used. 

<b>-pubkey</b>		public key to solve, both compressed and uncompressed keys are supported. If not specified, software starts in benchmark mode and solves random keys. 

<b>-start</b>		start offset of the key, in hex. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 start offset is "1000000000000000000000". 

<b>-range</b>		bit range of private the key. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 bit range is "84" (84 bits). Must be in range 32...170. 

<b>-dp</b>		DP bits. Must be in range 14...60. Low DP bits values cause larger DB but reduces DP overhead and vice versa. 

<b>-max</b>		option to limit max number of operations. For example, value 5.5 limits number of operations to 5.5 * 1.15 * sqrt(range), software stops when the limit is reached. 

<b>-tames</b>           filename with tames. If file not found, software generates tames (option "-max" is required) and saves them to the file. If the file is found, software memory-maps the binary tames to speed up access and automatically falls back to in-memory loading or Base128 if needed.

<b>-base128</b>        when generating tames, save the output file in Base128 format instead of the default binary format.

<b>--phi-fold</b>      fold points under the secp256k1 endomorphism φ when generating tames. Example: "--phi-fold 2" uses both P and φ(P) to shrink the tame set and improve cache locality.

<b>--multi-dp</b>      allow multiple distinguished-point tables (1 to enable, 0 to disable). Disabling may save memory at the cost of more DP collisions.

<b>--bloom-mbits</b>   size of the Bloom filter in bits, expressed as a power of two. "--bloom-mbits 27" allocates 2^27 bits (~16 MB).

<b>--bloom-k</b>       number of hash functions used by the Bloom filter. Higher values reduce false positives but increase hashing cost.

When public key is solved, software displays it and also writes it to "RESULTS.TXT" file. 

Sample command line for puzzle #85:

RCKangaroo.exe -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a

Sample command to generate tames:

RCKangaroo.exe -dp 16 -range 76 -tames tames76.dat -max 10
You can also quickly generate a tames file using the helper tool (count parameter supports 64-bit values):

./tamesgen -range 76 tames76.dat 10000000000
Add the <code>-base128</code> flag to emit a Base128 encoded file instead of the default binary format.

Then you can restart software with same parameters to see less K in benchmark mode or add "-tames tames76.dat" to solve some public key in 76-bit range faster.

Examples of additional options:

RCKangaroo.exe -dp 16 -range 76 -tames tames76.dat -max 10 -base128
  save the generated tames in Base128 format.

RCKangaroo.exe -dp 16 -range 76 -tames t76.dat -max 10 --phi-fold 2
  fold each point with its φ(P) image to reduce the table size.

RCKangaroo.exe -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a --multi-dp 1 --bloom-mbits 27 --bloom-k 4
  enable multiple DP tables and tune the Bloom filter parameters.

Binary tames load much faster because the OS can memory-map them directly. Base128 files are smaller but must be decoded at
startup. Switch between the formats by regenerating the file with or without the <code>-base128</code> flag.

<b>Binary tames header format:</b>

Binary files begin with the following structure (little-endian unless the flag bit 0 is cleared):
magic="PMAP" | version=1 | stride=DP record size | flags | rec_cnt
Bits 8..15 of <code>flags</code> encode the bit range shift. See <code>utils.h</code> for details.

<b>Some notes:</b>

Fastest ECDLP solvers will always use SOTA/SOTA+ method, as it's 1.4/1.5 times faster and requires less memory for DPs compared to the best 3-way kangaroos with K=1.6. 
Even if you already have a faster implementation of kangaroo jumps, incorporating SOTA method will improve it further. 
While adding the necessary loop-handling code will cause you to lose about 5–15% of your current speed, the SOTA method itself will provide a 40% performance increase. 
Overall, this translates to roughly a 25% net improvement, which should not be ignored if your goal is to build a truly fast solver. 


<b>Changelog:</b>

v3.0:

- added "-tames" and "-max" options.
- fixed some bugs.

v2.0:

- added support for 30xx, 20xx and 1xxx cards.
- some minor changes.

v1.1:

- added ability to start software on 30xx cards.

v1.0:

- initial release.
