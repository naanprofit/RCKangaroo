#pragma once
#include <vector>
#include <cstdint>

class BloomFilter {
    std::vector<uint8_t> bits;
    uint64_t mask;
    int k;
    static inline uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
public:
    BloomFilter() : mask(0), k(0) {}
    void Init(int mbits, int k_){
        uint64_t size = 1ULL << mbits;
        bits.assign((size + 7) / 8, 0);
        mask = size - 1;
        k = k_;
    }
    void Add(uint64_t key){
        uint64_t h = key;
        for(int i=0;i<k;i++){
            h = mix(h + i*0x9e3779b97f4a7c15ULL);
            uint64_t bit = h & mask;
            bits[bit >> 3] |= (1u << (bit & 7));
        }
    }
    bool Test(uint64_t key) const{
        uint64_t h = key;
        for(int i=0;i<k;i++){
            h = mix(h + i*0x9e3779b97f4a7c15ULL);
            uint64_t bit = h & mask;
            if((bits[bit >> 3] & (1u << (bit & 7)))==0)
                return false;
        }
        return true;
    }
};
