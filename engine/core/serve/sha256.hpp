#pragma once

#include "sha256.h"

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

namespace clozn {

inline std::string sha256_bytes_to_hex(const unsigned char* digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        out << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return out.str();
}

class RunningSha256 {
public:
    RunningSha256() { sha256_init(&context_); }

    void update(const void* data, std::size_t length) {
        sha256_update(&context_, static_cast<const unsigned char*>(data), length);
    }

    template <typename T>
    void update_pod(const T& value) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "update_pod needs a trivially copyable type");
        update(&value, sizeof(T));
    }

    std::string hex_digest() {
        unsigned char digest[SHA256_DIGEST_SIZE];
        sha256_final(&context_, digest);
        return sha256_bytes_to_hex(digest);
    }

private:
    sha256_t context_;
};

}  // namespace clozn
