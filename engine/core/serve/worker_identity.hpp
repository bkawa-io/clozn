#pragma once

#include <array>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

namespace clozn {

inline std::string make_worker_generation_id() {
    std::random_device source;
    std::array<unsigned char, 16> bytes{};
    for (unsigned char& byte : bytes) {
        byte = static_cast<unsigned char>(source() & 0xffU);
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : bytes) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

}  // namespace clozn
