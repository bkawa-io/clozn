/* Vendored verbatim (plus the extern "C" guard below, added for clozn) from llama.cpp's
   examples/gguf-hash/deps/sha256/sha256.h (public domain). clozn's checkpoint-export payload hash
   (engine/core/serve/checkpoint_codec.hpp) uses this directly so it does not depend on llama.cpp's
   examples/ tree surviving future bootstrap_llama.py re-vendors (examples/ is not built by
   CLOZN_BUILD_GGML at all). CMakeLists.txt compiles sha256.c as C++ (this project never enables a C
   toolchain) -- the extern "C" guard below is therefore load-bearing, not decorative: without it,
   sha256.c's definitions would get C++-mangled names while any C++ caller declares them via plain
   `#include`, and the two would never link. */

/* Sha256.h -- SHA-256 Hash
2010-06-11 : Igor Pavlov : Public domain */

#ifndef __CRYPTO_SHA256_H
#define __CRYPTO_SHA256_H

#include <stdlib.h>
#include <stdint.h>

#define SHA256_DIGEST_SIZE 32

typedef struct sha256_t
{
  uint32_t state[8];
  uint64_t count;
  unsigned char buffer[64];
} sha256_t;

#ifdef __cplusplus
extern "C" {
#endif

void sha256_init(sha256_t *p);
void sha256_update(sha256_t *p, const unsigned char *data, size_t size);
void sha256_final(sha256_t *p, unsigned char *digest);
void sha256_hash(unsigned char *buf, const unsigned char *data, size_t size);

#ifdef __cplusplus
}
#endif

#endif
