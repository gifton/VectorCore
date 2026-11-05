#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Runtime mode retains identical code generation for the leak/clean controls.
 * leak-control-symbols/disassembly and both runtime markers verify malloc/free. */
__attribute__((noinline)) static void control(int clean) {
    volatile unsigned char *block = malloc(4096);
    if (!block) exit(2);
    block[0] = 42;
    printf("LEAK_CONTROL allocated=%p value=%u clean=%d\n", (void *)block, block[0], clean);
    if (clean) free((void *)block);
}
int main(int argc, char **argv) {
    for (int i = 0; i < 8; ++i) {
        control(argc == 2 && strcmp(argv[1], "clean") == 0);
    }
    return 0;
}
