#define main leak_fixture_unused_main
#include "LeakControl.c"
#undef main
void vectorcoreLeakFixture(int clean) {
    for (int i = 0; i < 8; ++i) control(clean);
}
