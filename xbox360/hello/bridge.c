// The C side of the Rust test title, built with xsdk's GCC (-mcall-xsdk).
//
// Rust's code is LLVM's 32-bit SysV PowerPC, which xsdk's calling convention is not: xsdk passes arguments in 8-byte
// slots from a 64-byte parameter save area every caller leaves at 16(r1), and keeps r14-r31 whole, 64 bits. So the
// two never call each other directly (xbox360/README.md, "The calling convention boundary"):
//
// - Rust's object has every undefined symbol X renamed __rs2c_X (tools/rust_object.py), and the RS2C thunks below are
//   the only definitions of those names. A Rust call to anything without a thunk fails the link.
// - A thunk gives the C callee a frame of xsdk's shape: 16 fixed bytes, then the 64-byte parameter save area a
//   varargs or address-taking callee may spill r3-r10 into, which in Rust's own frame would be Rust's locals.
// - The one way in, rust_title_main, goes through enter_rust, which saves r14-r31 whole around it: Rust's prologues
//   save them with stw and restore them with lwz, which would zero the high words xsdk's code keeps there.
//
// Arguments and results across the boundary are 32-bit integers, pointers and doubles only: SysV passes a 64-bit
// integer in a register pair and xsdk in one register, and xsdk leaves a narrow result unwidened where LLVM assumes it
// was widened.

#include <stdint.h>
#include <stdlib.h>

#include "xsdk/display.h"
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/splash.h"

enum {
	BUFFERS = 2,
	SWAP_INTERVAL = 1,
};

// 16 fixed bytes + 64 of parameter save area, rounded to the 16-byte stack alignment. A macro: the asm takes it.
#define THUNK_FRAME 96

// Rust to C: __rs2c_<name> calls <name> from a frame of xsdk's shape. LR goes in the caller's frame at 4(r1), where
// both conventions keep it.
#define STR2(x) #x
#define STR(x) STR2(x)
#define RS2C(name)                                                                                                     \
	__asm__(".section .text.__rs2c_" #name ",\"ax\",@progbits\n"                                                       \
		".globl __rs2c_" #name "\n"                                                                                \
		".type __rs2c_" #name ",@function\n"                                                                       \
		".p2align 2\n"                                                                                             \
		"__rs2c_" #name ":\n"                                                                                      \
		"	mflr 0\n"                                                                                           \
		"	stw 0,4(1)\n"                                                                                       \
		"	stwu 1,-" STR(THUNK_FRAME) "(1)\n"                                                                  \
		"	bl " #name "\n"                                                                                     \
		"	addi 1,1," STR(THUNK_FRAME) "\n"                                                                    \
		"	lwz 0,4(1)\n"                                                                                       \
		"	mtlr 0\n"                                                                                           \
		"	blr\n"                                                                                              \
		".size __rs2c_" #name ",.-__rs2c_" #name "\n"                                                              \
		".previous\n")

// What the Rust title may call. Each must take and return only 32-bit integers, pointers or doubles.
RS2C(bridge_log);
RS2C(bridge_present_open);
RS2C(bridge_set_rumble);
RS2C(bridge_abort);
RS2C(xsdk_splash_show);
RS2C(xsdk_gpu_open);
RS2C(xsdk_gpu_close);
RS2C(xsdk_present_begin);
RS2C(xsdk_gpu_resolve);
RS2C(xsdk_present_pixels);
RS2C(xsdk_present_end);
RS2C(xsdk_present_close);
RS2C(XamInputGetState);

// C to Rust: enter_rust(f) calls f, an LLVM SysV function taking nothing and returning a 32-bit int, with r14-r31
// saved whole. 8 bytes of SysV header, 18 doublewords, padded to 16.
__asm__(".section .text.enter_rust,\"ax\",@progbits\n"
	".globl enter_rust\n"
	".type enter_rust,@function\n"
	".p2align 2\n"
	"enter_rust:\n"
	"	mflr 0\n"
	"	stw 0,4(1)\n"
	"	stwu 1,-176(1)\n"
	"	std 14,16(1)\n	std 15,24(1)\n	std 16,32(1)\n	std 17,40(1)\n	std 18,48(1)\n	std 19,56(1)\n"
	"	std 20,64(1)\n	std 21,72(1)\n	std 22,80(1)\n	std 23,88(1)\n	std 24,96(1)\n	std 25,104(1)\n"
	"	std 26,112(1)\n	std 27,120(1)\n	std 28,128(1)\n	std 29,136(1)\n	std 30,144(1)\n	std 31,152(1)\n"
	"	mtctr 3\n"
	"	bctrl\n"
	// The result is 32 bits; xsdk's caller widens it.
	"	ld 14,16(1)\n	ld 15,24(1)\n	ld 16,32(1)\n	ld 17,40(1)\n	ld 18,48(1)\n	ld 19,56(1)\n"
	"	ld 20,64(1)\n	ld 21,72(1)\n	ld 22,80(1)\n	ld 23,88(1)\n	ld 24,96(1)\n	ld 25,104(1)\n"
	"	ld 26,112(1)\n	ld 27,120(1)\n	ld 28,128(1)\n	ld 29,136(1)\n	ld 30,144(1)\n	ld 31,152(1)\n"
	"	addi 1,1,176\n"
	"	lwz 0,4(1)\n"
	"	mtlr 0\n"
	"	blr\n"
	".size enter_rust,.-enter_rust\n"
	".previous\n");

int32_t enter_rust(int32_t (*title_main)(void));
int32_t rust_title_main(void);

// The helpers Rust calls where xsdk's own call can't cross the boundary as it is.

// xsdk_log_write is varargs, so it takes a length-bounded string from Rust here.
void bridge_log(const char *text, uint32_t length)
{
	xsdk_log_write("%.*s", (int)length, text);
}

// The display mode is an enum, whose size the boundary shouldn't depend on.
NTSTATUS bridge_present_open(uint32_t *width, uint32_t *height)
{
	return xsdk_present_open(XSDK_DISPLAY_720P, BUFFERS, SWAP_INTERVAL, width, height);
}

// INPUT_VIBRATION holds two uint16_t: built here rather than shared as a struct.
uint32_t bridge_set_rumble(uint32_t user, uint32_t low, uint32_t high)
{
	const INPUT_VIBRATION vibration = { (uint16_t)low, (uint16_t)high };
	return XamInputSetState(user, 0, &vibration);
}

// Rust's panic handler ends here; exit writes the log's last line.
void bridge_abort(void)
{
	xsdk_log_write("rust_hello: aborting");
	exit(3);
}

int main(void)
{
	return enter_rust(rust_title_main);
}
