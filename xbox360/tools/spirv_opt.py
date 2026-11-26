#!/usr/bin/env python3
"""SPIRV-Tools' optimizer and disassembler through their C API (the system's libSPIRV-Tools), for reshaping the
engine's SPIR-V into what the Xbox 360's shader compiler (xost) reads. Nothing here runs on the console: a module keeps
the hash of what the engine passes vkCreateShaderModule, and only xost sees the reshaped words.

    spirv_opt.py <in.spv> <out.spv> [--pass ...]   # default passes: XOST_PASSES
    spirv_opt.py --dis <in.spv>
"""
import ctypes
import struct
import sys

SPV_ENV_UNIVERSAL_1_0 = 0
SPV_SUCCESS = 0
FRIENDLY_NAMES, INDENT = 64, 8

# xost inlines only functions with one return, and reads no loop in SSA form: merge each function's returns, inline
# every call, and drop what's left. No --ssa-rewrite or --scalar-replacement, which would make the Phis it can't read.
XOST_PASSES = ("--merge-return", "--inline-entry-points-exhaustive", "--eliminate-dead-functions")


class _Binary(ctypes.Structure):
    _fields_ = [("code", ctypes.POINTER(ctypes.c_uint32)), ("wordCount", ctypes.c_size_t)]


class _Text(ctypes.Structure):
    _fields_ = [("str", ctypes.c_char_p), ("length", ctypes.c_size_t)]


_tools = ctypes.CDLL("libSPIRV-Tools.so", mode=ctypes.RTLD_GLOBAL)
_opt = ctypes.CDLL("libSPIRV-Tools-opt.so", mode=ctypes.RTLD_GLOBAL)
_opt.spvOptimizerCreate.restype = ctypes.c_void_p
_opt.spvOptimizerCreate.argtypes = [ctypes.c_int]
_opt.spvOptimizerDestroy.argtypes = [ctypes.c_void_p]
_opt.spvOptimizerRegisterPassFromFlag.restype = ctypes.c_bool
_opt.spvOptimizerRegisterPassFromFlag.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_opt.spvOptimizerRun.restype = ctypes.c_int
_opt.spvOptimizerRun.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.POINTER(_Binary)), ctypes.c_void_p]
# The optimizer calls std::function on every message: without a consumer one aborts the process.
_CONSUMER = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p)
_opt.spvOptimizerSetMessageConsumer.argtypes = [ctypes.c_void_p, _CONSUMER]
MESSAGES = []
_consume = _CONSUMER(lambda level, source, position, message: MESSAGES.append((level, (message or b"").decode())))
_tools.spvOptimizerOptionsCreate.restype = ctypes.c_void_p
_tools.spvOptimizerOptionsSetRunValidator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_tools.spvOptimizerOptionsDestroy.argtypes = [ctypes.c_void_p]
_tools.spvBinaryDestroy.argtypes = [ctypes.POINTER(_Binary)]
_tools.spvContextCreate.restype = ctypes.c_void_p
_tools.spvContextCreate.argtypes = [ctypes.c_int]
_tools.spvBinaryToText.restype = ctypes.c_int
_tools.spvBinaryToText.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.POINTER(_Text)), ctypes.c_void_p]
_tools.spvTextDestroy.argtypes = [ctypes.POINTER(_Text)]


def _words(spirv: bytes):
    count = len(spirv) // 4
    return (ctypes.c_uint32 * count)(*struct.unpack(f"<{count}I", spirv)), count


def optimize(spirv: bytes, passes=XOST_PASSES) -> bytes:
    """`spirv` (little-endian bytes) through `passes`, spirv-opt's flags."""
    optimizer = _opt.spvOptimizerCreate(SPV_ENV_UNIVERSAL_1_0)
    _opt.spvOptimizerSetMessageConsumer(optimizer, _consume)
    options = _tools.spvOptimizerOptionsCreate()
    try:
        _tools.spvOptimizerOptionsSetRunValidator(options, False)
        for flag in passes:
            if not _opt.spvOptimizerRegisterPassFromFlag(optimizer, flag.encode()):
                raise ValueError(f"spirv-opt doesn't know {flag}")
        words, count = _words(spirv)
        out = ctypes.POINTER(_Binary)()
        if _opt.spvOptimizerRun(optimizer, words, count, ctypes.byref(out), options) != SPV_SUCCESS:
            raise ValueError("spirv-opt failed: " + "; ".join(m for _, m in MESSAGES[-3:]))
        try:
            return struct.pack(f"<{out.contents.wordCount}I", *out.contents.code[:out.contents.wordCount])
        finally:
            _tools.spvBinaryDestroy(out)
    finally:
        _tools.spvOptimizerOptionsDestroy(options)
        _opt.spvOptimizerDestroy(optimizer)


def disassemble(spirv: bytes) -> str:
    context = _tools.spvContextCreate(SPV_ENV_UNIVERSAL_1_0)
    words, count = _words(spirv)
    text = ctypes.POINTER(_Text)()
    if _tools.spvBinaryToText(context, words, count, FRIENDLY_NAMES | INDENT, ctypes.byref(text), None) != SPV_SUCCESS:
        raise ValueError("not SPIR-V the disassembler reads")
    try:
        return text.contents.str.decode()
    finally:
        _tools.spvTextDestroy(text)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dis":
        print(disassemble(open(sys.argv[2], "rb").read()))
    elif len(sys.argv) >= 3:
        passes = sys.argv[3:] or XOST_PASSES
        open(sys.argv[2], "wb").write(optimize(open(sys.argv[1], "rb").read(), passes))
    else:
        sys.exit(__doc__)
