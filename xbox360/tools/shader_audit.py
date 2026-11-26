#!/usr/bin/env python3
"""Tries every shader module the engine made (SHOCK2_SHADER_DUMP's <hash>.<stage>.spv) through xost, as vulkan4xenon's
tools/build_shaders.py lowers it, and reports which xost takes and why it refuses the others.

    shader_audit.py <vulkan4xenon> <xost (xenos-toolchain)> <dump dir> [report.md]

A refusal's GLSL is found through the module's `OpSource`-less SPIR-V by matching the dump's <hash>.<stage>.glsl
files on the compiled SPIR-V the link started from, when there is exactly one candidate of that stage and size.
"""
import collections
import importlib.util
import re
import sys
from pathlib import Path


def load_build_shaders(v4x):
    spec = importlib.util.spec_from_file_location("build_shaders", v4x / "tools" / "build_shaders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORAGE = {0: "UniformConstant", 1: "Input", 2: "Uniform", 3: "Output", 4: "Workgroup", 6: "Private", 7: "Function",
           9: "PushConstant", 12: "StorageBuffer"}
OP_LOAD, OP_ACCESS_CHAIN, OP_VARIABLE = 61, 65, 59


def explain(xost, spirv, message):
    """What each %id in xost's message is: the variable it loads, through access chains, with its storage and name."""
    sys.path.insert(0, str(xost))
    from xsc.spirv import Module
    module = Module.parse(spirv)
    parts = []
    for ident in dict.fromkeys(int(n) for n in re.findall(r"%(\d+)", message)):
        ins = module.by_result.get(ident)
        if ins is None:
            parts.append(f"%{ident}: not defined (OpUndef/OpConstantNull?)")
            continue
        chain = [ins.name]
        while ins and ins.opcode in (OP_LOAD, OP_ACCESS_CHAIN):
            ins = module.by_result.get(ins.operands[0])
            chain.append(ins.name if ins else "?")
        where = ""
        if ins and ins.opcode == OP_VARIABLE:
            name = module.names.get(ins.result_id, "")
            where = f" {STORAGE.get(ins.operands[0], ins.operands[0])} `{name}`"
        parts.append(f"%{ident}: {' of '.join(chain)}{where}")
    return "; ".join(parts)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import spirv_opt
    # --opt lowers each module after spirv_opt.XOST_PASSES. Tried on 2026-11-26: xost refuses the early return
    # merge-return leaves in an inlined function, so the engine's GLSL keeps one return per function instead.
    # xost's own optimiser, as build_shaders.py --optimize runs it; --no-optimize leaves it off.
    optimize = "--no-optimize" not in sys.argv
    if not optimize:
        sys.argv.remove("--no-optimize")
    raw = "--opt" not in sys.argv
    if not raw:
        sys.argv.remove("--opt")
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    v4x, xost, dump = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    report_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    bs = load_build_shaders(v4x)
    modules = sorted(p for p in dump.glob("*.spv") if not p.name.endswith(".compiled.spv"))
    results = []
    for path in modules:
        stage = path.name.split(".")[1]
        spirv = path.read_bytes()
        if not raw:
            spirv = spirv_opt.optimize(spirv)
        entry = {"name": path.name, "stage": stage, "bytes": len(spirv)}
        try:
            for instanced in (False, True):
                _, literals, resources, clips = bs.lower(xost, spirv, path, optimize, instanced=instanced)
                entry.setdefault("resources", len(resources))
                if stage != "vert" or any(r[0] == "instance" for r in resources):
                    break
            entry["ok"] = True
        except SystemExit as error:
            entry["ok"] = False
            entry["error"] = str(error).replace(str(path) + ": ", "")
            try:
                entry["why"] = explain(xost, spirv, entry["error"])
            except Exception as why:
                entry["why"] = f"(no explanation: {why})"
        except Exception as error:  # xost's own crash: worth seeing, not hiding
            entry["ok"] = False
            entry["error"] = f"xost raised {type(error).__name__}: {error}"
        results.append(entry)
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    # Group refusals by their message with numbers and names taken out.
    kinds = collections.Counter(re.sub(r"\b(0x[0-9a-f]+|\d+)\b", "N", r["error"].splitlines()[0])[:160] for r in bad)
    lines = [f"# Shader audit: {len(ok)} of {len(results)} modules lower through xost", ""]
    for stage in ("vert", "frag"):
        of = [r for r in results if r["stage"] == stage]
        lines.append(f"- {stage}: {sum(r['ok'] for r in of)} of {len(of)}")
    lines += ["", "## Refusals by kind", ""]
    for kind, n in kinds.most_common():
        lines.append(f"- {n} x `{kind}`")
    lines += ["", "## Each refusal", ""]
    for r in bad:
        why = f" -- {r['why']}" if r.get("why") else ""
        lines.append(f"- `{r['name']}` ({r['bytes']} bytes): {r['error'].splitlines()[0][:300]}{why}")
    report = "\n".join(lines) + "\n"
    if report_path:
        report_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
