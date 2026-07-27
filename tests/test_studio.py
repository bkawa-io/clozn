"""test_studio.py -- smoke test for the LIVE studio backend.

Imports the modules that make up the live clozn studio and checks their key structure WITHOUT loading
any model (no GPU, no weights), so it catches import/syntax/refactor regressions in seconds. The live
backend is the clozn/ package -- everything left in research/ is research spikes from the journey.

    cloze .venv python tests/test_studio.py        # exits 0 if green, 1 if anything regressed
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def has_all(obj, names):
    return all(hasattr(obj, n) for n in names)


def main():
    """Run the smoke checks; return an exit code (0 green, 1 regressed).

    Everything lives inside main() -- including the imports and sys.path setup -- so that a plain
    `pytest` collection can import this module without executing the checks or calling sys.exit
    (which would abort the collector). Run it directly for the smoke test; see the module docstring.
    """
    sys.path.insert(0, os.path.dirname(HERE))                          # repo root, for `from clozn import ...`
    sys.path.insert(0, os.path.join(HERE, "..", "engine", "client"))   # cloze_engine SDK

    _checks = []

    def ok(name, cond):
        _checks.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}", flush=True)

    # --- product gateway: importing it must remain Torch-free ---------------------------------------
    from clozn.server import app as cs

    ok("Substrate base has _memory + _steer", has_all(cs.Substrate, ("_memory", "_steer")))
    ok("EngineSubstrate is the product adapter", issubclass(cs.EngineSubstrate, cs.Substrate))
    ok("product gateway import does not load torch", "torch" not in sys.modules)

    # Lab adapters now live in clozn/lab/substrates.py -- the product gateway must NOT expose them...
    ok("product gateway does not expose lab substrates", not hasattr(cs, "QwenSubstrate"))
    # ...and importing them stays Torch-free (optional deps load lazily, only on instantiation).
    from clozn.lab import substrates as lab_subs
    ok("lab substrate import stays torch-free", "torch" not in sys.modules)
    ok("QwenSubstrate(Substrate)", issubclass(lab_subs.QwenSubstrate, cs.Substrate))
    ok("QwenSubstrate: chat + chat_stream + _gen + handle",
       has_all(lab_subs.QwenSubstrate, ("chat", "chat_stream", "_gen", "handle")))

    # --- steering: the tone dials (AR) --------------------------------------------------------------
    import clozn.behavior.steering as steering

    ok("10 base tone axes", len(steering.AXES) == 10)
    ok("EngineSteer: compute/set/generate (tone dials on any GGUF via the engine)",
       has_all(steering.EngineSteer, ("compute", "set", "generate")))

    # --- optional PyTorch lab ----------------------------------------------------------------------
    import importlib.util
    if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        ok("SteeringControl: compute/set/engage/save_state/load_state",
           has_all(steering.SteeringControl,
                   ("compute", "set", "engage", "disengage", "save_state", "load_state")))

        import clozn.lab.substrates.self_teach as self_teach_server
        ok("SelfTeach: say/consolidate/save/load/_generate",
           has_all(self_teach_server.SelfTeach, ("say", "consolidate", "save", "load", "_generate")))

        import clozn.readouts.brain as brain_readout
        ok("BrainReadout: think/concepts_only/concepts_from_engine",
           has_all(brain_readout.BrainReadout, ("think", "concepts_only", "concepts_from_engine")))

        from clozn.readouts import sae7b
        ok("sae7b: GpuSAE/load7b/feats7b", has_all(sae7b, ("GpuSAE", "load7b", "feats7b")))
    else:
        print("  SKIP  optional PyTorch lab modules (install lab dependencies to exercise)", flush=True)

    from clozn.readouts import atlas_concepts

    ok("atlas_concepts.content_word", hasattr(atlas_concepts, "content_word"))

    passed = sum(_checks)
    print(f"\n{passed}/{len(_checks)} checks passed", flush=True)
    return 0 if passed == len(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
