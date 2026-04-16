import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        pgm_dir = Path(__file__).parent
        result = subprocess.run(
            ["just", "build"],
            # ["swiftc", "-emit-library", "-o", str(pgm_dir / "libPgm.dylib"), str(pgm_dir / "pgm.swift")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("Swift build failed")
