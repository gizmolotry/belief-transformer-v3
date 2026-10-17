import argparse
import os
import sys


if os.path.basename(sys.argv[0]).lower() == "run_experiments.py":
    _argument_parser_init = argparse.ArgumentParser.__init__

    def _init_with_resolved_conflicts(self, *args, **kwargs):
        kwargs.setdefault("conflict_handler", "resolve")
        return _argument_parser_init(self, *args, **kwargs)

    argparse.ArgumentParser.__init__ = _init_with_resolved_conflicts
