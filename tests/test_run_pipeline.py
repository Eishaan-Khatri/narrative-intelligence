import types
import unittest
import ast
from pathlib import Path
from unittest.mock import patch

import run_pipeline


class RunPipelineTests(unittest.TestCase):
    def test_run_step_calls_configured_fallback_function(self):
        called = []
        module = types.SimpleNamespace(run_pipeline=lambda: called.append("run"))

        with patch.object(run_pipeline.importlib, "import_module", lambda _name: module):
            step = {
                "name": "dummy",
                "description": "dummy fallback",
                "module": "dummy.module",
                "function": "run_pipeline",
            }

            self.assertTrue(run_pipeline.run_step(step))
            self.assertEqual(called, ["run"])

    def test_run_step_fails_when_configured_function_is_missing(self):
        module = types.SimpleNamespace()

        with patch.object(run_pipeline.importlib, "import_module", lambda _name: module):
            step = {
                "name": "dummy",
                "description": "missing function",
                "module": "dummy.module",
                "function": "run_pipeline",
            }

            self.assertFalse(run_pipeline.run_step(step))

    def test_all_configured_step_functions_exist_in_source(self):
        project_root = Path(run_pipeline.__file__).resolve().parent

        for step in run_pipeline.STEPS:
            function_name = step.get("function")
            if function_name is None:
                continue

            module_path = project_root.joinpath(*step["module"].split(".")).with_suffix(".py")
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(module_path))
            function_names = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            self.assertIn(
                function_name,
                function_names,
                f"{step['name']} points to missing function "
                f"{step['module']}.{function_name}",
            )


if __name__ == "__main__":
    unittest.main()
