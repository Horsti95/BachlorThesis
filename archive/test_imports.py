"""
Regression tests for module imports.

Prevents future import errors like the cli_menu -> interactive_menu bug.
Tests that all critical imports work correctly.

Author: Automated fix verification
Date: 2026-01-19
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class TestModuleImports(unittest.TestCase):
    """Test that all critical modules can be imported correctly."""

    def test_interactive_menu_import(self):
        """Test that interactive_menu module exists and can be imported."""
        try:
            import interactive_menu
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import interactive_menu: {e}")

    def test_run_interactive_menu_function(self):
        """Test that run_interactive_menu function exists and is callable."""
        from interactive_menu import run_interactive_menu
        self.assertTrue(callable(run_interactive_menu))

    def test_interactive_menu_class(self):
        """Test that InteractiveMenu class exists."""
        from interactive_menu import InteractiveMenu
        self.assertTrue(isinstance(InteractiveMenu, type))

    def test_no_cli_menu_module(self):
        """Test that the old/wrong 'cli_menu' module name does not exist."""
        with self.assertRaises(ImportError):
            import cli_menu  # noqa: F401

    def test_run_experiment_imports(self):
        """Test that run_experiment.py can import its dependencies."""
        # This tests the specific import that was failing
        from interactive_menu import run_interactive_menu

        # Verify it's the correct function
        self.assertTrue(callable(run_interactive_menu))
        self.assertEqual(run_interactive_menu.__name__, 'run_interactive_menu')


class TestRunExperimentSyntax(unittest.TestCase):
    """Test that run_experiment.py has valid Python syntax."""

    def test_run_experiment_syntax(self):
        """Ensure run_experiment.py can be parsed without syntax errors."""
        import ast

        run_experiment_path = Path(__file__).parent / 'run_experiment.py'
        with open(run_experiment_path, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            self.fail(f"run_experiment.py has syntax error: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
