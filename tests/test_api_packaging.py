"""Offline API distribution checks; native Mach-O/TCC need a real Mac."""
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class APIPackagingTests(unittest.TestCase):
    def test_base_dependencies_exclude_local_stack(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
        names = {d.split('>')[0].split(';')[0].strip() for d in project['dependencies']}
        self.assertTrue(names.isdisjoint({'torch', 'laya', 'transformers', 'huggingface-hub'}))
        self.assertEqual(project['requires-python'], '>=3.12,<3.13')
        for dep in project['optional-dependencies']['local']:
            self.assertIn("sys_platform == 'darwin'", dep)
            self.assertIn("platform_machine == 'arm64'", dep)

    def test_architecture_gate_allows_stale_intel_local_config(self):
        helper = ROOT / 'packaging/bootstrap_uv.sh'
        for arch, backend, expected in [('x86_64', 'api', 0), ('x86_64', '', 0),
                                        ('x86_64', 'local', 0), ('arm64', 'local', 0)]:
            with self.subTest(arch=arch, backend=backend):
                env = dict(os.environ, JUDGE_BACKEND=backend, TEST_ARCH=arch)
                code = 'uname() { echo "$TEST_ARCH"; }; . "$1"; jev_check_arch; result=$?; echo "$JEV_ARCH_ERROR"; exit "$result"'
                result = subprocess.run(['sh', '-c', code, 'test', str(helper)],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected:
                    self.assertIn('JUDGE_BACKEND=api', result.stdout)

    def test_local_extra_only_selected_on_native_arm(self):
        helper = ROOT / 'packaging/bootstrap_uv.sh'
        for arch, backend, expected in [('x86_64', 'local', 1), ('arm64', 'local', 0),
                                        ('arm64', 'api', 1)]:
            env = dict(os.environ, JEV_USE_LOCAL="1" if backend == "local" else "0", TEST_ARCH=arch)
            result = subprocess.run(['sh', '-c',
                'uname() { echo "$TEST_ARCH"; }; . "$1"; jev_use_local',
                'test', str(helper)], env=env)
            self.assertEqual(result.returncode, expected)

    def test_resolver_uses_python_parser_and_keeps_intel_lean(self):
        code = ('. "$1"; uname() { echo "$TEST_ARCH"; }; '
                'uv() { printf "%s" "$TEST_MODE"; }; '
                'jev_resolve_backend /fixture; result=$?; '
                'printf "%s:%s" "$result" "$JEV_USE_LOCAL"')
        for arch, mode, expected in [('arm64', 'local', '0:1'), ('arm64', 'api', '0:0'),
                                     ('arm64', 'invalid', '1:0'), ('x86_64', 'local', '0:0')]:
            env = dict(os.environ, TEST_ARCH=arch, TEST_MODE=mode)
            result = subprocess.check_output(['sh', '-c', code, 'test',
                str(ROOT / 'packaging/bootstrap_uv.sh')], env=env, text=True)
            self.assertEqual(result, expected)

    def test_backend_parser_honors_all_sources_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "packaging").mkdir()
            shutil.copy(ROOT / "src/userconfig.py", root / "src/userconfig.py")
            shutil.copy(ROOT / "packaging/backend_mode.py", root / "packaging/backend_mode.py")
            home = root / "home"
            xdg = root / "xdg"
            env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(xdg))
            env.pop("JUDGE_BACKEND", None)
            paths = [root / ".env", home / "Library/Application Support/jev-jarvis/env",
                     home / ".config/jev-jarvis/env", xdg / "jev-jarvis/env"]

            def mode():
                return subprocess.check_output([sys.executable, str(root / "packaging/backend_mode.py")],
                                               env=env, text=True).strip()

            self.assertEqual(mode(), "api")
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('export JUDGE_BACKEND="  LoCaL  " # opt in\n')
                self.assertEqual(mode(), "local", str(path))
                path.write_text('JUDGE_BACKEND=api\n')
                self.assertEqual(mode(), "api", str(path))
            env["JUDGE_BACKEND"] = " LOCAL "
            self.assertEqual(mode(), "local")

    def test_shell_sourcing_preserves_external_backend_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "env"
            config.write_text('export JUDGE_BACKEND=local\n')
            code = '. "$1"; jev_load_env "$2"; printf "%s" "${JUDGE_BACKEND-unset}"'
            for original in (None, "api"):
                env = dict(os.environ)
                env.pop("JUDGE_BACKEND", None)
                if original is not None:
                    env["JUDGE_BACKEND"] = original
                result = subprocess.check_output(['sh', '-c', code, 'test',
                    str(ROOT / "packaging/bootstrap_uv.sh"), str(config)], env=env, text=True)
                self.assertEqual(result, original or "unset")

    def test_both_launchers_explicitly_select_local_extra(self):
        for path in ['start.command', 'packaging/build_app.sh']:
            text = (ROOT / path).read_text()
            self.assertIn('if jev_use_local; then', text)
            self.assertIn('--extra local', text)
        text = (ROOT / 'packaging/build_app.sh').read_text()
        self.assertIn('-arch arm64 -arch x86_64', text)
        self.assertIn('lipo -verify_arch arm64 x86_64', text)
        self.assertIn('<string>x86_64</string>', text)
        self.assertNotIn('--exclude=builtin.py', text)
        self.assertIn('VENV="$SUPPORT/venv-$(uname -m)"', text)


if __name__ == '__main__':
    unittest.main()
