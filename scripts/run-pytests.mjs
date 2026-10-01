// Portable Python test runner.
// `python3` is the macOS/Linux name but does not exist on Windows (where the
// interpreter is `python` or the `py` launcher). Try each candidate until one
// actually runs, then forward unittest's exit code.
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'
import { existsSync } from 'node:fs'

const apiDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'api')
const projectRoot = dirname(apiDir)
const configuredPython = process.env.MODLY_API_TEST_PYTHON
  ? resolve(process.env.MODLY_API_TEST_PYTHON)
  : null

const candidates = [
  ...(configuredPython && existsSync(configuredPython)
    ? [[configuredPython, []]]
    : []),
  ['python3', []],
  ['python', []],
  ['py', ['-3']],
]

function works(cmd, prefix) {
  try {
    const r = spawnSync(cmd, [...prefix, '--version'], { stdio: 'ignore' })
    return r.status === 0
  } catch {
    return false
  }
}

const found = candidates.find(([cmd, prefix]) => works(cmd, prefix))
if (!found) {
  console.error('[run-pytests] No Python interpreter found (tried python3, python, py -3).')
  process.exit(1)
}

const [cmd, prefix] = found
const bootstrap = [
  'import typing_extensions',
  'import sys, unittest',
  'sys.path.insert(0, "api/tests")',
  'sys.path.insert(0, "api")',
  // The API tests are a flat directory without a package __init__.py. Use it
  // as unittest's import root; api/tests and api are separately on sys.path
  // for the legacy absolute imports used by existing tests.
  'suite = unittest.defaultTestLoader.discover("api/tests", top_level_dir="api/tests")',
  'result = unittest.TextTestRunner(verbosity=2).run(suite)',
  'raise SystemExit(not result.wasSuccessful())',
].join('; ')
const result = spawnSync(cmd, [...prefix, '-c', bootstrap], {
  cwd: projectRoot,
  stdio: 'inherit',
})
process.exit(result.status ?? 1)
