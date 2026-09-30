import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import {
  getSemanticAdapterDistributionPath,
  hashSemanticAdapterSources,
  semanticAdapterInstallArgs,
} from './python-setup-adapter.ts'

test('adapter install location derives from the packaged requirements path, not cwd', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'modly-setup-adapter-'))
  t.after(() => rmSync(root, { recursive: true, force: true }))
  const apiDir = join(root, 'app', 'resources', 'api')
  const requirements = join(apiDir, 'requirements.txt')
  assert.equal(getSemanticAdapterDistributionPath(requirements), resolve(apiDir))
  assert.deepEqual(semanticAdapterInstallArgs(apiDir), [
    '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--disable-pip-version-check', resolve(apiDir),
  ])
})

test('setup fingerprint covers pyproject and every executable adapter source', (t) => {
  const apiDir = mkdtempSync(join(tmpdir(), 'modly-adapter-fingerprint-'))
  t.after(() => rmSync(apiDir, { recursive: true, force: true }))
  const parts = join(apiDir, 'runtime', 'adapters', 'parts')
  mkdirSync(parts, { recursive: true })
  writeFileSync(join(apiDir, 'pyproject.toml'), '[project]\nname="test"\n')
  writeFileSync(join(parts, 'florence2_pinned.py'), 'VALUE = 1\n')
  writeFileSync(join(parts, 'helper.py'), 'VALUE = 1\n')
  writeFileSync(join(parts, 'asset.json'), '{}\n')
  const before = hashSemanticAdapterSources(apiDir)
  writeFileSync(join(parts, 'helper.py'), 'VALUE = 2\n')
  assert.notEqual(hashSemanticAdapterSources(apiDir), before)
})
