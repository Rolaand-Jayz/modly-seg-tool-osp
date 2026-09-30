import test from 'node:test'
import assert from 'node:assert/strict'
import { validateInstallManifest } from './extension-install-utils.ts'

const processFiles = {
  hasEntryFile: () => true,
  hasGeneratorFile: () => false,
}

test('legacy extension manifests remain valid with no capabilities field', () => {
  const validated = validateInstallManifest(
    { id: 'legacy-model', generator_class: 'Generator' },
    { hasEntryFile: () => false, hasGeneratorFile: () => true },
    'repository',
  )
  assert.equal(validated.capabilities, undefined)
})

test('Structured Asset descriptor declares vendor-neutral I/O and is marked declaration-only', () => {
  const validated = validateInstallManifest(
    {
      id: 'mesh-import',
      type: 'process',
      capabilities: [{
        capability_id: 'import-geometry',
        contract_version: '1.0.0',
        inputs: ['mesh'],
        outputs: ['structured-asset', 'stage-artifacts'],
        adapter_id: 'modly.mesh-import',
        adapter_revision: 'git:0123456789abcdef0123456789abcdef01234567',
        adapter_trust: 'pinned-reference',
      }],
    },
    processFiles,
    'repository',
  )
  assert.deepEqual(validated.capabilities?.[0], {
    capability_id: 'import-geometry',
    contract_version: '1.0.0',
    inputs: ['mesh'],
    outputs: ['structured-asset', 'stage-artifacts'],
    adapter_id: 'modly.mesh-import',
    adapter_revision: 'git:0123456789abcdef0123456789abcdef01234567',
    adapter_trust: 'pinned-reference',
    identity_status: 'declared-only',
  })
})

test('unknown versions, operations, trust values, and unpinned reference claims fail closed', () => {
  const base = {
    capability_id: 'import-geometry',
    contract_version: '1.0.0',
    inputs: ['mesh'],
    outputs: ['structured-asset'],
    adapter_id: 'example.adapter',
    adapter_revision: '1.0.0',
    adapter_trust: 'third-party-unpinned',
  }
  const invalid = [
    { ...base, capability_id: 'future-operation' },
    { ...base, contract_version: '2.0.0' },
    { ...base, adapter_trust: 'trusted' },
    { ...base, adapter_trust: 'pinned-reference' },
    { ...base, inputs: ['mesh', 'mesh'] },
    { ...base, model_weights_id: 'weights:v1' },
  ]
  for (const capability of invalid) {
    assert.throws(
      () => validateInstallManifest(
        { id: 'mesh-import', type: 'process', capabilities: [capability] },
        processFiles,
        'repository',
      ),
      /Structured Asset|unsupported Structured Asset capability|pinned-reference adapter_revision/,
    )
  }
})
