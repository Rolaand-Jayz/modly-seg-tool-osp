import test from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile, readdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { readStructuredStageCache, structuredStageCacheKey, writeStructuredStageCache } from './structured-stage-cache.ts'

const sha = value => `sha256:${createHash('sha256').update(value).digest('hex')}`
const descriptor = {
  capability_id: 'validate-structured-asset', contract_version: '1.0.0', inputs: ['structured-asset'],
  outputs: ['structured-asset'], adapter_id: 'modly.validate', adapter_revision: 'sha256:' + 'a'.repeat(64),
  adapter_trust: 'builtin', identity_status: 'declared-only',
}

async function fixture(t) {
  const workspaceDir = await mkdtemp(join(tmpdir(), 'modly-stage-cache-'))
  t.after(() => rm(workspaceDir, { recursive: true, force: true }))
  const extensionDir = join(workspaceDir, 'extension')
  await mkdir(extensionDir)
  await writeFile(join(extensionDir, 'manifest.json'), '{"v":1}')
  const geometryBytes = Buffer.from('mesh-data')
  await mkdir(join(workspaceDir, 'Geometry'))
  await writeFile(join(workspaceDir, 'Geometry', 'mesh.glb'), geometryBytes)
  const geometryRef = { artifact_id: sha(geometryBytes), workspace_path: 'Geometry/mesh.glb', digest: sha(geometryBytes), media_type: 'model/gltf-binary' }
  const asset = {
    schema_id: 'org.modly.structured-asset', schema_version: '1.0.0', asset_id: 'asset-1', geometry: geometryRef,
    topology_revision: 'sha256:' + 'c'.repeat(64), topology_counts: { mesh_count: 1, primitive_count: 1, vertex_count: 3, face_count: 1 },
    coordinate_frame: { basis: 'xyz', handedness: 'right', units: 'm', transforms: [] },
    source_observations: [], object_components: [], part_segments: [], material_regions: [], mappings: [],
    assertions: [], corrections: [], provenance: { actor_id: 'unknown', parameters: {} },
    validation_state: 'valid', stage_artifacts: [],
  }
  await mkdir(join(workspaceDir, 'StructuredAssets'))
  const inputSidecarPath = join(workspaceDir, 'StructuredAssets', 'input.json')
  const outputPath = join(workspaceDir, 'StructuredAssets', 'output.json')
  await writeFile(inputSidecarPath, JSON.stringify(asset))
  const stageOutputArtifact = { stage_id: 'validation', artifact: geometryRef }
  const outputAsset = { ...asset, provenance: { ...asset.provenance, sequence: 2 }, stage_artifacts: [stageOutputArtifact] }
  await writeFile(outputPath, JSON.stringify(outputAsset))
  let validationCalls = 0
  const validateAsset = async path => {
    validationCalls++
    const parsed = JSON.parse(await readFile(path, 'utf8'))
    if (parsed.schema_id !== 'org.modly.structured-asset' || parsed.schema_version !== '1.0.0'
        || !parsed.geometry || parsed.validation_state !== 'valid') return null
    const root = await realpath(workspaceDir)
    const geometryPath = await realpath(join(root, parsed.geometry.workspace_path))
    if (!geometryPath.startsWith(`${root}/`)
        || parsed.geometry.digest !== sha(await readFile(geometryPath))) return null
    return parsed
  }
  const request = {
    workspaceDir, extensionDir, extensionId: 'validate-ext', nodeId: 'validate', descriptor,
    inputSidecarPath, params: {}, builtin: true, pythonEntry: false, hostVersion: '0.4.2', validateAsset,
  }
  const result = {
    structuredAssetPath: outputPath, structuredAsset: outputAsset,
    stageOutputArtifact,
  }
  return { workspaceDir, extensionDir, inputSidecarPath, outputPath, request, asset, outputAsset, result, validationCalls: () => validationCalls }
}

test('key binds full extension dependency tree and authorized parameters after authoritative validation', async t => {
  const f = await fixture(t)
  const original = await structuredStageCacheKey(f.request)
  assert.ok(original)
  assert.ok(f.validationCalls() > 0)
  assert.notEqual(await structuredStageCacheKey({ ...f.request, params: { changed: true } }), original)
  await mkdir(join(f.extensionDir, 'node_modules'))
  await writeFile(join(f.extensionDir, 'node_modules', 'runtime.js'), 'dep-v1')
  const withDependency = await structuredStageCacheKey(f.request)
  assert.notEqual(withDependency, original)
  await writeFile(join(f.extensionDir, 'node_modules', 'runtime.js'), 'dep-v2')
  assert.notEqual(await structuredStageCacheKey(f.request), withDependency)
})

test('runtime, Python, model-weight, unsupported capability, and third-party identities fail closed', async t => {
  const f = await fixture(t)
  assert.equal(await structuredStageCacheKey({ ...f.request, pythonEntry: true }), null)
  assert.equal(await structuredStageCacheKey({ ...f.request, builtin: false }), null)
  assert.equal(await structuredStageCacheKey({ ...f.request, descriptor: { ...descriptor, model_weights_id: 'weights', model_weights_digest: 'sha256:' + 'b'.repeat(64) } }), null)
  assert.equal(await structuredStageCacheKey({ ...f.request, descriptor: { ...descriptor, capability_id: 'segment-parts' } }), null)
  assert.equal(await structuredStageCacheKey({ ...f.request, descriptor: { ...descriptor, adapter_trust: 'third-party-unpinned' } }), null)
})

test('symlinked input and cache roots cannot escape the workspace', async t => {
  const f = await fixture(t)
  const outside = await mkdtemp(join(tmpdir(), 'modly-cache-outside-'))
  t.after(() => rm(outside, { recursive: true, force: true }))
  await writeFile(join(outside, 'mesh.glb'), 'mesh-data')
  await symlink(outside, join(f.workspaceDir, 'external-geometry'))
  const escapedAsset = { ...f.asset, geometry: { ...f.asset.geometry, workspace_path: 'external-geometry/mesh.glb' } }
  await writeFile(f.inputSidecarPath, JSON.stringify(escapedAsset))
  assert.equal(await structuredStageCacheKey(f.request), null)

  await writeFile(f.inputSidecarPath, JSON.stringify(f.asset))
  await symlink(outside, join(f.workspaceDir, '.modly-amd-runtime'))
  const key = await structuredStageCacheKey(f.request)
  assert.ok(key)
  assert.equal(await readStructuredStageCache(f.request, key), null)
  assert.deepEqual(await readdir(outside), ['mesh.glb'])
})

test('authoritative validation rejects malformed or changed artifacts before cache reuse', async t => {
  const f = await fixture(t)
  const key = await structuredStageCacheKey(f.request)
  assert.ok(key)
  await writeFile(f.inputSidecarPath, JSON.stringify({ schema_id: 'org.modly.structured-asset', schema_version: '1.0.0' }))
  assert.equal(await structuredStageCacheKey(f.request), null)
  await writeFile(f.inputSidecarPath, JSON.stringify(f.asset))
  await writeFile(join(f.workspaceDir, 'Geometry', 'mesh.glb'), 'changed bytes')
  assert.equal(await structuredStageCacheKey(f.request), null)
})

test('atomic validated cache reuse preserves successful result fields and bypasses unsupported shapes', async t => {
  const f = await fixture(t)
  const key = await structuredStageCacheKey(f.request)
  assert.ok(key)
  assert.equal(await writeStructuredStageCache(f.request, key, f.result, f.outputPath), true)
  const hit = await readStructuredStageCache(f.request, key)
  assert.equal(hit?.cache.state, 'hit')
  assert.deepEqual(Object.keys(hit.result).sort(), Object.keys(f.result).sort())
  assert.equal(hit.result.stageOutputArtifact.stage_id, f.result.stageOutputArtifact.stage_id)
  assert.notEqual(hit.result.structuredAssetPath, f.outputPath)
  assert.deepEqual(hit.result.structuredAsset, f.outputAsset)
  assert.equal(await writeStructuredStageCache(f.request, key, { ...f.result, telemetry: { latency_ms: 1 } }, f.outputPath), false)

  const cacheDir = join(f.workspaceDir, '.modly-amd-runtime', 'structured-stage-cache', key)
  assert.deepEqual((await readdir(cacheDir)).filter(name => name.endsWith('.tmp')), [])
  const entry = JSON.parse(await readFile(join(cacheDir, 'entry.json'), 'utf8'))
  assert.equal(entry.format, 2)
})

test('cache entries are revalidated and symlinked entry files fail closed', async t => {
  const f = await fixture(t)
  const key = await structuredStageCacheKey(f.request)
  assert.ok(key)
  assert.equal(await writeStructuredStageCache(f.request, key, f.result, f.outputPath), true)
  const cacheDir = join(f.workspaceDir, '.modly-amd-runtime', 'structured-stage-cache', key)
  const entryPath = join(cacheDir, 'entry.json')
  const outside = join(f.workspaceDir, 'outside-cache-entry.json')
  await writeFile(outside, await readFile(entryPath))
  await rm(entryPath)
  await symlink(outside, entryPath)
  assert.equal(await readStructuredStageCache(f.request, key), null)
})
