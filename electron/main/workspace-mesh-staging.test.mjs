import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { stageWorkspaceMeshFile } from './workspace-mesh-staging.mjs'

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'modly-mesh-staging-'))
  t.after(() => rm(root, { recursive: true, force: true }))
  const sourceDir = path.join(root, 'source')
  const workspace = path.join(root, 'workspace')
  await mkdir(sourceDir)
  await mkdir(workspace)
  const source = path.join(sourceDir, 'chair.glb')
  await writeFile(source, Buffer.from('glb-data'))
  return { root, sourceDir, workspace, source }
}

test('stages selected GLB into workspace/imports/meshes and copies exact bytes', async (t) => {
  const { workspace, source } = await fixture(t)
  const result = await stageWorkspaceMeshFile(source, workspace)
  assert.equal(result.fileName, 'chair.glb')
  assert.equal(result.filePath, path.join(workspace, 'imports', 'meshes', 'chair.glb'))
  assert.deepEqual(await readFile(result.filePath), Buffer.from('glb-data'))
})

test('uses collision-safe names without replacing an existing staged mesh', async (t) => {
  const { workspace, source } = await fixture(t)
  const first = await stageWorkspaceMeshFile(source, workspace)
  await writeFile(source, Buffer.from('second'))
  const second = await stageWorkspaceMeshFile(source, workspace)
  assert.equal(first.fileName, 'chair.glb')
  assert.equal(second.fileName, 'chair-1.glb')
  assert.deepEqual(await readFile(first.filePath), Buffer.from('glb-data'))
  assert.deepEqual(await readFile(second.filePath), Buffer.from('second'))
})

test('rejects a source changed during copy and removes the partial staged output', async (t) => {
  const { workspace, source } = await fixture(t)
  const original = Buffer.alloc(2 * 1024 * 1024, 0x41)
  await writeFile(source, original)

  await assert.rejects(
    stageWorkspaceMeshFile(source, workspace, {
      afterFirstChunk: async () => writeFile(source, Buffer.from('changed during copy')),
    }),
    /changed while it was being copied/i,
  )
  assert.deepEqual(await readdir(path.join(workspace, 'imports', 'meshes')), [])
})

test('accepts case-insensitive glTF extensions but rejects unsupported types', async (t) => {
  const { sourceDir, workspace, source } = await fixture(t)
  const gltf = path.join(sourceDir, 'model.GLTF')
  await writeFile(gltf, JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri: 'data:application/octet-stream;base64,AA==' }] }))
  assert.equal((await stageWorkspaceMeshFile(gltf, workspace)).fileName, 'model.gltf')
  await assert.rejects(stageWorkspaceMeshFile(source.replace(/\.glb$/, '.obj'), workspace), /only \.glb and \.gltf/i)
})

test('stages only referenced external buffer and image files with working relative paths', async (t) => {
  const { sourceDir, workspace } = await fixture(t)
  const gltf = path.join(sourceDir, 'scene.gltf')
  const bufferBytes = Buffer.from('mesh-buffer-data')
  const imageBytes = Buffer.from('image-data')
  await mkdir(path.join(sourceDir, 'assets'))
  await mkdir(path.join(sourceDir, 'textures'))
  await writeFile(path.join(sourceDir, 'assets', 'mesh.bin'), bufferBytes)
  await writeFile(path.join(sourceDir, 'textures', 'albedo map.png'), imageBytes)
  await writeFile(gltf, JSON.stringify({
    asset: { version: '2.0' },
    buffers: [{ uri: 'assets/mesh.bin', byteLength: bufferBytes.length }],
    images: [{ uri: 'textures/albedo%20map.png' }],
  }))

  const result = await stageWorkspaceMeshFile(gltf, workspace)
  const document = JSON.parse(await readFile(result.filePath, 'utf8'))
  const bufferPath = path.resolve(path.dirname(result.filePath), decodeURIComponent(document.buffers[0].uri))
  const imagePath = path.resolve(path.dirname(result.filePath), decodeURIComponent(document.images[0].uri))
  assert.equal(path.basename(path.dirname(result.filePath)), 'scene')
  assert.deepEqual(await readFile(bufferPath), bufferBytes)
  assert.deepEqual(await readFile(imagePath), imageBytes)
  assert.deepEqual(await readdir(path.join(workspace, 'imports', 'meshes', 'scene')), ['assets', 'scene.gltf', 'textures'])
})

test('reports missing external glTF dependencies and stages no partial package', async (t) => {
  const { sourceDir, workspace } = await fixture(t)
  const gltf = path.join(sourceDir, 'missing.gltf')
  await writeFile(gltf, JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri: 'missing.bin', byteLength: 1 }] }))
  await assert.rejects(stageWorkspaceMeshFile(gltf, workspace), /is missing.*missing\.bin/i)
  assert.deepEqual(await readdir(path.join(workspace, 'imports', 'meshes')), [])
})

test('rejects remote and decoded traversal resource URIs before staging', async (t) => {
  const { sourceDir, workspace } = await fixture(t)
  for (const [name, uri, expected] of [
    ['remote', 'https://example.invalid/mesh.bin', /remote or absolute URI/i],
    ['traversal', '%2e%2e/mesh.bin', /traversal/i],
  ]) {
    const gltf = path.join(sourceDir, `${name}.gltf`)
    await writeFile(gltf, JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri, byteLength: 1 }] }))
    await assert.rejects(stageWorkspaceMeshFile(gltf, workspace), expected)
  }
  assert.deepEqual(await readdir(path.join(workspace, 'imports', 'meshes')), [])
})

test('rejects symlinked glTF resource path components', async (t) => {
  const { root, sourceDir, workspace } = await fixture(t)
  const outside = path.join(root, 'outside')
  await mkdir(outside)
  await writeFile(path.join(outside, 'mesh.bin'), 'mesh')
  await symlink(outside, path.join(sourceDir, 'linked-assets'))
  const gltf = path.join(sourceDir, 'symlinked.gltf')
  await writeFile(gltf, JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri: 'linked-assets/mesh.bin', byteLength: 4 }] }))
  await assert.rejects(stageWorkspaceMeshFile(gltf, workspace), /symbolic link/i)
  assert.deepEqual(await readdir(path.join(workspace, 'imports', 'meshes')), [])
})

test('rejects selected source symlinks and non-files', async (t) => {
  const { sourceDir, workspace, source } = await fixture(t)
  const linked = path.join(sourceDir, 'linked.glb')
  await symlink(source, linked)
  await assert.rejects(stageWorkspaceMeshFile(linked, workspace), /regular file, not a symbolic link/i)
  const directoryWithMeshSuffix = path.join(sourceDir, 'folder.glb')
  await mkdir(directoryWithMeshSuffix)
  await assert.rejects(stageWorkspaceMeshFile(directoryWithMeshSuffix, workspace), /regular file, not a symbolic link/i)
})

test('rejects symlinked imports directories before writing outside the workspace', async (t) => {
  const { root, workspace, source } = await fixture(t)
  const outside = path.join(root, 'outside')
  await mkdir(outside)
  await symlink(outside, path.join(workspace, 'imports'))
  await assert.rejects(stageWorkspaceMeshFile(source, workspace), /workspace imports folder is unsafe/i)
  assert.deepEqual(await (await import('node:fs/promises')).readdir(outside), [])
})

test('rejects symlinked mesh destination directory', async (t) => {
  const { root, workspace, source } = await fixture(t)
  const outside = path.join(root, 'outside')
  await mkdir(outside)
  await mkdir(path.join(workspace, 'imports'))
  await symlink(outside, path.join(workspace, 'imports', 'meshes'))
  await assert.rejects(stageWorkspaceMeshFile(source, workspace), /workspace imports folder is unsafe/i)
})
