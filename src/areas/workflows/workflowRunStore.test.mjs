import test from 'node:test'
import assert from 'node:assert/strict'
import { buildSync } from 'esbuild'
import { createRequire } from 'node:module'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

function loadStore() {
  const dir = mkdtempSync(join(tmpdir(), 'modly-workflow-rerun-test-'))
  const stub = (name, content) => {
    const path = join(dir, `${name}.cjs`)
    writeFileSync(path, content, 'utf8')
    return path
  }
  const aliases = {
    zustand: stub('zustand', "exports.create = (initializer) => { let state; const set = (next) => { state = { ...state, ...(typeof next === 'function' ? next(state) : next) } }; const get = () => state; state = initializer(set, get); const useStore = () => state; useStore.getState = get; return useStore }") ,
    axios: stub('axios', 'module.exports = {}'),
    '@shared/stores/appStore': stub('appStore', 'exports.useAppStore = { getState: () => ({ updateCurrentJob() {}, apiUrl: "" }) }'),
    '@shared/utils/notification': stub('notification', 'exports.showCompletionNotification = async () => {}'),
  }
  const result = buildSync({
    entryPoints: [resolve('src/areas/workflows/workflowRunStore.ts')],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    alias: aliases,
  })
  const outfile = join(dir, 'workflowRunStore.cjs')
  writeFileSync(outfile, result.outputFiles[0].text, 'utf8')
  return createRequire(import.meta.url)(outfile)
}

const { planCorrectionWorkflowRerun, isCurrentSceneGeometry } = loadStore()
const node = (id, extensionId, enabled = true) => ({
  id, type: 'extensionNode', position: { x: 0, y: 0 },
  data: { enabled, extensionId, params: {}, label: id },
})
const edge = (source, target) => ({ id: `${source}->${target}`, source, target })
const extension = (id) => ({
  id: `${id}/process`, extensionId: id, extensionName: id, extensionAuthor: '', nodeId: 'process',
  name: id, description: '', input: 'mesh', output: 'mesh', params: [], builtin: true, type: 'process',
})
const workflow = (nodes, edges) => ({ id: 'workflow-1', name: 'Test', nodes, edges })

test('corrected sidecar geometry must match the current workflow scene, not any earlier branch output', () => {
  assert.equal(isCurrentSceneGeometry('/workspace/project', '/workspace/current/model.glb', '/workspace/project/current/model.glb'), true)
  assert.equal(isCurrentSceneGeometry('/workspace/project', '/workspace/current/model.glb', '/workspace/project/earlier/model.glb'), false)
  assert.equal(isCurrentSceneGeometry('/workspace/project', undefined, '/workspace/project/current/model.glb'), false)
})

test('semantic-label correction reruns its configured semantic stage and true dependents only', () => {
  const geometry = node('geometry', 'geometry-generator')
  const parts = node('parts', 'part-segmenter')
  const semantics = node('semantics', 'identify-part-semantics/process')
  const identity = node('identity', 'project-owned-material-identity/process')
  const pbr = node('pbr', 'project-owned-pbr-estimation/process')
  const unrelated = node('unrelated', 'unrelated-branch')
  const extensions = [
    extension('identify-part-semantics'), extension('project-owned-material-identity'),
    extension('project-owned-pbr-estimation'), extension('unrelated-branch'),
  ]
  const plan = planCorrectionWorkflowRerun(workflow(
    [geometry, parts, semantics, identity, pbr, unrelated],
    [edge('geometry', 'parts'), edge('parts', 'semantics'), edge('semantics', 'identity'),
      edge('identity', 'pbr'), edge('geometry', 'unrelated')],
  ), extensions, 'part.semantic-label')

  assert.deepEqual(plan.sourceNodeIds, ['semantics'])
  assert.deepEqual(plan.rerunNodes.map(({ nodeId }) => nodeId), ['semantics', 'identity', 'pbr'])
  assert.deepEqual(plan.invalidateNodeIds, ['semantics', 'identity', 'pbr'])
  assert.deepEqual(plan.preserveNodeIds, ['geometry', 'parts', 'unrelated'])
  assert.deepEqual(plan.blockers, [])
})

test('material-name correction includes PBR only when the workflow declares that dependency', () => {
  const identity = node('identity', 'project-owned-material-identity/process')
  const pbr = node('pbr', 'project-owned-pbr-estimation/process')
  const extensions = [extension('project-owned-material-identity'), extension('project-owned-pbr-estimation')]
  const withoutDependency = planCorrectionWorkflowRerun(
    workflow([identity, pbr], []), extensions, 'material.name',
  )
  assert.deepEqual(withoutDependency.rerunNodes.map(({ nodeId }) => nodeId), ['identity'])
  assert.deepEqual(withoutDependency.preserveNodeIds, ['pbr'])

  const withDependency = planCorrectionWorkflowRerun(
    workflow([identity, pbr], [edge('identity', 'pbr')]), extensions, 'material.name',
  )
  assert.deepEqual(withDependency.rerunNodes.map(({ nodeId }) => nodeId), ['identity', 'pbr'])
  assert.deepEqual(withDependency.invalidateNodeIds, ['identity', 'pbr'])
})

test('planner fails closed when an affected stage is disabled or crosses unsupported control flow', () => {
  const identity = node('identity', 'project-owned-material-identity/process')
  const disabledPbr = node('pbr', 'project-owned-pbr-estimation/process', false)
  const plan = planCorrectionWorkflowRerun(
    workflow([identity, disabledPbr], [edge('identity', 'pbr')]),
    [extension('project-owned-material-identity'), extension('project-owned-pbr-estimation')],
    'material.name',
  )
  assert.ok(plan.blockers.some((blocker) => blocker.includes('disabled')))
})
