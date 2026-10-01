import { create } from 'zustand'
import axios, { AxiosInstance } from 'axios'
import { useAppStore } from '@shared/stores/appStore'
import { getWorkflowExtension } from './mockExtensions'
import { showCompletionNotification } from '@shared/utils/notification'
import type { WorkflowExtension } from './mockExtensions'
import type { Workflow, WFNode, WFEdge } from '@shared/types/electron.d'
import { isBranchStarter, isSceneOutput, resolveDataSource, reachesSceneOutput, nearestUpstreamWaits } from './nodeBehaviors'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface WorkflowRunState {
  status:        'idle' | 'running' | 'paused' | 'done' | 'error'
  blockIndex:    number
  blockTotal:    number
  blockProgress: number
  blockStep:     string
  outputUrl?:    string
  outputPath?:   string
  structuredAssetPath?: string
  /** Measurements reported by workflow stages; absent measurements stay unknown. */
  stageTelemetry?: WorkflowStageTelemetry[]
  stageTelemetryOutputUrl?: string
  error?:        string
}

export interface WorkflowStageTelemetry {
  nodeId: string
  nodeLabel: string
  stageId: string
  backend?: string
  device?: string
  runtime?: string
  latencyMs?: number
  peakVramAllocatedBytes?: number
  peakVramReservedBytes?: number
  evidencePath?: string
  status?: string
}

export type WaitState = 'blocked' | 'pending' | 'running' | 'done' | 'error'

const IDLE: WorkflowRunState = {
  status: 'idle', blockIndex: 0, blockTotal: 0, blockProgress: 0, blockStep: '',
}

// ─── Module-level run context (survives between run() and continueRun(id)) ───

const _cancel      = { current: false }
const _activeJobId = { current: null as string | null }
const _activeProcessExtensionId = { current: null as string | null }
// While container (manual mode) pause/resume — set by continueWhile()/retryWhile().
const _resume      = { current: null as (() => void) | null }
const _retry       = { current: false }
// For Each auto-loop: the user asked to pause at the next iteration boundary.
const _pauseRequested = { current: false }
// Live node params — the UI pushes edits here so a looping/paused run re-reads the
// latest values when a body node starts (instead of the snapshot from run start).
const _liveParams  = { current: new Map<string, Record<string, unknown>>() }

function flushResume(): void {
  const fn = _resume.current
  _resume.current = null
  if (fn) fn()
}

interface NodeOutput {
  filePath?: string
  text?: string
  outputType?: string
  structuredAssetPath?: string
  stageOutputArtifact?: Record<string, unknown>
  stageTelemetry?: WorkflowStageTelemetry[]
}

function finiteMetric(record: Record<string, unknown>, keys: string[]): number | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value
  }
  return undefined
}

function textMetric(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return undefined
}

function telemetryRows(
  report: unknown,
  base: Omit<WorkflowStageTelemetry, 'stageId'> & { stageId?: string },
  includeUnknown = false,
): WorkflowStageTelemetry[] {
  if (!report || typeof report !== 'object' || Array.isArray(report)) return []
  const root = report as Record<string, unknown>
  const reports = Array.isArray(root.runtime_reports) ? root.runtime_reports
    : Array.isArray(root.runtime_module_reports) ? root.runtime_module_reports : null
  const sources = reports?.filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)) ?? [root]
  const telemetryKeys = ['backend', 'selected_backend', 'selectedBackend', 'device', 'device_identity', 'deviceIdentity',
    'runtime', 'runtime_version', 'runtimeVersion', 'latency_ms', 'latencyMs', 'warm_inference_latency_ms',
    'peak_vram_allocated_bytes', 'peak_allocated_bytes', 'peak_vram_bytes', 'accelerator_vram_bytes',
    'peak_vram_reserved_bytes', 'peak_reserved_bytes', 'execution_backend', 'status', 'host_wall_time_ms',
    'inference_backend', 'resources']
  if (!includeUnknown && ![root, ...sources].some((source) => telemetryKeys.some((key) => source[key] !== undefined))) return []
  return sources.map((source, index) => ({
    ...base,
    stageId: textMetric(source, ['stage', 'stage_id', 'stageId'])
      ?? textMetric(root, ['stage_id', 'stageId'])
      ?? (base.stageId ? (sources.length > 1 ? `${base.stageId}:${index + 1}` : base.stageId) : `${base.nodeId}:${index + 1}`),
    backend: textMetric(source, ['backend', 'selected_backend', 'selectedBackend'])
      ?? (source.inference_backend && typeof source.inference_backend === 'object'
        ? (((source.inference_backend as Record<string, unknown>).state === 'unknown') ? 'unknown'
          : textMetric(source.inference_backend as Record<string, unknown>, ['value'])) : undefined)
      ?? textMetric(root, ['backend', 'selected_backend', 'selectedBackend']),
    device: textMetric(source, ['device', 'device_identity', 'deviceIdentity'])
      ?? (source.device && typeof source.device === 'object'
        ? (((source.device as Record<string, unknown>).state === 'unknown') ? 'unknown'
          : textMetric(source.device as Record<string, unknown>, ['value'])) : undefined)
      ?? textMetric(root, ['device', 'device_identity', 'deviceIdentity']),
    runtime: textMetric(source, ['runtime', 'runtime_version', 'runtimeVersion'])
      ?? textMetric(root, ['runtime', 'runtime_version', 'runtimeVersion']),
    latencyMs: finiteMetric(source, ['latency_ms', 'latencyMs', 'warm_inference_latency_ms'])
      ?? finiteMetric(root, ['latency_ms', 'latencyMs', 'warm_inference_latency_ms']),
    peakVramAllocatedBytes: finiteMetric(source, ['peak_vram_allocated_bytes', 'peak_allocated_bytes', 'peak_vram_bytes', 'accelerator_vram_bytes'])
      ?? finiteMetric(root, ['peak_vram_allocated_bytes', 'peak_allocated_bytes', 'peak_vram_bytes', 'accelerator_vram_bytes']),
    peakVramReservedBytes: finiteMetric(source, ['peak_vram_reserved_bytes', 'peak_reserved_bytes'])
      ?? finiteMetric(root, ['peak_vram_reserved_bytes', 'peak_reserved_bytes']),
    status: textMetric(source, ['status']),
    evidencePath: typeof root.evidencePath === 'string' ? root.evidencePath : base.evidencePath,
  }))
}

/** Read digest-bound stage reports referenced by the process result's Structured Asset. */
async function readStructuredStageTelemetry(
  structuredAsset: unknown,
  workspaceDir: string,
  base: Omit<WorkflowStageTelemetry, 'stageId'>,
): Promise<WorkflowStageTelemetry[]> {
  if (!structuredAsset || typeof structuredAsset !== 'object' || Array.isArray(structuredAsset)) return []
  const stageArtifacts = (structuredAsset as Record<string, unknown>).stage_artifacts
  if (!Array.isArray(stageArtifacts)) return []
  const output: WorkflowStageTelemetry[] = []
  for (const item of stageArtifacts) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue
    const entry = item as Record<string, unknown>
    const stageId = textMetric(entry, ['stage_id', 'stageId'])
    const artifact = entry.artifact
    if (!stageId || !artifact || typeof artifact !== 'object' || Array.isArray(artifact)) continue
    const ref = artifact as Record<string, unknown>
    const relativePath = textMetric(ref, ['workspace_path', 'workspacePath'])
    const expectedDigest = textMetric(ref, ['digest'])
    if (!relativePath || !expectedDigest?.startsWith('sha256:') || relativePath.startsWith('/')
      || relativePath.split(/[\\/]/).some((segment) => segment === '..' || segment === '')) continue
    const absolutePath = `${workspaceDir.replace(/[\\/]+$/, '')}/${relativePath}`
    try {
      const encoded = await window.electron.fs.readFileBase64(absolutePath)
      const bytes = Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0))
      const digest = `sha256:${Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (value) => value.toString(16).padStart(2, '0')).join('')}`
      if (digest !== expectedDigest) continue
      const document = JSON.parse(new TextDecoder().decode(bytes)) as unknown
      output.push(...telemetryRows(document, { ...base, stageId, evidencePath: relativePath }, true))
    } catch {
      // Telemetry is optional evidence. An unreadable or stale artifact remains unknown.
    }
  }
  return output
}

function isSceneMeshOutput(output: NodeOutput | undefined): output is NodeOutput & { filePath: string } {
  return output?.outputType === 'mesh' && typeof output.filePath === 'string'
}

// ─── For Each iterator (image / text / mesh) ───────────────────────────────────
// A "For Each" node walks a folder alphabetically and emits one file per loop
// iteration. Its loop body = the executable nodes reachable downstream, which
// re-run for every file. The `mode` param picks what it emits and which files it
// matches.

const FOR_EACH_MODES: Record<string, { exts: string[]; outputType: 'image' | 'text' | 'mesh' }> = {
  image: { exts: ['png', 'jpg', 'jpeg', 'webp'],              outputType: 'image' },
  text:  { exts: ['txt', 'md', 'prompt'],                     outputType: 'text'  },
  mesh:  { exts: ['glb', 'gltf', 'obj', 'stl', 'ply', 'fbx'], outputType: 'mesh'  },
}

function iteratorConfig(node: WFNode): { exts: string[]; outputType: 'image' | 'text' | 'mesh' } {
  return FOR_EACH_MODES[(node.data.params?.mode as string) ?? 'image'] ?? FOR_EACH_MODES.image
}

function isIterator(type: string | undefined): boolean {
  return type === 'forEachNode'
}

function isMemberOfActiveLoop(body: Set<string> | null, nodeId: string): boolean {
  return body !== null && body.has(nodeId)
}

/** True for nodes the runner executes (and can re-run inside a loop body). */
function isExecutable(node: WFNode): boolean {
  if (isIterator(node.type)) return true
  return node.type === 'extensionNode' && !!node.data.enabled
}

/** Absolute, alphabetically-sorted paths of an iterator's files (listFiles sorts). */
async function listIteratorFiles(dir: string, exts: string[]): Promise<string[]> {
  const names = await window.electron.fs.listFiles(dir, exts)
  const norm  = dir.replace(/\\/g, '/').replace(/\/+$/, '')
  return names.map((n) => `${norm}/${n}`)
}

/** Read a UTF-8 text file through the base64 IPC bridge. */
async function readTextFile(filePath: string): Promise<string> {
  const b64 = await window.electron.fs.readFileBase64(filePath)
  return new TextDecoder('utf-8').decode(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)))
}

/**
 * Executable nodes reachable downstream from `startId` (its loop body). Traversal
 * stops at Wait boundaries — those nodes belong to branches, not the pre-phase loop.
 */
function reachableExecutable(startId: string, edges: WFEdge[], nodeMap: Map<string, WFNode>): Set<string> {
  const body = new Set<string>([startId])
  const stack = [startId]
  const seen = new Set<string>([startId])
  while (stack.length > 0) {
    const id = stack.pop()!
    for (const e of edges) {
      if (e.source !== id || seen.has(e.target)) continue
      seen.add(e.target)
      const t = nodeMap.get(e.target)
      if (!t || isBranchStarter(t.type)) continue
      if (isExecutable(t)) body.add(e.target)
      stack.push(e.target)
    }
  }
  return body
}

function toWorkspaceUrl(filePath: string, workspaceDir: string): string | undefined {
  const norm = filePath.replace(/\\/g, '/')
  if (!norm.startsWith(workspaceDir)) return undefined
  return `/workspace/${norm.slice(workspaceDir.length).replace(/^\//, '')}`
}

interface RunContext {
  workflow:           Workflow
  allExtensions:      WorkflowExtension[]
  client:             AxiosInstance
  workspaceDir:       string
  selectedImagePath:  string | undefined
  selectedImageData?: string
  overrideImageData?: string
  nodeOutputs:        Map<string, NodeOutput>
  nodeMap:            Map<string, WFNode>
  /** nodes in execution (topological) order */
  ordered:            WFNode[]
  branches:           Map<string, WFNode[]>
  waitIds:            string[]
  /** waitId → nearest upstream waitId (null = top-level, runnable from the start) */
  parentWait:         Map<string, string | null>
  /** iterator node id → its resolved file paths, one per loop iteration */
  iteratorFiles:      Map<string, string[]>
  /** workspace URL of the most recently pushed scene mesh (last branch the user ran wins) */
  lastSceneMesh?:     string
  /** Sidecar belonging to lastSceneMesh, when the producing stage emitted one. */
  lastSceneMeshStructuredAssetPath?: string
}
const _ctx = { current: null as RunContext | null }

export interface TargetedWorkflowRerunNode {
  nodeId: string
  label: string
  extensionId: string
}

export interface TargetedWorkflowRerunPlan {
  correctionProperty: string
  sourceNodeIds: string[]
  rerunNodes: TargetedWorkflowRerunNode[]
  invalidateNodeIds: string[]
  preserveNodeIds: string[]
  notConfiguredCapabilities: string[]
  blockers: string[]
}

export interface CorrectionRerunPreview {
  ready: boolean
  plan: TargetedWorkflowRerunPlan
  latestStructuredAssetPath: string
  geometryPath?: string
  blockers: string[]
}

export interface CorrectionRerunResult {
  status: 'done' | 'blocked' | 'error'
  preview: CorrectionRerunPreview
  completedNodeIds: string[]
  error?: string
}

const CORRECTION_STAGE_CONFIG: Record<string, { label: string; extensionIds: string[] }[]> = {
  'part.membership': [
    { label: 'Part semantics', extensionIds: ['identify-part-semantics'] },
    { label: 'Material identity', extensionIds: ['reference-material-identity', 'project-owned-material-identity'] },
    { label: 'PBR estimation', extensionIds: ['project-owned-pbr-estimation'] },
  ],
  'part.semantic-label': [
    { label: 'Part semantics', extensionIds: ['identify-part-semantics'] },
  ],
  'material.identity': [
    { label: 'Material identity', extensionIds: ['reference-material-identity', 'project-owned-material-identity'] },
    { label: 'PBR estimation', extensionIds: ['project-owned-pbr-estimation'] },
  ],
  'material.name': [
    { label: 'Material identity', extensionIds: ['reference-material-identity', 'project-owned-material-identity'] },
  ],
}

function correctionStages(property: string): { label: string; extensionIds: string[] }[] | undefined {
  if (property.startsWith('pbr.') && property.length > 4) {
    return [{ label: 'PBR estimation', extensionIds: ['project-owned-pbr-estimation'] }]
  }
  if (property === 'part.membership.history') return CORRECTION_STAGE_CONFIG['part.membership']
  return CORRECTION_STAGE_CONFIG[property]
}

function workflowTopologicalOrder(workflow: Workflow): WFNode[] | undefined {
  const nodes = new Map(workflow.nodes.map((node) => [node.id, node]))
  if (nodes.size !== workflow.nodes.length) return undefined
  const indegree = new Map(workflow.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map(workflow.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of workflow.edges) {
    if (!nodes.has(edge.source) || !nodes.has(edge.target)) return undefined
    outgoing.get(edge.source)!.push(edge.target)
    indegree.set(edge.target, indegree.get(edge.target)! + 1)
  }
  const ready = workflow.nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id)
  const ordered: WFNode[] = []
  while (ready.length > 0) {
    const id = ready.shift()!
    ordered.push(nodes.get(id)!)
    for (const target of outgoing.get(id)!) {
      const next = indegree.get(target)! - 1
      indegree.set(target, next)
      if (next === 0) ready.push(target)
    }
  }
  return ordered.length === workflow.nodes.length ? ordered : undefined
}

/** Pure preview planner: map a saved correction to configured stages and their downstream dependents. */
export function planCorrectionWorkflowRerun(
  workflow: Workflow,
  allExtensions: WorkflowExtension[],
  correctionProperty: string,
): TargetedWorkflowRerunPlan {
  const blockers: string[] = []
  const config = correctionStages(correctionProperty)
  if (!correctionProperty.trim() || !config) blockers.push(`No rerun mapping is defined for correction property '${correctionProperty}'.`)
  const ordered = workflowTopologicalOrder(workflow)
  if (!ordered) blockers.push('The configured workflow has missing node references, duplicate node IDs, or a dependency cycle.')

  const extensionByNodeId = new Map(allExtensions.map((extension) => [extension.id, extension]))
  const matchedByCapability = (config ?? []).map((capability) => ({
    label: capability.label,
    nodes: workflow.nodes.filter((node) => {
      if (node.type !== 'extensionNode' || !node.data.enabled) return false
      const extension = extensionByNodeId.get(node.data.extensionId ?? '')
      return !!extension && capability.extensionIds.includes(extension.extensionId)
    }),
  }))
  const candidateNodes = matchedByCapability.flatMap((entry) => entry.nodes)
  const notConfiguredCapabilities = matchedByCapability
    .filter((entry) => entry.nodes.length === 0)
    .map((entry) => entry.label)
  if (candidateNodes.length === 0 && config?.length) {
    blockers.push('No enabled workflow node is configured for this correction. Add the required stage to the current workflow before rerunning.')
  }

  const adjacency = new Map(workflow.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of workflow.edges) {
    if (adjacency.has(edge.source) && adjacency.has(edge.target)) adjacency.get(edge.source)!.push(edge.target)
  }
  const reaches = (start: string, target: string): boolean => {
    const seen = new Set<string>()
    const stack = [...(adjacency.get(start) ?? [])]
    while (stack.length) {
      const current = stack.pop()!
      if (current === target) return true
      if (seen.has(current)) continue
      seen.add(current)
      stack.push(...(adjacency.get(current) ?? []))
    }
    return false
  }
  // If one correction stage feeds another, start at the earliest configured
  // stage and let the existing dataflow carry its fresh sidecar downstream.
  const sourceNodes = candidateNodes.filter((node) =>
    !candidateNodes.some((other) => other.id !== node.id && reaches(other.id, node.id)))
  const affected = new Set<string>()
  const stack = sourceNodes.map((node) => node.id)
  while (stack.length) {
    const id = stack.pop()!
    if (affected.has(id)) continue
    affected.add(id)
    stack.push(...(adjacency.get(id) ?? []))
  }

  const invalidateNodeIds = (ordered ?? workflow.nodes)
    .filter((node) => affected.has(node.id) && (node.type === 'extensionNode' || isIterator(node.type)))
    .map((node) => node.id)
  for (const node of workflow.nodes) {
    if (affected.has(node.id) && node.type === 'extensionNode' && !node.data.enabled) {
      blockers.push(`Downstream node '${node.id}' is disabled, so its stale output cannot be refreshed safely.`)
    }
    if (affected.has(node.id) && ['forEachNode', 'whileNode', 'waitNode'].includes(node.type)) {
      blockers.push(`Node '${node.id}' uses a loop or wait boundary; targeted rerun cannot safely reproduce that control flow.`)
    }
  }
  const rerunNodes = (ordered ?? workflow.nodes)
    .filter((node) => affected.has(node.id) && isExecutable(node))
    .map((node) => {
      const extensionId = node.data.extensionId ?? ''
      const extension = extensionByNodeId.get(extensionId)
      if (node.type === 'extensionNode' && (!extension || extension.type !== 'process'
        || extension.input !== 'mesh' || extension.output !== 'mesh')) {
        blockers.push(`Node '${node.id}' is not an available mesh process stage and cannot consume the corrected Structured Asset safely.`)
      }
      return {
        nodeId: node.id,
        label: typeof node.data.label === 'string' && node.data.label.trim()
          ? node.data.label.trim() : (extension?.name ?? node.id),
        extensionId,
      }
    })
  const sourceNodeIds = sourceNodes.map((node) => node.id)
  for (const source of sourceNodes) {
    const extension = extensionByNodeId.get(source.data.extensionId ?? '')
    if (!extension || extension.type !== 'process' || extension.input !== 'mesh' || extension.output !== 'mesh') {
      blockers.push(`Correction source '${source.id}' cannot accept the existing mesh and latest Structured Asset sidecar.`)
    }
  }
  return {
    correctionProperty,
    sourceNodeIds,
    rerunNodes,
    invalidateNodeIds,
    preserveNodeIds: workflow.nodes.filter((node) => !affected.has(node.id)).map((node) => node.id),
    notConfiguredCapabilities,
    blockers: [...new Set(blockers)],
  }
}

function normalizePathForComparison(path: string): string {
  return path.replace(/\\/g, '/').replace(/\/+$/, '')
}

function isWorkspaceRelativePath(path: unknown): path is string {
  return typeof path === 'string' && path.length > 0 && !path.includes('\0')
    && !path.startsWith('/') && !path.startsWith('\\') && !/^[A-Za-z]:/.test(path)
    && !path.replace(/\\/g, '/').split('/').some((part) => part === '' || part === '.' || part === '..')
}

/** A correction may target only the mesh currently shown by this workflow run. */
export function isCurrentSceneGeometry(
  workspaceDir: string,
  lastSceneMesh: string | undefined,
  absoluteGeometryPath: string,
): boolean {
  if (!lastSceneMesh?.startsWith('/workspace/')) return false
  const workspace = normalizePathForComparison(workspaceDir)
  const sceneRelativePath = lastSceneMesh.slice('/workspace/'.length)
  const activeSceneGeometryPath = `${workspace}/${sceneRelativePath}`
  return normalizePathForComparison(activeSceneGeometryPath) === normalizePathForComparison(absoluteGeometryPath)
}

interface StructuredAssetRerunInput {
  geometryPath: string
  absoluteGeometryPath: string
}

async function readTargetedRerunInput(ctx: RunContext, sidecarRelativePath: string): Promise<StructuredAssetRerunInput> {
  if (!isWorkspaceRelativePath(sidecarRelativePath)) throw new Error('The latest Structured Asset sidecar path is invalid or outside the workspace.')
  const workspace = normalizePathForComparison(ctx.workspaceDir)
  const absoluteSidecarPath = `${workspace}/${sidecarRelativePath.replace(/\\/g, '/')}`
  const sidecarB64 = await window.electron.fs.readFileBase64(absoluteSidecarPath)
  const sidecarBytes = Uint8Array.from(atob(sidecarB64), (char) => char.charCodeAt(0))
  const sidecar = JSON.parse(new TextDecoder().decode(sidecarBytes)) as Record<string, unknown>
  const schemaId = sidecar.schema_id
  const schemaVersion = sidecar.schema_version
  const geometry = sidecar.geometry
  if (schemaId !== 'org.modly.structured-asset' || schemaVersion !== '1.0.0'
    || !geometry || typeof geometry !== 'object' || Array.isArray(geometry)) {
    throw new Error('The selected sidecar is not a supported Structured Asset v1 record.')
  }
  const geometryRelativePath = (geometry as Record<string, unknown>).workspace_path
  if (!isWorkspaceRelativePath(geometryRelativePath)) throw new Error('The latest Structured Asset does not contain a safe geometry path.')
  const absoluteGeometryPath = `${workspace}/${geometryRelativePath.replace(/\\/g, '/')}`
  await window.electron.fs.readFileBase64(absoluteGeometryPath)
  if (!isCurrentSceneGeometry(ctx.workspaceDir, ctx.lastSceneMesh, absoluteGeometryPath)) {
    throw new Error('The corrected Structured Asset geometry does not match the current workflow scene output.')
  }
  const outputPath = (path: string): string => {
    if (isWorkspaceRelativePath(path)) return `${workspace}/${path.replace(/\\/g, '/')}`
    return normalizePathForComparison(path)
  }
  const existsInRun = [...ctx.nodeOutputs.values()].some((output) => output.filePath
    && outputPath(output.filePath) === normalizePathForComparison(absoluteGeometryPath))
  if (!existsInRun) {
    throw new Error('The corrected asset geometry is not an output of the current workflow run; targeted rerun would use stale or unrelated geometry.')
  }
  return { geometryPath: geometryRelativePath, absoluteGeometryPath }
}

function missingTargetedStageInputs(ctx: RunContext, plan: TargetedWorkflowRerunPlan): string[] {
  const blockers: string[] = []
  const planned = new Set(plan.rerunNodes.map((node) => node.nodeId))
  const sourceNodes = new Set(plan.sourceNodeIds)
  const completedEarlier = new Set<string>()
  for (const entry of plan.rerunNodes) {
    if (!sourceNodes.has(entry.nodeId)) {
      const edges = ctx.workflow.edges.filter((edge) => edge.target === entry.nodeId)
      if (edges.length === 0) {
        blockers.push(`Node '${entry.nodeId}' has no configured input connection.`)
      } else {
        let hasMeshInput = false
        for (const edge of edges) {
          const dataSource = resolveDataSource(edge.source, ctx.workflow.edges, ctx.nodeMap)
          const output = dataSource ? ctx.nodeOutputs.get(dataSource) : undefined
          if (dataSource && planned.has(dataSource)) {
            if (completedEarlier.has(dataSource)) hasMeshInput = true
            else blockers.push(`Node '${entry.nodeId}' depends on '${dataSource}', which is not available before it runs.`)
          } else if (output?.filePath) {
            hasMeshInput = true
          } else {
            blockers.push(`Node '${entry.nodeId}' has a missing input from '${edge.source}'.`)
          }
        }
        if (!hasMeshInput) blockers.push(`Node '${entry.nodeId}' has no available mesh input.`)
      }
    }
    completedEarlier.add(entry.nodeId)
  }
  return [...new Set(blockers)]
}

// ─── Topological sort (DFS preorder, branch-first) ───────────────────────────

function topoSort(nodes: WFNode[], edges: WFEdge[]): WFNode[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  const adj     = new Map(nodes.map((n) => [n.id, [] as string[]]))
  const inDeg   = new Map(nodes.map((n) => [n.id, 0]))
  for (const e of edges) {
    if (!nodeMap.has(e.source) || !nodeMap.has(e.target)) continue
    adj.get(e.source)!.push(e.target)
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1)
  }

  const visited = new Set<string>()
  const result: WFNode[] = []

  const visit = (id: string): void => {
    if (visited.has(id)) return
    for (const e of edges) {
      if (e.target === id && !visited.has(e.source) && nodeMap.has(e.source)) return
    }
    const node = nodeMap.get(id)
    if (!node) return
    visited.add(id)
    result.push(node)
    for (const childId of adj.get(id) ?? []) visit(childId)
  }

  for (const node of nodes) if ((inDeg.get(node.id) ?? 0) === 0) visit(node.id)
  for (const node of nodes) if (!visited.has(node.id)) visit(node.id)
  return result
}

// ─── While container geometry ──────────────────────────────────────────────────
// Body membership can't rely on parentId alone: React Flow only assigns it when a
// node is dragged into the container, so a While resized around existing nodes (or
// nodes added by palette click) leaves them unparented. We therefore also test
// on-canvas containment at run time.

interface WhileBounds { x: number; y: number; w: number; h: number }

function nodeSize(n: WFNode): { w: number; h: number } {
  const measured = (n as { measured?: { width?: number; height?: number } }).measured
  const styleW = n.style?.width
  const styleH = n.style?.height
  return {
    w: measured?.width  ?? n.width  ?? (typeof styleW === 'number' ? styleW : 200),
    h: measured?.height ?? n.height ?? (typeof styleH === 'number' ? styleH : 80),
  }
}

function whileBounds(w: WFNode): WhileBounds {
  const s = nodeSize(w)
  return { x: w.position.x, y: w.position.y, w: s.w, h: s.h }
}

function isInsideWhile(n: WFNode, whileId: string, b: WhileBounds): boolean {
  if (n.parentId === whileId) return true
  if (n.parentId) return false   // explicit child of another container
  const s = nodeSize(n)
  const cx = n.position.x + s.w / 2
  const cy = n.position.y + s.h / 2
  return cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h
}

// ─── Branch identification ────────────────────────────────────────────────────
// A node belongs to Wait W's branch if its single nearest upstream Wait is W
// (dominance). Nodes with no upstream Wait — or with multiple (merges) — execute
// in the pre-phase before any user pause.

function identifyBranches(workflow: Workflow): {
  preExecExtNodes: WFNode[]
  branches:        Map<string, WFNode[]>
  waitIds:         string[]
  parentWait:      Map<string, string | null>
  ordered:         WFNode[]
} {
  const ordered = topoSort(workflow.nodes, workflow.edges)
  const nodeMap = new Map(workflow.nodes.map((n) => [n.id, n]))
  const waitIds = ordered.filter((n) => isBranchStarter(n.type)).map((n) => n.id)

  // A node is owned by its single nearest upstream Wait (dominance). This lets
  // Wait → … → Wait chains nest: nodes after the 2nd Wait belong to it, not the 1st.
  const branchOwner = new Map<string, string>()
  for (const node of workflow.nodes) {
    if (isBranchStarter(node.type) || !isExecutable(node)) continue
    const nearest = nearestUpstreamWaits(node.id, workflow.edges, nodeMap)
    if (nearest.size === 1) branchOwner.set(node.id, [...nearest][0])
  }

  // Each Wait's parent = its own nearest upstream Wait (null if top-level).
  const parentWait = new Map<string, string | null>()
  for (const w of waitIds) {
    const nearest = nearestUpstreamWaits(w, workflow.edges, nodeMap)
    parentWait.set(w, nearest.size === 1 ? [...nearest][0] : null)
  }

  const branches = new Map<string, WFNode[]>()
  for (const w of waitIds) branches.set(w, [])
  const preExecExtNodes: WFNode[] = []
  for (const node of ordered) {
    if (!isExecutable(node)) continue
    const owner = branchOwner.get(node.id)
    if (owner) branches.get(owner)!.push(node)
    else preExecExtNodes.push(node)
  }

  return { preExecExtNodes, branches, waitIds, parentWait, ordered }
}

// ─── For Each iterator execution ───────────────────────────────────────────────
// Emits the current iteration's file. Image iterators emit an image path; text
// iterators read the file and emit its text. The iteration index comes from the
// loop's progress (its own node id keys the loop).

async function executeIteratorNode(
  node:        WFNode,
  ctx:         RunContext,
  setRunState: (updater: (s: WorkflowRunState) => WorkflowRunState) => void,
): Promise<void> {
  const files   = ctx.iteratorFiles.get(node.id) ?? []
  const current = useWorkflowRunStore.getState().whileProgress[node.id]?.current ?? 1
  const path    = files[current - 1]
  if (!path) throw new Error('For Each: no file for this iteration')

  const kind = iteratorConfig(node)
  const name = path.split(/[\\/]/).pop()
  setRunState((s) => ({ ...s, blockProgress: 30, blockStep: `Reading ${name}` }))

  if (kind.outputType === 'text') {
    const text = await readTextFile(path)
    ctx.nodeOutputs.set(node.id, { text, outputType: 'text' })
  } else {
    ctx.nodeOutputs.set(node.id, { filePath: path, outputType: kind.outputType })
  }
  setRunState((s) => ({ ...s, blockProgress: 100, blockStep: `Loaded ${name}` }))
}

// ─── Per-node execution ──────────────────────────────────────────────────────
// Resolves inputs (walking through Wait passthroughs), runs the extension
// (model or process), updates nodeOutputs, and pushes the mesh to the scene
// if it feeds an Add-to-Scene through Waits.

async function executeExtensionNode(
  node:        WFNode,
  ctx:         RunContext,
  setRunState: (updater: (s: WorkflowRunState) => WorkflowRunState) => void,
  options: { inputOverride?: NodeOutput; publishScene?: boolean } = {},
): Promise<void> {
  if (isIterator(node.type)) {
    await executeIteratorNode(node, ctx, setRunState)
    return
  }

  const { workflow, allExtensions, client, workspaceDir, nodeOutputs, nodeMap,
          selectedImagePath, selectedImageData } = ctx

  const ext = getWorkflowExtension(node.data.extensionId ?? '', allExtensions)
  // Freshest params at the moment the node starts (so loop iterations / Retry pick
  // up edits made while paused, not the values captured at run start).
  const liveParams = _liveParams.current.get(node.id) ?? node.data.params ?? {}

  const resolveSource = (sourceId: string): NodeOutput | undefined => {
    const realId = resolveDataSource(sourceId, workflow.edges, nodeMap)
    return realId ? nodeOutputs.get(realId) : undefined
  }

  let nodeInputPath:     string | undefined
  let nodeInputText:     string | undefined
  let nodeInputMeshPath: string | undefined
  let nodeInputStructuredAssetPath: string | undefined
  let nodeInputStageArtifact: Record<string, unknown> | undefined
  let nodeStageTelemetry: WorkflowStageTelemetry[] = []
  // Per-slot texts for multi-text-input nodes (e.g. positive/negative prompts).
  // Indexed by target handle: input-0 → texts[0], input-1 → texts[1].
  const nodeInputTexts: (string | undefined)[] = []
  // Every image beyond the first resolved slot, for extensions that take several
  // (e.g. a texture node taking a mesh plus multiple reference images).
  const extraImagePaths: string[] = []

  const incomingEdges = workflow.edges.filter((e) => e.target === node.id)

  if (ext?.inputs && ext.inputs.length > 1) {
    const inputTypes  = ext.inputs
    // Resolved by target handle first, then typed by that slot's declared input --
    // not by the arrival order of `incomingEdges`, which does not match slot order.
    const inputPaths  = new Array<string | undefined>(inputTypes.length).fill(undefined)

    for (const edge of incomingEdges) {
      const src = resolveSource(edge.source)
      if (!src) continue
      const slotMatch = /^input-(\d+)$/.exec(edge.targetHandle ?? '')
      const slot = slotMatch ? Number(slotMatch[1]) : 0
      if (src.filePath !== undefined && slot < inputTypes.length) inputPaths[slot] = src.filePath
      if (src.structuredAssetPath !== undefined && inputTypes[slot] === 'mesh') {
        nodeInputStructuredAssetPath = src.structuredAssetPath
      }
      if (src.text !== undefined && src.text.trim().length > 0) {
        nodeInputText = src.text
        if (slotMatch) nodeInputTexts[slot] = src.text
      }
    }

    for (let i = 0; i < inputTypes.length; i++) {
      const fp = inputPaths[i]
      if (!fp) continue
      if (inputTypes[i] === 'mesh') {
        nodeInputMeshPath = fp
      } else if (inputTypes[i] === 'image') {
        if (!nodeInputPath) nodeInputPath = fp
        else extraImagePaths.push(fp)
      }
    }
  } else {
    for (const edge of incomingEdges) {
      const src = resolveSource(edge.source)
      if (src?.filePath !== undefined) nodeInputPath = src.filePath
      if (src?.text !== undefined && src.text.trim().length > 0) nodeInputText = src.text
      if (src?.structuredAssetPath !== undefined) nodeInputStructuredAssetPath = src.structuredAssetPath
    }
  }

  if (options.inputOverride) {
    if (ext?.type !== 'process' || ext.input !== 'mesh' || !options.inputOverride.filePath
      || !options.inputOverride.structuredAssetPath) {
      throw new Error(`Node '${node.id}' cannot consume the corrected mesh and Structured Asset sidecar.`)
    }
    nodeInputMeshPath = options.inputOverride.filePath
    nodeInputPath = options.inputOverride.filePath
    nodeInputStructuredAssetPath = options.inputOverride.structuredAssetPath
  }

  const isModelNode = ext?.type === 'model'

  if (isModelNode) {
    const isTextInput = ext?.inputs ? ext.inputs.every((i) => i === 'text') : ext?.input === 'text'
    const activeImagePath = isTextInput ? undefined : (nodeInputPath ?? selectedImagePath)
    if (!isTextInput && !selectedImageData && (!activeImagePath || activeImagePath.trim().length === 0)) {
      throw new Error('No input image selected for model node')
    }

    let blob: Blob
    let fname: string
    if (isTextInput || (selectedImageData && nodeInputPath === undefined)) {
      const base64 = selectedImageData && nodeInputPath === undefined
        ? selectedImageData
        : 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==' // 1x1 transparent PNG
      fname = 'placeholder.png'
      blob = new Blob([Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))], { type: 'image/png' })
    } else {
      const base64 = await window.electron.fs.readFileBase64(activeImagePath as string)
      const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
      blob = new Blob([bytes], { type: 'image/png' })
      fname = activeImagePath?.split(/[\\/]/).pop() ?? 'image.png'
    }

    const extraParams: Record<string, unknown> = {}
    if (nodeInputMeshPath) {
      const norm = nodeInputMeshPath.replace(/\\/g, '/')
      extraParams.mesh_path = norm.startsWith(workspaceDir)
        ? norm.slice(workspaceDir.length).replace(/^\//, '')
        : norm
    }
    if (nodeInputStructuredAssetPath) {
      extraParams.structured_asset_path = nodeInputStructuredAssetPath
    }
    if (nodeInputText !== undefined && nodeInputText.trim().length > 0) {
      extraParams.prompt = nodeInputText
      extraParams.text   = nodeInputText
    }
    if (extraImagePaths.length > 0) {
      extraParams.extra_image_paths = extraImagePaths
    }

    const schemaDefaults = Object.fromEntries(
      (ext.params ?? []).map((p) => [p.id, p.default]),
    )
    const effectiveParams = { ...schemaDefaults, ...liveParams }

    const fd = new FormData()
    fd.append('image', blob, fname)
    fd.append('model_id', node.data.extensionId ?? '')
    fd.append('collection', 'Workflows')
    fd.append('remesh', 'none')
    fd.append('enable_texture', 'false')
    fd.append('texture_resolution', '1024')
    fd.append('params', JSON.stringify({ ...effectiveParams, ...extraParams }))

    setRunState((s) => ({ ...s, blockProgress: 5, blockStep: 'Submitting to model…' }))

    const { data } = await client.post<{ job_id: string }>(
      '/generate/from-image', fd,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    )
    _activeJobId.current = data.job_id

    while (true) {
      if (_cancel.current) {
        await client.post(`/generate/cancel/${_activeJobId.current}`).catch(() => {})
        _activeJobId.current = null
        throw new Error('Cancelled')
      }
      await new Promise((r) => setTimeout(r, 1200))

      const { data: st } = await client.get<{
        status: string; progress?: number; step?: string; output_url?: string; error?: string
      }>(`/generate/status/${_activeJobId.current}`)

      if (st.status === 'done' && st.output_url) {
        const rel = st.output_url.replace(/^\/workspace\//, '')
        nodeInputPath = `${workspaceDir}/${rel}`
        // Model generation creates a new topology, so an upstream sidecar cannot
        // be propagated unless the generator explicitly emits a replacement.
        nodeInputStructuredAssetPath = undefined
        _activeJobId.current = null
        setRunState((s) => ({ ...s, blockProgress: 100, blockStep: 'Generation complete' }))
        break
      }
      if (st.status === 'error') throw new Error(st.error ?? 'Generation failed')

      setRunState((s) => ({ ...s, blockProgress: st.progress ?? s.blockProgress, blockStep: st.step ?? 'Generating…' }))
      useAppStore.getState().updateCurrentJob({ status: 'generating', progress: st.progress, step: st.step })
    }
  } else {
    if (ext?.input === 'mesh'  && !nodeInputPath) throw new Error(`${ext.name} needs an incoming mesh connection`)
    if (ext?.input === 'image' && !nodeInputPath) throw new Error(`${ext.name} needs an incoming image connection`)
    if (ext?.input === 'audio' && !nodeInputPath) throw new Error(`${ext.name} needs an incoming audio connection`)
    if (ext?.input === 'text'  && !nodeInputText) throw new Error(`${ext.name} needs an incoming text connection`)

    const parts  = (node.data.extensionId ?? '').split('/')
    const extId  = parts[0]
    const nid    = parts[1] ?? ''

    // Freshest params (liveParams), same as the model-node branch above, so a
    // multi-image process node re-run inside a loop also picks up edits made while
    // paused rather than the snapshot captured at run start.
    const processParams: Record<string, unknown> = { ...liveParams }
    if (nodeInputMeshPath && nodeInputPath) {
      // Texture node: mesh is filePath, all images in extra_image_paths
      processParams.extra_image_paths = [nodeInputPath, ...extraImagePaths]
    } else if (extraImagePaths.length > 0) {
      processParams.extra_image_paths = extraImagePaths
    }

    _activeProcessExtensionId.current = extId
    let result: Awaited<ReturnType<typeof window.electron.extensions.runProcess>>
    try {
      result = await window.electron.extensions.runProcess(
        extId,
        {
          filePath: nodeInputMeshPath ?? nodeInputPath,
          text:     nodeInputText,
          texts:    nodeInputTexts.length > 0 ? nodeInputTexts : undefined,
          structuredAssetPath: nodeInputStructuredAssetPath,
          nodeId:   nid,
        },
        processParams,
      )
    } finally {
      if (_activeProcessExtensionId.current === extId) _activeProcessExtensionId.current = null
    }
    if (!result.success) {
      if (result.telemetry) {
        const failedTelemetry = telemetryRows(result.telemetry, {
          nodeId: node.id,
          nodeLabel: (typeof node.data.label === 'string' && node.data.label.trim())
            ? node.data.label.trim() : (ext?.name || node.id),
          stageId: node.data.extensionId || node.id,
          evidencePath: result.evidencePath,
        }, true)
        setRunState((s) => ({ ...s, stageTelemetry: [
          ...(s.stageTelemetry ?? []).filter((row) => !failedTelemetry.some((fresh) => fresh.stageId === row.stageId)),
          ...failedTelemetry,
        ] }))
      }
      throw new Error(result.error ?? (result.cancelled ? 'Process extension cancelled' : 'Process extension failed'))
    }
    const processResult = result.result as Record<string, unknown> | undefined
    nodeInputPath = result.result?.filePath ?? nodeInputPath
    nodeInputText = result.result?.text     ?? nodeInputText
    // A process extension may change topology; never attach an upstream sidecar
    // to its output unless it explicitly returns a new structured-asset path.
    nodeInputStructuredAssetPath = result.result?.structuredAssetPath
    nodeInputStageArtifact = result.result?.stageOutputArtifact
    const telemetryBase = {
      nodeId: node.id,
      nodeLabel: (typeof node.data.label === 'string' && node.data.label.trim())
        ? node.data.label.trim() : (ext?.name || node.id),
      stageId: node.data.extensionId || node.id,
    }
    const directReports = [
      processResult?.qualitySummary,
      processResult?.stageOutputArtifact,
      processResult?.classification,
      processResult?.telemetry,
      processResult?.cacheReuse && typeof processResult.cacheReuse === 'object'
        ? {
          stage_id: `${node.data.extensionId || node.id}:cache`,
          backend: 'content-addressed-cache',
          status: (processResult.cacheReuse as Record<string, unknown>).state,
          cache_key: (processResult.cacheReuse as Record<string, unknown>).key,
        }
        : undefined,
    ]
    for (const report of directReports) nodeStageTelemetry.push(...telemetryRows(report, telemetryBase))
    nodeStageTelemetry.push(...await readStructuredStageTelemetry(processResult?.structuredAsset, workspaceDir, telemetryBase))
    // Keep the latest report for each stage in loops; do not present repeated loop
    // iterations as separate stages or combine their peak measurements.
    nodeStageTelemetry = [...new Map(nodeStageTelemetry.map((row) => [row.stageId, row])).values()]
    setRunState((s) => ({
      ...s,
      stageTelemetry: [
        ...(s.stageTelemetry ?? []).filter((row) => !nodeStageTelemetry.some((fresh) => fresh.stageId === row.stageId)),
        ...nodeStageTelemetry,
      ],
    }))
    setRunState((s) => ({ ...s, blockProgress: 100, blockStep: 'Done' }))
  }

  const outputType = ext?.output ?? (nodeInputPath ? 'mesh' : undefined)
  nodeOutputs.set(node.id, {
    filePath: nodeInputPath,
    text: nodeInputText,
    outputType,
    structuredAssetPath: nodeInputStructuredAssetPath,
    stageOutputArtifact: nodeInputStageArtifact,
    stageTelemetry: nodeStageTelemetry,
  })

  const output = nodeOutputs.get(node.id)
  const url = isSceneMeshOutput(output) ? toWorkspaceUrl(output.filePath, workspaceDir) : undefined
  if (isSceneMeshOutput(output) && url && reachesSceneOutput(node.id, workflow.edges, nodeMap)) {
    ctx.lastSceneMesh = url   // remember it so finalize() keeps the last-run branch in view
    ctx.lastSceneMeshStructuredAssetPath = output.structuredAssetPath
    if (options.publishScene !== false) {
      useAppStore.getState().updateCurrentJob({ status: 'done', progress: 100, outputUrl: url, structuredAssetPath: output.structuredAssetPath })
    }
  }
}

// ─── Wait dependency helpers ───────────────────────────────────────────────────

/** All Waits nested (transitively) under `rootId`, via the parentWait chain. */
function descendantWaits(rootId: string, ctx: RunContext): Set<string> {
  const out = new Set<string>()
  let frontier = new Set<string>([rootId])
  while (frontier.size > 0) {
    const next = new Set<string>()
    for (const w of ctx.waitIds) {
      const parent = ctx.parentWait.get(w)
      if (parent && frontier.has(parent) && !out.has(w)) { out.add(w); next.add(w) }
    }
    frontier = next
  }
  return out
}

/**
 * Push the mesh of every scene output owned by `waitId`'s branch to the viewer.
 * A branch whose only scene output has no in-branch processing (e.g. Wait → Add
 * to Scene) gets no immediate push during execution, so the display has to be
 * driven here, when the user continues that branch.
 */
function pushBranchSceneMesh(ctx: RunContext, waitId: string): void {
  for (const node of ctx.ordered) {
    if (!isSceneOutput(node.type)) continue
    const owners = nearestUpstreamWaits(node.id, ctx.workflow.edges, ctx.nodeMap)
    if (owners.size !== 1 || [...owners][0] !== waitId) continue
    const inEdge = ctx.workflow.edges.find((e) => e.target === node.id)
    if (!inEdge) continue
    const srcId = resolveDataSource(inEdge.source, ctx.workflow.edges, ctx.nodeMap)
    const sourceOutput = srcId ? ctx.nodeOutputs.get(srcId) : undefined
    const url = isSceneMeshOutput(sourceOutput) ? toWorkspaceUrl(sourceOutput.filePath, ctx.workspaceDir) : undefined
    if (isSceneMeshOutput(sourceOutput) && url) {
      ctx.lastSceneMesh = url
      ctx.lastSceneMeshStructuredAssetPath = sourceOutput.structuredAssetPath
      useAppStore.getState().updateCurrentJob({ status: 'done', progress: 100, outputUrl: url, structuredAssetPath: sourceOutput.structuredAssetPath })
    }
  }
}

// ─── Store ────────────────────────────────────────────────────────────────────

interface WorkflowRunStore {
  runState:         WorkflowRunState
  activeNodeId:     string | null
  activeWorkflowId: string | null
  nodeImageOutputs: Record<string, string>
  waitStates:       Record<string, WaitState>
  runningBranchId:  string | null
  /** whileId → current iteration / total (total null = manual/unbounded) */
  whileProgress:    Record<string, { current: number; total: number | null }>
  /** iterator ids paused together at a shared boundary (lockstep For Each group) */
  pausedGroup:      string[]

  run:         (workflow: Workflow, allExtensions: WorkflowExtension[], overrideImageData?: string) => Promise<void>
  cancel:      () => void
  reset:       () => void
  continueRun: (waitId: string) => Promise<void>
  /** While container: resume past the loop (Continue) */
  continueWhile: () => void
  /** While container: resume and re-run the loop body once more (Retry) */
  retryWhile:    () => void
  /** For Each container: request a pause at the next file boundary */
  pauseWhile:    () => void
  /** UI → runner: push the latest params for a node so a looping run uses them */
  setLiveNodeParams: (nodeId: string, params: Record<string, unknown>) => void
  /** Preview correction-dependent stages using only the completed active workflow context. */
  previewCorrectionRerun: (correctionProperty: string, latestStructuredAssetPath: string) => Promise<CorrectionRerunPreview>
  /** Execute the freshly planned stages against the corrected sidecar and unchanged geometry. */
  rerunForCorrection: (correctionProperty: string, latestStructuredAssetPath: string) => Promise<CorrectionRerunResult>
}

export const useWorkflowRunStore = create<WorkflowRunStore>((set, get) => {
  const setRunState = (updater: (s: WorkflowRunState) => WorkflowRunState): void => {
    set((s) => ({ runState: updater(s.runState) }))
  }

  const collectImageOutputs = (ctx: RunContext): Record<string, string> => {
    const out: Record<string, string> = {}
    for (const [nodeId, o] of ctx.nodeOutputs) {
      if (o.outputType === 'image' && o.filePath) {
        const norm = o.filePath.replace(/\\/g, '/')
        if (norm.startsWith(ctx.workspaceDir)) {
          out[nodeId] = `/workspace/${norm.slice(ctx.workspaceDir.length).replace(/^\//, '')}`
        }
      }
    }
    return out
  }

  const finalize = (ctx: RunContext, finalWaitStates?: Record<string, WaitState>): void => {
    // Prefer the mesh of the last branch the user actually ran — it's already in the
    // viewer, and topo order must not override the user's last action.
    let outputUrl:  string | undefined = ctx.lastSceneMesh
    let outputPath: string | undefined
    let structuredAssetPath: string | undefined = ctx.lastSceneMeshStructuredAssetPath

    const lastOutputNode = outputUrl ? undefined : [...ctx.ordered].reverse().find((n) => isSceneOutput(n.type))
    if (lastOutputNode) {
      for (const edge of ctx.workflow.edges.filter((e) => e.target === lastOutputNode.id)) {
        const src = ctx.nodeOutputs.get(edge.source)
        if (isSceneMeshOutput(src)) {
          outputUrl = toWorkspaceUrl(src.filePath, ctx.workspaceDir)
          structuredAssetPath = src.structuredAssetPath
        }
      }
    }
    if (!outputUrl) {
      for (const [, o] of ctx.nodeOutputs) {
        if (o.filePath) {
          if (o.outputType === 'audio') {
            outputPath = o.filePath
            continue
          }
          const workspaceUrl = toWorkspaceUrl(o.filePath, ctx.workspaceDir)
          if (workspaceUrl) outputUrl = workspaceUrl
          else outputPath = o.filePath
          if (workspaceUrl && o.outputType === 'mesh') structuredAssetPath = o.structuredAssetPath
        }
      }
    }

    set((s) => ({
      activeNodeId:     null,
      runningBranchId:  null,
      whileProgress:    {},
      pausedGroup:      [],
      waitStates:       finalWaitStates ?? s.waitStates,
      nodeImageOutputs: collectImageOutputs(ctx),
      runState: {
        status:        'done',
        blockIndex:    0,
        blockTotal:    0,
        blockProgress: 100,
        blockStep:     'Done',
        outputUrl,
        outputPath,
        structuredAssetPath,
        stageTelemetry: get().runState.stageTelemetry ?? [],
        stageTelemetryOutputUrl: outputUrl,
      },
    }))
    useAppStore.getState().updateCurrentJob({ status: 'done', progress: 100, outputUrl, structuredAssetPath })
    void showCompletionNotification('Workflow run complete')
  }

  return {
    runState:         IDLE,
    activeNodeId:     null,
    activeWorkflowId: null,
    nodeImageOutputs: {},
    waitStates:       {},
    runningBranchId:  null,
    whileProgress:    {},
    pausedGroup:      [],

    async previewCorrectionRerun(correctionProperty, latestStructuredAssetPath) {
      const ctx = _ctx.current
      const basePlan = ctx
        ? planCorrectionWorkflowRerun(ctx.workflow, ctx.allExtensions, correctionProperty)
        : {
            correctionProperty, sourceNodeIds: [], rerunNodes: [], invalidateNodeIds: [], preserveNodeIds: [],
            notConfiguredCapabilities: [], blockers: [],
          }
      const blockers = [...basePlan.blockers]
      const state = get()
      if (!ctx) blockers.push('There is no active workflow context. Run the workflow first, then save the correction and preview again.')
      else {
        if (state.activeWorkflowId !== ctx.workflow.id) blockers.push('The completed workflow context does not match the currently active workflow.')
        if (state.runState.status !== 'done') blockers.push('Targeted rerun requires a completed workflow with no active stage or branch.')
        if (state.runningBranchId !== null || state.activeNodeId !== null) blockers.push('A workflow stage or branch is still active.')
        if (!blockers.length) blockers.push(...missingTargetedStageInputs(ctx, basePlan))
      }
      let geometryPath: string | undefined
      if (ctx && !blockers.length) {
        try {
          const input = await readTargetedRerunInput(ctx, latestStructuredAssetPath)
          geometryPath = input.geometryPath
        } catch (error) {
          blockers.push(error instanceof Error ? error.message : 'The corrected asset inputs could not be read.')
        }
      }
      return {
        ready: blockers.length === 0,
        plan: { ...basePlan, blockers: [...new Set(blockers)] },
        latestStructuredAssetPath,
        geometryPath,
        blockers: [...new Set(blockers)],
      }
    },

    async rerunForCorrection(correctionProperty, latestStructuredAssetPath) {
      const preview = await get().previewCorrectionRerun(correctionProperty, latestStructuredAssetPath)
      if (!preview.ready) return { status: 'blocked', preview, completedNodeIds: [] }
      const ctx = _ctx.current
      if (!ctx) {
        const blocked = { ...preview, ready: false, blockers: [...preview.blockers, 'The workflow context ended before execution.'] }
        return { status: 'blocked', preview: blocked, completedNodeIds: [] }
      }
      // Re-read and verify inputs immediately before mutation/execution. Preview
      // state is informational and may be stale by the time the user confirms.
      let input: StructuredAssetRerunInput
      try {
        input = await readTargetedRerunInput(ctx, latestStructuredAssetPath)
      } catch (error) {
        const blocked = { ...preview, ready: false, blockers: [error instanceof Error ? error.message : 'The corrected asset inputs could not be read.'] }
        return { status: 'blocked', preview: blocked, completedNodeIds: [] }
      }
      const rerunIds = new Set(preview.plan.rerunNodes.map((node) => node.nodeId))
      const sourceIds = new Set(preview.plan.sourceNodeIds)
      const oldSceneMesh = ctx.lastSceneMesh
      const oldSceneSidecar = ctx.lastSceneMeshStructuredAssetPath
      const sceneWillChange = preview.plan.rerunNodes.some((node) => reachesSceneOutput(node.nodeId, ctx.workflow.edges, ctx.nodeMap))
      const priorOutputUrl = get().runState.outputUrl
      const priorOutputSidecar = get().runState.structuredAssetPath
      if (sceneWillChange) {
        ctx.lastSceneMesh = undefined
        ctx.lastSceneMeshStructuredAssetPath = undefined
      }
      for (const nodeId of preview.plan.invalidateNodeIds) ctx.nodeOutputs.delete(nodeId)
      _cancel.current = false
      set((state) => ({
        activeNodeId: null,
        runningBranchId: null,
        runState: {
          ...state.runState,
          status: 'running',
          blockIndex: 0,
          blockTotal: preview.plan.rerunNodes.length,
          blockProgress: 0,
          blockStep: 'Rerunning corrected stages…',
          error: undefined,
          stageTelemetry: (state.runState.stageTelemetry ?? []).filter((row) => !rerunIds.has(row.nodeId)),
        },
      }))
      const completedNodeIds: string[] = []
      try {
        for (let index = 0; index < preview.plan.rerunNodes.length; index++) {
          if (_cancel.current) throw new Error('Targeted workflow rerun was cancelled.')
          if (_ctx.current !== ctx) throw new Error('The active workflow context changed before the targeted rerun completed.')
          const entry = preview.plan.rerunNodes[index]
          const node = ctx.nodeMap.get(entry.nodeId)
          if (!node || !isExecutable(node)) throw new Error(`Planned workflow node '${entry.nodeId}' is no longer available.`)
          set((state) => ({
            activeNodeId: node.id,
            runState: { ...state.runState, blockIndex: index, blockProgress: 0, blockStep: `Starting ${entry.label}…` },
          }))
          await executeExtensionNode(node, ctx, setRunState, {
            inputOverride: sourceIds.has(node.id) ? {
              filePath: input.absoluteGeometryPath,
              outputType: 'mesh',
              structuredAssetPath: latestStructuredAssetPath,
            } : undefined,
            publishScene: false,
          })
          completedNodeIds.push(node.id)
        }
        if (_ctx.current !== ctx) throw new Error('The active workflow context changed before the targeted rerun completed.')
        if (sceneWillChange && !ctx.lastSceneMesh) {
          throw new Error('The rerun finished without producing a fresh mesh for the affected scene output.')
        }
        const outputUrl = ctx.lastSceneMesh ?? (sceneWillChange ? undefined : priorOutputUrl)
        const structuredAssetPath = ctx.lastSceneMeshStructuredAssetPath ?? (sceneWillChange ? undefined : priorOutputSidecar)
        set((state) => ({
          activeNodeId: null,
          runningBranchId: null,
          nodeImageOutputs: collectImageOutputs(ctx),
          runState: {
            ...state.runState,
            status: 'done',
            blockIndex: preview.plan.rerunNodes.length,
            blockTotal: preview.plan.rerunNodes.length,
            blockProgress: 100,
            blockStep: 'Corrected stages complete',
            outputUrl,
            structuredAssetPath,
            stageTelemetryOutputUrl: outputUrl,
          },
        }))
        useAppStore.getState().updateCurrentJob({ status: 'done', progress: 100, outputUrl, structuredAssetPath })
        return { status: 'done', preview, completedNodeIds }
      } catch (error) {
        // Drop every affected output, including successfully rerun ancestors, so
        // no partial chain can be mistaken for a completed corrected result.
        for (const nodeId of preview.plan.invalidateNodeIds) ctx.nodeOutputs.delete(nodeId)
        ctx.lastSceneMesh = oldSceneMesh
        ctx.lastSceneMeshStructuredAssetPath = oldSceneSidecar
        const message = error instanceof Error ? error.message : String(error)
        if (_ctx.current !== ctx) return { status: 'error', preview, completedNodeIds, error: message }
        set((state) => ({
          activeNodeId: null,
          runningBranchId: null,
          runState: { ...state.runState, status: 'error', error: message, blockStep: 'Corrected stage rerun failed' },
        }))
        useAppStore.getState().updateCurrentJob({ status: 'error', error: message })
        return { status: 'error', preview, completedNodeIds, error: message }
      }
    },

    async run(workflow, allExtensions, overrideImageData?) {
      _cancel.current = false
      _pauseRequested.current = false
      // Seed live params from the snapshot; UI edits during the run override these.
      _liveParams.current = new Map(workflow.nodes.map((n) => [n.id, { ...(n.data.params ?? {}) }]))

      const appState = useAppStore.getState()
      const apiUrl   = appState.apiUrl

      const { preExecExtNodes, branches, waitIds, parentWait, ordered } = identifyBranches(workflow)
      const branchSteps = waitIds.reduce((acc, w) => acc + (branches.get(w)?.length ?? 0), 0)

      const nodeMap = new Map(workflow.nodes.map((n) => [n.id, n]))

      // ── For Each iterators → resolve their folders up front ────────────────────
      // The loop count is driven by the folder contents, so the listing must resolve
      // before the loop table (and its progress totals) below.
      const iteratorFiles = new Map<string, string[]>()
      for (const w of workflow.nodes) {
        if (!isIterator(w.type)) continue
        const dir = (w.data.params?.dir as string | undefined)?.trim()
        const fail = (msg: string, step: string): void => {
          set((s) => ({ runState: { ...s.runState, status: 'error', error: msg, blockStep: step }, activeNodeId: null }))
        }
        if (!dir) { fail('For Each: pick a folder first', 'No folder selected'); return }
        try {
          const files = await listIteratorFiles(dir, iteratorConfig(w).exts)
          if (files.length === 0) { fail(`For Each: no matching files in ${dir}`, 'Empty folder'); return }
          iteratorFiles.set(w.id, files)
        } catch (err) {
          fail(String(err), 'Failed to read folder'); return
        }
      }

      // ── Loop table ─────────────────────────────────────────────────────────────
      // While containers loop their geometric body N× (or manually). For Each
      // iterators loop the executable nodes reachable downstream, once per file.
      // Replays filter by bodyIds membership (not a contiguous range), so unrelated
      // pre-phase nodes sorting between body members aren't replayed.
      interface LoopInfo { whileId: string; kind: 'while' | 'forEach'; firstIdx: number; lastIdx: number; bodyIds: Set<string>; iterations: number | null }
      const loops: LoopInfo[] = []
      const indexOf = new Map(preExecExtNodes.map((n, i) => [n.id, i]))

      for (const w of workflow.nodes) {
        if (w.type !== 'whileNode') continue
        const bounds = whileBounds(w)
        const idxs = preExecExtNodes.reduce<number[]>((acc, n, idx) => {
          if (isInsideWhile(n, w.id, bounds)) acc.push(idx)
          return acc
        }, [])
        if (idxs.length === 0) continue
        const iters = Number(w.data?.iterations)
        loops.push({
          whileId:  w.id,
          kind:     'while',
          firstIdx: Math.min(...idxs),
          lastIdx:  Math.max(...idxs),
          bodyIds:  new Set(idxs.map((i) => preExecExtNodes[i].id)),
          iterations: Number.isFinite(iters) && iters > 0 ? Math.floor(iters) : null,
        })
      }

      for (const [iterId, files] of iteratorFiles) {
        const bodyIds = new Set([...reachableExecutable(iterId, workflow.edges, nodeMap)].filter((id) => indexOf.has(id)))
        const idxs = [...bodyIds].map((id) => indexOf.get(id)!)
        if (idxs.length === 0) continue
        loops.push({
          whileId:    iterId,
          kind:       'forEach',
          firstIdx:   Math.min(...idxs),
          lastIdx:    Math.max(...idxs),
          bodyIds,
          iterations: files.length,   // one pass per file
        })
      }
      const loopCounters = new Map(loops.map((l) => [l.whileId, l.iterations]))
      // Auto-mode loops replay their body N times; count the extra passes so the
      // progress total reflects the real work (manual loops stay unbounded). While
      // loops are independent. For Each loops sharing a boundary (same lastIdx) run
      // in lockstep over the union of their bodies, so that union is replayed once
      // per pass — count it once, for max(files) − 1 extra passes.
      const forEachGroups = new Map<number, LoopInfo[]>()
      let loopExtraSteps = 0
      for (const l of loops) {
        if (l.kind === 'while') {
          loopExtraSteps += l.iterations != null ? (l.iterations - 1) * l.bodyIds.size : 0
        } else {
          const g = forEachGroups.get(l.lastIdx) ?? []
          g.push(l)
          forEachGroups.set(l.lastIdx, g)
        }
      }
      for (const group of forEachGroups.values()) {
        const union = new Set<string>()
        let maxIter = 0
        for (const l of group) {
          l.bodyIds.forEach((id) => union.add(id))
          if (l.iterations != null) maxIter = Math.max(maxIter, l.iterations)
        }
        if (maxIter > 0) loopExtraSteps += (maxIter - 1) * union.size
      }
      const totalSteps = preExecExtNodes.length + branchSteps + loopExtraSteps

      const selectedImagePath = appState.selectedImagePath ?? undefined
      const selectedImageData = overrideImageData ?? appState.selectedImageData ?? undefined
      const currentMeshUrl    = appState.currentJob?.outputUrl

      set({
        activeWorkflowId: workflow.id,
        nodeImageOutputs: {},
        // Top-level Waits are pending; nested Waits start blocked until their parent finishes.
        waitStates:       Object.fromEntries(waitIds.map((id) => [id, parentWait.get(id) ? 'blocked' as WaitState : 'pending' as WaitState])),
        runningBranchId:  null,
        whileProgress:    Object.fromEntries(loops.map((l) => [l.whileId, { current: 1, total: l.iterations }])),
        pausedGroup:      [],
        runState: {
          status: 'running', blockIndex: 0, blockTotal: totalSteps,
          blockProgress: 0, blockStep: 'Starting…', stageTelemetry: [], stageTelemetryOutputUrl: undefined,
        },
      })

      appState.setCurrentJob({
        id:        crypto.randomUUID(),
        imageFile: selectedImagePath ?? '__workflow__',
        status:    'generating',
        progress:  0,
        createdAt: Date.now(),
      })

      try {
        const client       = axios.create({ baseURL: apiUrl })
        const settings     = await window.electron.settings.get()
        const workspaceDir = settings.workspaceDir.replace(/\\/g, '/')

        const tmpAbsPath = settings.workspaceDir.replace(/[\\/]+$/, '') + '/tmp'
        window.electron.fs.deleteDirectory(tmpAbsPath).catch(() => {})

        const nodeOutputs = new Map<string, NodeOutput>()

        // Pre-populate source nodes
        for (const node of ordered) {
          if (node.type === 'imageNode') {
            const fp = node.data.params?.filePath as string | undefined
            const resolvedPath = overrideImageData ? undefined : (fp ?? selectedImagePath ?? undefined)
            nodeOutputs.set(node.id, { filePath: resolvedPath, outputType: 'image' })
          }
          if (node.type === 'textNode') {
            nodeOutputs.set(node.id, { text: node.data.params?.text as string | undefined })
          }
          if (node.type === 'meshNode') {
            const source = node.data.params?.source as 'file' | 'current' | undefined
            if (source === 'current' && currentMeshUrl) {
              let meshFilePath: string
              if (currentMeshUrl.includes('serve-file?path=')) {
                const encoded = currentMeshUrl.split('serve-file?path=')[1]
                meshFilePath = decodeURIComponent(encoded).replace(/\\/g, '/')
              } else {
                const rel = currentMeshUrl.replace(/^\/workspace\//, '')
                meshFilePath = `${workspaceDir}/${rel}`
              }
              nodeOutputs.set(node.id, { filePath: meshFilePath, outputType: 'mesh' })
            } else {
              const fp = node.data.params?.filePath as string | undefined
              if (fp) nodeOutputs.set(node.id, { filePath: fp, outputType: 'mesh' })
            }
          }
        }

        const ctx: RunContext = {
          workflow, allExtensions, client, workspaceDir, selectedImagePath, selectedImageData,
          overrideImageData, nodeOutputs, nodeMap, ordered, branches, waitIds, parentWait, iteratorFiles,
        }
        _ctx.current = ctx

        // While the runner is replaying a loop body, this holds the body nodes of the
        // active loop (or the union of several For Each loops sharing a boundary);
        // re-iterations then execute only those members and skip everything else in
        // the range. null = first pass / no active loop (run all nodes once).
        let activeLoopBody: Set<string> | null = null

        // End-of-body handler for While containers. Called after each pre-phase node;
        // when the index is a loop's last body node, it either jumps back (auto N× or
        // Retry) or pauses for Continue/Retry. Returns the index to resume at,
        // 'cancel', or undefined to continue normally.
        const bumpWhileProgress = (whileId: string): void => set((s) => {
          const prev = s.whileProgress[whileId]
          return { whileProgress: { ...s.whileProgress, [whileId]: { current: (prev?.current ?? 1) + 1, total: prev?.total ?? null } } }
        }
        )

        const handleLoopEnd = async (idx: number): Promise<number | 'cancel' | undefined> => {
          const forEachLoops = loops.filter((l) => l.lastIdx === idx && l.kind === 'forEach')
          const whileLoop    = loops.find((l) => l.lastIdx === idx && l.kind === 'while')

          // ── For Each: run through every file automatically. Several iterators can
          // share the same downstream body (e.g. an image folder + a mesh folder both
          // feeding one node); they advance together, in lockstep. It only stops if
          // the user hit Pause; then Continue advances to the next file(s) and Retry
          // re-runs the current one. No forced pause at the end — it just finishes.
          if (forEachLoops.length > 0) {
            const groupBody = new Set<string>()
            let jumpTo = Infinity
            for (const loop of forEachLoops) {
              loop.bodyIds.forEach((id) => groupBody.add(id))
              jumpTo = Math.min(jumpTo, loop.firstIdx)
            }

            if (_pauseRequested.current) {
              _pauseRequested.current = false
              _retry.current = false
              // Pause every iterator of the group together (they show the same state).
              const groupIds = forEachLoops.map((l) => l.whileId)
              set({ activeNodeId: groupIds[0], runningBranchId: groupIds[0], pausedGroup: groupIds })
              setRunState((s) => ({ ...s, status: 'paused', blockStep: 'Paused — Continue or Retry' }))
              await new Promise<void>((resolve) => { _resume.current = resolve })
              if (_cancel.current) return 'cancel'
              set({ runningBranchId: null, pausedGroup: [] })
              setRunState((s) => ({ ...s, status: 'running' }))
              if (_retry.current) {   // re-run the current file(s), no advance
                _retry.current = false
                activeLoopBody = groupBody
                return jumpTo
              }
              // Continue → fall through to the normal advance below
            }

            // Advance every iterator that still has files left; they move together.
            let anyMore = false
            for (const loop of forEachLoops) {
              const remaining = loopCounters.get(loop.whileId)
              if (remaining != null && remaining > 1) {
                loopCounters.set(loop.whileId, remaining - 1)
                bumpWhileProgress(loop.whileId)
                anyMore = true
              }
            }
            if (anyMore) {
              setRunState((s) => ({ ...s, blockStep: 'Next file…' }))
              activeLoopBody = groupBody
              return jumpTo
            }
            activeLoopBody = null
            return undefined
          }

          if (!whileLoop) return undefined
          const remaining = loopCounters.get(whileLoop.whileId)

          // Auto mode with iterations left → loop back automatically.
          if (remaining != null && remaining > 1) {
            loopCounters.set(whileLoop.whileId, remaining - 1)
            bumpWhileProgress(whileLoop.whileId)
            setRunState((s) => ({ ...s, blockStep: `Looping… ${remaining - 1} left` }))
            activeLoopBody = whileLoop.bodyIds
            return whileLoop.firstIdx
          }
          // Otherwise the auto counter is exhausted (or it's manual mode): pause on
          // the While and wait for Continue (proceed) or Retry (run the body again).
          // runningBranchId blocks Wait branches while the pre-phase is parked here.
          _retry.current = false
          set({ activeNodeId: whileLoop.whileId, runningBranchId: whileLoop.whileId })
          setRunState((s) => ({ ...s, status: 'paused', blockStep: 'Loop finished — Continue or Retry' }))
          await new Promise<void>((resolve) => { _resume.current = resolve })
          if (_cancel.current) return 'cancel'
          set({ runningBranchId: null })
          setRunState((s) => ({ ...s, status: 'running' }))
          if (_retry.current) {
            _retry.current = false
            bumpWhileProgress(whileLoop.whileId)
            activeLoopBody = whileLoop.bodyIds
            return whileLoop.firstIdx
          }
          activeLoopBody = null   // Continue → resume normal forward execution
          return undefined
        }

        // Pre-phase: nodes that don't belong to any single branch (sources + merges).
        let stepsDone = 0
        for (let i = 0; i < preExecExtNodes.length; i++) {
          if (_cancel.current) { _ctx.current = null; set({ runState: IDLE, activeNodeId: null }); return }
          const node = preExecExtNodes[i]
          // During a loop replay, only re-run the active loop's body members.
          if (activeLoopBody !== null && !isMemberOfActiveLoop(activeLoopBody, node.id)) continue
          set((s) => ({
            activeNodeId: node.id,
            runState: { ...s.runState, blockIndex: stepsDone, blockProgress: 0, blockStep: 'Starting…' },
          }))
          await executeExtensionNode(node, ctx, setRunState)
          stepsDone++

          const jump = await handleLoopEnd(i)
          if (jump === 'cancel') { _ctx.current = null; set({ runState: IDLE, activeNodeId: null }); return }
          if (jump !== undefined) { i = jump - 1 }
        }

        if (waitIds.length > 0) {
          // Hand off to the user — branches run on demand via continueRun(id).
          set((s) => ({
            activeNodeId: null,
            runState: { ...s.runState, status: 'paused', blockStep: 'Pick a branch and click Continue' },
          }))
          return
        }

        finalize(ctx)
      } catch (err) {
        if (!_cancel.current) {
          set((s) => ({ runState: { ...s.runState, status: 'error', error: String(err) }, activeNodeId: null }))
          useAppStore.getState().updateCurrentJob({ status: 'error', error: String(err) })
        }
      }
    },

    async continueRun(waitId) {
      const state = get()
      if (state.runningBranchId !== null) return
      // Only runnable Waits: blocked (parent not done) and running are not.
      const ws = state.waitStates[waitId]
      if (ws !== 'pending' && ws !== 'done' && ws !== 'error') return
      // A pending Wait only runs after a clean handoff. If the run errored in the
      // pre-phase, it never handed off — don't start a branch with missing inputs.
      if (ws === 'pending' && state.runState.status === 'error') return
      const ctx = _ctx.current
      if (!ctx) {
        console.warn('continueRun: no active run context — was the module hot-reloaded mid-run?')
        return
      }

      const branch = ctx.branches.get(waitId) ?? []
      // Re-running a Wait invalidates everything downstream: descendant branches
      // were computed against the old output, so drop their outputs and reset
      // them to blocked until this branch produces a fresh result.
      const descendants = descendantWaits(waitId, ctx)

      _cancel.current = false

      // Reset outputs for this branch's nodes so Retry re-executes cleanly.
      for (const node of branch) ctx.nodeOutputs.delete(node.id)
      for (const d of descendants) for (const node of ctx.branches.get(d) ?? []) ctx.nodeOutputs.delete(node.id)

      set((s) => {
        const waitStates = { ...s.waitStates, [waitId]: 'running' as WaitState }
        for (const d of descendants) waitStates[d] = 'blocked'
        return {
          runningBranchId: waitId,
          waitStates,
          runState: { ...s.runState, status: 'running', blockIndex: 0, blockTotal: branch.length, blockProgress: 0, blockStep: branch.length === 0 ? 'Done' : 'Starting…' },
        }
      })

      const finishBranch = (next: WaitState, err?: string): void => {
        if (_cancel.current) return
        const newWaitStates = { ...get().waitStates, [waitId]: next }
        // Unblock nested Waits whose parent branch just finished, and push this
        // branch's scene output to the viewer.
        if (next === 'done') {
          for (const w of ctx.waitIds) {
            if (ctx.parentWait.get(w) === waitId && newWaitStates[w] === 'blocked') newWaitStates[w] = 'pending'
          }
          pushBranchSceneMesh(ctx, waitId)
        }
        // A failed branch can never feed its descendants — surface them as error
        // too, otherwise they stay 'blocked' and the run hangs on 'paused' forever.
        if (next === 'error') {
          for (const d of descendantWaits(waitId, ctx)) {
            if (newWaitStates[d] === 'blocked') newWaitStates[d] = 'error'
          }
        }
        const allFinished = ctx.waitIds.every((id) => newWaitStates[id] === 'done' || newWaitStates[id] === 'error')
        const anyError    = ctx.waitIds.some((id) => newWaitStates[id] === 'error')

        if (allFinished && !anyError) {
          finalize(ctx, newWaitStates)
        } else {
          set((s) => ({
            activeNodeId:    null,
            runningBranchId: null,
            waitStates:      newWaitStates,
            runState: {
              ...s.runState,
              status:    allFinished ? 'error' : 'paused',
              error:     anyError ? (err ?? s.runState.error) : undefined,
              blockStep: err ? `Branch failed: ${err}` : 'Pick a branch and click Continue',
            },
          }))
        }
      }

      try {
        for (let i = 0; i < branch.length; i++) {
          if (_cancel.current) return
          const node = branch[i]
          set((s) => ({
            activeNodeId: node.id,
            runState: { ...s.runState, blockIndex: i, blockProgress: 0, blockStep: 'Starting…' },
          }))
          await executeExtensionNode(node, ctx, setRunState)
        }
        finishBranch('done')
      } catch (err) {
        finishBranch('error', String(err))
      }
    },

    cancel() {
      _cancel.current = true
      if (_activeProcessExtensionId.current) {
        void window.electron.extensions.cancelProcess(_activeProcessExtensionId.current).catch(() => {})
        _activeProcessExtensionId.current = null
      }
      _pauseRequested.current = false
      flushResume()   // unblock a manual While pause so the run can tear down
      if (_activeJobId.current) {
        const apiUrl = useAppStore.getState().apiUrl
        axios.create({ baseURL: apiUrl }).post(`/generate/cancel/${_activeJobId.current}`).catch(() => {})
        _activeJobId.current = null
      }
      _ctx.current = null
      set({ runState: IDLE, activeNodeId: null, activeWorkflowId: null, nodeImageOutputs: {}, waitStates: {}, runningBranchId: null, whileProgress: {}, pausedGroup: [] })
      useAppStore.getState().setCurrentJob(null)
    },

    reset() {
      _ctx.current = null
      set({ runState: IDLE, activeNodeId: null, activeWorkflowId: null, nodeImageOutputs: {}, waitStates: {}, runningBranchId: null, whileProgress: {}, pausedGroup: [] })
    },

    continueWhile() {
      flushResume()
    },

    retryWhile() {
      _retry.current = true
      flushResume()
    },

    pauseWhile() {
      _pauseRequested.current = true
    },

    setLiveNodeParams(nodeId, params) {
      _liveParams.current.set(nodeId, params)
    },
  }
})
