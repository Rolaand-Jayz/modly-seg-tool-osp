import { createHash, randomUUID } from 'node:crypto'
import { link, lstat, mkdir, readFile, readdir, realpath, unlink, writeFile } from 'node:fs/promises'
import { join, relative, resolve, sep } from 'node:path'
import type { StructuredAssetCapabilityDescriptor } from '../../src/shared/types/structuredAssetCapability'

const CACHE_FORMAT = 2
const digest = (bytes: Buffer | string) => createHash('sha256').update(bytes).digest('hex')
const NO_MODEL_CAPABILITIES = new Set(['validate-structured-asset'])
const ALLOWED_RESULT_KEYS = new Set(['structuredAssetPath', 'structuredAsset', 'stageOutputArtifact', 'filePath', 'evidenceArtifact'])

function canonical(value: unknown): string {
  if (value === undefined) return '"$undefined"'
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null'
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  const record = value as Record<string, unknown>
  return `{${Object.keys(record).sort().map(key => `${JSON.stringify(key)}:${canonical(record[key])}`).join(',')}}`
}

async function treeDigest(root: string): Promise<string> {
  const entries: Array<[string, string]> = []
  async function visit(dir: string): Promise<void> {
    const children = await readdir(dir, { withFileTypes: true })
    children.sort((a, b) => a.name.localeCompare(b.name))
    for (const child of children) {
      if (child.name === '.git' || child.name === '__pycache__') continue
      const path = join(dir, child.name)
      const info = await lstat(path)
      // Dependency symlinks make the loaded executable tree ambiguous. Fail closed.
      if (info.isSymbolicLink()) throw new Error('extension tree contains a symbolic link')
      const rel = relative(root, path).split(sep).join('/')
      if (info.isDirectory()) await visit(path)
      else if (info.isFile()) entries.push([rel, digest(await readFile(path))])
    }
  }
  await visit(root)
  return digest(canonical(entries))
}

async function containedRealFile(workspace: string, workspacePath: unknown): Promise<string | null> {
  if (typeof workspacePath !== 'string' || !workspacePath || workspacePath.includes('\\')) return null
  try {
    const root = await realpath(workspace)
    const candidate = await realpath(resolve(root, workspacePath))
    return candidate.startsWith(`${root}${sep}`) ? candidate : null
  } catch { return null }
}

async function safeWorkspaceDirectory(workspace: string, relativePath: string): Promise<string | null> {
  const root = await realpath(workspace)
  const target = resolve(root, relativePath)
  if (!target.startsWith(`${root}${sep}`)) return null
  const parts = relative(root, target).split(sep)
  let current = root
  for (const part of parts) {
    current = join(current, part)
    try {
      const info = await lstat(current)
      if (info.isSymbolicLink() || !info.isDirectory()) return null
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') return null
      try { await mkdir(current) } catch (mkdirError) {
        if ((mkdirError as NodeJS.ErrnoException).code !== 'EEXIST') return null
      }
      const info = await lstat(current)
      if (info.isSymbolicLink() || !info.isDirectory()) return null
    }
  }
  try { return (await realpath(target)).startsWith(`${root}${sep}`) ? target : null } catch { return null }
}

export type ValidatedAsset = Record<string, unknown>
export type AuthoritativeAssetValidator = (sidecarPath: string) => Promise<ValidatedAsset | null>

async function validatedAsset(sidecarPath: string, workspace: string, validate: AuthoritativeAssetValidator): Promise<ValidatedAsset | null> {
  const safeSidecar = await containedRealFile(workspace, sidecarPath)
  if (!safeSidecar) return null
  try {
    const raw = JSON.parse(await readFile(safeSidecar, 'utf8')) as Record<string, unknown>
    if (raw.schema_id !== 'org.modly.structured-asset' || raw.schema_version !== '1.0.0'
        || !raw.geometry || typeof raw.geometry !== 'object') return null
    const validated = await validate(safeSidecar)
    if (!validated || validated.schema_id !== 'org.modly.structured-asset'
        || validated.schema_version !== '1.0.0'
        || (validated.validation_state !== 'valid' && validated.validation_state !== 'needs-review')) return null
    return validated
  } catch { return null }
}

export interface StructuredStageCacheRequest {
  workspaceDir: string
  extensionDir: string
  extensionId: string
  nodeId: string
  descriptor: StructuredAssetCapabilityDescriptor
  inputSidecarPath?: string
  params: Record<string, unknown>
  builtin: boolean
  pythonEntry: boolean
  hostVersion: string
  validateAsset: AuthoritativeAssetValidator
}

interface CacheEntry { format: number; key: string; asset: ValidatedAsset; result: Record<string, unknown> }

async function resultMatchesAsset(result: Record<string, unknown>, asset: ValidatedAsset, workspace: string): Promise<boolean> {
  if (Object.keys(result).some(field => !ALLOWED_RESULT_KEYS.has(field))
      || typeof result.structuredAssetPath !== 'string'
      || !result.structuredAsset || typeof result.structuredAsset !== 'object'
      || canonical(result.structuredAsset) !== canonical(asset)) return false
  const storedSidecar = await containedRealFile(workspace, result.structuredAssetPath)
  const geometryPath = asset.geometry && typeof asset.geometry === 'object'
    ? await containedRealFile(workspace, (asset.geometry as Record<string, unknown>).workspace_path) : null
  if (!storedSidecar || !geometryPath) return false
  if (result.filePath !== undefined && await containedRealFile(workspace, result.filePath) !== geometryPath) return false
  const stages = Array.isArray(asset.stage_artifacts) ? asset.stage_artifacts : []
  if (result.stageOutputArtifact !== undefined
      && !stages.some(stage => canonical(stage) === canonical(result.stageOutputArtifact)
        || (stage && typeof stage === 'object' && canonical((stage as Record<string, unknown>).artifact) === canonical(result.stageOutputArtifact)))) return false
  if (result.evidenceArtifact !== undefined
      && !stages.some(stage => stage && typeof stage === 'object'
        && canonical((stage as Record<string, unknown>).artifact) === canonical(result.evidenceArtifact))) return false
  return true
}

export async function structuredStageCacheKey(request: StructuredStageCacheRequest): Promise<string | null> {
  // The current desktop host can verify the exact JS dependency tree and Node
  // runtime. Python envs, external model caches, and declared (not resolved)
  // weight identities are intentionally unsupported until the host can attest them.
  if (!request.inputSidecarPath || !request.builtin || request.pythonEntry || !request.hostVersion
      || request.descriptor.adapter_trust !== 'builtin'
      || request.descriptor.capability_id !== 'validate-structured-asset'
      || !NO_MODEL_CAPABILITIES.has(request.descriptor.capability_id)
      || request.descriptor.model_weights_digest || request.descriptor.model_weights_id) return null
  const inputPath = await containedRealFile(request.workspaceDir, request.inputSidecarPath)
  if (!inputPath) return null
  const beforeValidation = await readFile(inputPath)
  const input = await validatedAsset(inputPath, request.workspaceDir, request.validateAsset)
  if (!input || !input.geometry || typeof input.geometry !== 'object') return null
  const afterValidation = await readFile(inputPath)
  if (!beforeValidation.equals(afterValidation)) return null
  const inputDigest = digest(beforeValidation)
  const executablePath = await realpath(request.extensionDir)
  const executableDigest = await treeDigest(executablePath)
  const keyData = {
    format: CACHE_FORMAT,
    input_sidecar_sha256: inputDigest,
    geometry_digest: (input.geometry as Record<string, unknown>).digest,
    extension_id: request.extensionId,
    node_id: request.nodeId,
    executable_tree_sha256: executableDigest,
    capability: request.descriptor,
    params: request.params,
    host_version: request.hostVersion,
    runtime: { node: process.version, platform: process.platform, arch: process.arch },
  }
  return digest(canonical(keyData))
}

async function readEntry(request: StructuredStageCacheRequest, key: string): Promise<CacheEntry | null> {
  const dir = await safeWorkspaceDirectory(request.workspaceDir, join('.modly-amd-runtime', 'structured-stage-cache', key))
  if (!dir) return null
  const entryPath = join(dir, 'entry.json')
  try {
    const entryInfo = await lstat(entryPath)
    if (entryInfo.isSymbolicLink() || !entryInfo.isFile()) return null
    const root = await realpath(request.workspaceDir)
    const actualEntryPath = await realpath(entryPath)
    if (!actualEntryPath.startsWith(`${root}${sep}`)) return null
    const entry = JSON.parse(await readFile(entryPath, 'utf8')) as CacheEntry
    if (entry.format !== CACHE_FORMAT || entry.key !== key || !entry.asset || !entry.result
        || typeof entry.result !== 'object' || Array.isArray(entry.result)) return null
    const outputPath = await containedRealFile(request.workspaceDir, entry.result.structuredAssetPath)
    const onDiskAsset = outputPath ? await validatedAsset(outputPath, request.workspaceDir, request.validateAsset) : null
    if (!onDiskAsset || canonical(onDiskAsset) !== canonical(entry.asset)) return null
    if (!await resultMatchesAsset(entry.result, onDiskAsset, request.workspaceDir)) return null
    return entry
  } catch { return null }
}

export async function readStructuredStageCache(request: StructuredStageCacheRequest, key: string) {
  const entry = await readEntry(request, key)
  if (!entry) return null
  const outputDir = await safeWorkspaceDirectory(request.workspaceDir, join('.modly-amd-runtime', 'structured-stage-cache', 'reused-assets'))
  if (!outputDir) return null
  const outputPath = join(outputDir, `${randomUUID()}.structured-asset.json`)
  try {
    await writeFile(outputPath, JSON.stringify(entry.asset, null, 2), { flag: 'wx' })
    const validated = await validatedAsset(outputPath, request.workspaceDir, request.validateAsset)
    if (!validated) { await unlink(outputPath).catch(() => undefined); return null }
    const workspaceRoot = await realpath(request.workspaceDir)
    return {
      result: { ...entry.result, structuredAssetPath: relative(workspaceRoot, outputPath).split(sep).join('/'), structuredAsset: validated },
      cache: { state: 'hit', key },
    }
  } catch { return null }
}

export async function writeStructuredStageCache(
  request: StructuredStageCacheRequest, key: string, result: Record<string, unknown>, outputPath: string,
): Promise<boolean> {
  if (!result.structuredAssetPath) return false
  const safeOutput = await containedRealFile(request.workspaceDir, outputPath)
  if (!safeOutput) return false
  const resultPath = await containedRealFile(request.workspaceDir, result.structuredAssetPath)
  if (resultPath !== safeOutput) return false
  const asset = await validatedAsset(safeOutput, request.workspaceDir, request.validateAsset)
  if (!asset || !await resultMatchesAsset(result, asset, request.workspaceDir)) return false
  const dir = await safeWorkspaceDirectory(request.workspaceDir, join('.modly-amd-runtime', 'structured-stage-cache', key))
  if (!dir) return false
  const entry: CacheEntry = { format: CACHE_FORMAT, key, asset, result: { ...result, structuredAsset: asset } }
  const target = join(dir, 'entry.json')
  const temp = join(dir, `.entry-${randomUUID()}.tmp`)
  try {
    await writeFile(temp, JSON.stringify(entry), { flag: 'wx' })
    try { await link(temp, target) } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
    }
  } catch {
    await unlink(temp).catch(() => undefined)
    return false
  }
  await unlink(temp).catch(() => undefined)
  // A claim of storage is made only after the immutable entry can be read and
  // its output sidecar passes the same live host validator.
  return (await readEntry(request, key)) !== null
}

export async function resolveWorkspaceStructuredAsset(workspaceDir: string, value: unknown): Promise<string | null> {
  if (typeof value !== 'string' || !value) return null
  return containedRealFile(workspaceDir, value)
}
