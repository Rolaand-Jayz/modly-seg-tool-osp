export const STRUCTURED_ASSET_CAPABILITY_IDS = [
  'acquire-observations',
  'generate-geometry',
  'import-geometry',
  'segment-parts',
  'generate-refine-parts',
  'identify-part-semantics',
  'segment-material-regions',
  'classify-material-identity',
  'estimate-pbr-properties',
  'fuse-evidence',
  'validate-structured-asset',
  'repair-geometry',
  'export-structured-asset',
] as const

export type StructuredAssetCapabilityId = typeof STRUCTURED_ASSET_CAPABILITY_IDS[number]
export type StructuredAssetCapabilityInput = 'observation' | 'mesh' | 'structured-asset'
export type StructuredAssetCapabilityOutput = 'geometry' | 'structured-asset' | 'assertions' | 'stage-artifacts'
export type StructuredAssetAdapterTrust = 'builtin' | 'pinned-reference' | 'third-party-unpinned'

export interface StructuredAssetCapabilityDescriptor {
  capability_id: StructuredAssetCapabilityId
  contract_version: '1.0.0'
  inputs: StructuredAssetCapabilityInput[]
  outputs: StructuredAssetCapabilityOutput[]
  adapter_id: string
  adapter_revision: string
  adapter_trust: StructuredAssetAdapterTrust
  /** Manifest assertions are metadata only; they do not prove installed code or weights. */
  identity_status: 'declared-only'
  model_weights_id?: string
  model_weights_digest?: string
}

const INPUTS = new Set<StructuredAssetCapabilityInput>(['observation', 'mesh', 'structured-asset'])
const OUTPUTS = new Set<StructuredAssetCapabilityOutput>(['geometry', 'structured-asset', 'assertions', 'stage-artifacts'])
const CAPABILITY_IDS = new Set<string>(STRUCTURED_ASSET_CAPABILITY_IDS)
const IMMUTABLE_REVISION = /^(?:git:)?[0-9a-f]{40}$|^sha256:[0-9a-f]{64}$/
const SHA256_DIGEST = /^sha256:[0-9a-f]{64}$/

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringArray<T extends string>(value: unknown, allowed: Set<T>, field: string): T[] {
  if (!Array.isArray(value) || value.length === 0 || value.some((item) => typeof item !== 'string' || !allowed.has(item as T))) {
    throw new Error(`manifest.json: Structured Asset capability ${field} must be a non-empty array of supported values`)
  }
  if (new Set(value).size !== value.length) {
    throw new Error(`manifest.json: Structured Asset capability ${field} must not contain duplicates`)
  }
  return [...value] as T[]
}

export function validateStructuredAssetCapability(value: unknown): StructuredAssetCapabilityDescriptor {
  if (!isRecord(value)) {
    throw new Error('manifest.json: Structured Asset capability descriptor must be an object')
  }
  const allowedKeys = new Set([
    'capability_id', 'contract_version', 'inputs', 'outputs', 'adapter_id', 'adapter_revision',
    'adapter_trust', 'model_weights_id', 'model_weights_digest',
  ])
  const unknownKeys = Object.keys(value).filter((key) => !allowedKeys.has(key))
  if (unknownKeys.length) {
    throw new Error(`manifest.json: unsupported Structured Asset capability field "${unknownKeys[0]}"`)
  }

  const capabilityId = value.capability_id
  if (typeof capabilityId !== 'string' || !CAPABILITY_IDS.has(capabilityId)) {
    throw new Error(`manifest.json: unsupported Structured Asset capability id "${String(capabilityId ?? '')}"`)
  }
  if (value.contract_version !== '1.0.0') {
    throw new Error(`manifest.json: unsupported Structured Asset capability contract version "${String(value.contract_version ?? '')}"`)
  }
  for (const key of ['adapter_id', 'adapter_revision'] as const) {
    if (typeof value[key] !== 'string' || !value[key].trim()) {
      throw new Error(`manifest.json: Structured Asset capability ${key} must be non-empty text`)
    }
  }
  const trust = value.adapter_trust
  if (trust !== 'builtin' && trust !== 'pinned-reference' && trust !== 'third-party-unpinned') {
    throw new Error('manifest.json: Structured Asset capability adapter_trust is invalid')
  }
  if (trust === 'pinned-reference' && !IMMUTABLE_REVISION.test(value.adapter_revision as string)) {
    throw new Error('manifest.json: pinned-reference adapter_revision must be a full immutable commit or sha256 digest')
  }

  const weightsId = value.model_weights_id
  const weightsDigest = value.model_weights_digest
  if ((weightsId === undefined) !== (weightsDigest === undefined)) {
    throw new Error('manifest.json: Structured Asset model weights require both id and digest')
  }
  if (weightsId !== undefined && (typeof weightsId !== 'string' || !weightsId.trim())) {
    throw new Error('manifest.json: Structured Asset model_weights_id must be non-empty text')
  }
  if (weightsDigest !== undefined && (typeof weightsDigest !== 'string' || !SHA256_DIGEST.test(weightsDigest))) {
    throw new Error('manifest.json: Structured Asset model_weights_digest must be sha256:<64 lowercase hex>')
  }

  return {
    capability_id: capabilityId as StructuredAssetCapabilityId,
    contract_version: '1.0.0',
    inputs: stringArray(value.inputs, INPUTS, 'inputs'),
    outputs: stringArray(value.outputs, OUTPUTS, 'outputs'),
    adapter_id: (value.adapter_id as string).trim(),
    adapter_revision: (value.adapter_revision as string).trim(),
    adapter_trust: trust,
    identity_status: 'declared-only',
    ...(weightsId === undefined ? {} : { model_weights_id: (weightsId as string).trim() }),
    ...(weightsDigest === undefined ? {} : { model_weights_digest: weightsDigest as string }),
  }
}

export function validateStructuredAssetCapabilities(value: unknown): StructuredAssetCapabilityDescriptor[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) {
    throw new Error('manifest.json: capabilities must be an array when declared')
  }
  const descriptors = value.map(validateStructuredAssetCapability)
  const ids = descriptors.map((descriptor) => descriptor.capability_id)
  if (new Set(ids).size !== ids.length) {
    throw new Error('manifest.json: duplicate Structured Asset capability declarations are not allowed')
  }
  return descriptors
}
