export type SeamRegionMapping = {
  region_id: string
  mapping: {
    topology_revision: string
    state: string
    element_type: string
    element_ids: number[]
  }
}

export type SeamCorrectionPreview = {
  topologyRevision: string
  sourceRegionId: string
  destinationRegionId: string
  moved: number[]
  sourceBefore: number[]
  sourceAfter: number[]
  destinationBefore: number[]
  destinationAfter: number[]
}

/** Validate and preview a face-membership correction without modifying the asset. */
export function buildSeamCorrectionPreview(args: {
  topologyRevision: string
  topologyFaceCount: number
  viewerFaceCount?: number
  source: SeamRegionMapping
  destination: SeamRegionMapping
  movedFaceIds: number[]
}): SeamCorrectionPreview {
  const { topologyRevision, topologyFaceCount, viewerFaceCount, source, destination } = args
  if (!topologyRevision || !Number.isSafeInteger(topologyFaceCount) || topologyFaceCount <= 0) {
    throw new Error('The Structured Asset has no valid topology face range.')
  }
  if (viewerFaceCount !== undefined && viewerFaceCount !== topologyFaceCount) {
    throw new Error('The loaded mesh face count does not match this Structured Asset topology.')
  }
  if (source.region_id === destination.region_id) throw new Error('Choose a different destination part.')
  for (const region of [source, destination]) {
    if (region.mapping.state !== 'valid' || region.mapping.element_type !== 'face') {
      throw new Error('Both regions need valid face mappings before seam edits.')
    }
    if (region.mapping.topology_revision !== topologyRevision) {
      throw new Error('A region mapping belongs to a different topology revision.')
    }
    if (new Set(region.mapping.element_ids).size !== region.mapping.element_ids.length ||
        region.mapping.element_ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= topologyFaceCount)) {
      throw new Error('A region contains duplicate or out-of-range face IDs.')
    }
  }
  const moved = [...new Set(args.movedFaceIds)].sort((a, b) => a - b)
  if (!moved.length || moved.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= topologyFaceCount)) {
    throw new Error('Choose one or more in-range face IDs to move.')
  }
  const sourceFaces = new Set(source.mapping.element_ids)
  const destinationFaces = new Set(destination.mapping.element_ids)
  if (!moved.every((id) => sourceFaces.has(id)) || moved.some((id) => destinationFaces.has(id))) {
    throw new Error('Every moved face must belong to the selected source part and not the destination.')
  }
  const sourceAfter = source.mapping.element_ids.filter((id) => !moved.includes(id))
  if (!sourceAfter.length) throw new Error('A seam edit cannot remove every face from the source part.')
  return {
    topologyRevision,
    sourceRegionId: source.region_id,
    destinationRegionId: destination.region_id,
    moved,
    sourceBefore: [...source.mapping.element_ids],
    sourceAfter,
    destinationBefore: [...destination.mapping.element_ids],
    destinationAfter: [...new Set([...destination.mapping.element_ids, ...moved])].sort((a, b) => a - b),
  }
}
