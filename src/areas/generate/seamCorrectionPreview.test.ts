import assert from 'node:assert/strict'
import test from 'node:test'
import { buildSeamCorrectionPreview, type SeamRegionMapping } from './seamCorrectionPreview'

const revision = 'sha256:synthetic-mesh-topology-r1'
const source: SeamRegionMapping = { region_id: 'car-door', mapping: {
  topology_revision: revision, state: 'valid', element_type: 'face', element_ids: [3, 4, 5, 6],
} }
const destination: SeamRegionMapping = { region_id: 'car-fender', mapping: {
  topology_revision: revision, state: 'valid', element_type: 'face', element_ids: [7, 8],
} }

test('previews exact seam membership without changing the input mappings', () => {
  const before = structuredClone({ source, destination })
  const preview = buildSeamCorrectionPreview({ topologyRevision: revision, topologyFaceCount: 10,
    viewerFaceCount: 10, source, destination, movedFaceIds: [5, 4, 5] })
  assert.deepEqual(preview, {
    topologyRevision: revision, sourceRegionId: 'car-door', destinationRegionId: 'car-fender',
    moved: [4, 5], sourceBefore: [3, 4, 5, 6], sourceAfter: [3, 6],
    destinationBefore: [7, 8], destinationAfter: [4, 5, 7, 8],
  })
  assert.deepEqual({ source, destination }, before)
})

test('rejects stale topology mappings and viewer geometry count mismatches', () => {
  assert.throws(() => buildSeamCorrectionPreview({ topologyRevision: revision, topologyFaceCount: 10,
    source: { ...source, mapping: { ...source.mapping, topology_revision: 'old-revision' } },
    destination, movedFaceIds: [4] }), /different topology revision/)
  assert.throws(() => buildSeamCorrectionPreview({ topologyRevision: revision, topologyFaceCount: 10,
    viewerFaceCount: 9, source, destination, movedFaceIds: [4] }), /face count does not match/)
})

test('rejects out-of-range or non-source faces and emptying the source region', () => {
  const preview = (movedFaceIds: number[]) => buildSeamCorrectionPreview({ topologyRevision: revision,
    topologyFaceCount: 10, source, destination, movedFaceIds })
  assert.throws(() => preview([10]), /in-range/)
  assert.throws(() => preview([7]), /belong to the selected source/)
  assert.throws(() => preview([3, 4, 5, 6]), /cannot remove every face/)
})
