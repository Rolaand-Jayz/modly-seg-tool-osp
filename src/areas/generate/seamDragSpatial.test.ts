import assert from 'node:assert/strict'
import test from 'node:test'
import { buildFaceSpatialIndex, queryFacesNearSegment } from './seamDragSpatial'

test('segment query returns only nearby indexed faces, including the segment endpoints', () => {
  const index = buildFaceSpatialIndex([
    [10, [0, 0, 0]],
    [11, [0.5, 0.05, 0]],
    [12, [1, 0.2, 0]],
    [13, [0.5, 0.8, 0]],
  ], 0.25)
  assert.deepEqual(queryFacesNearSegment(index, [0, 0, 0], [1, 0, 0], 0.1).sort(), [10, 11])
})

test('segment query obeys the sample cap for a very long pointer jump', () => {
  const index = buildFaceSpatialIndex([[7, [0, 0, 0]], [8, [100, 0, 0]], [9, [50, 0, 0]]], 0.01)
  const result = queryFacesNearSegment(index, [0, 0, 0], [100, 0, 0], 0.01, 1)
  assert.deepEqual(result.sort(), [7, 8])
  assert.equal(result.includes(9), false, 'interior samples are intentionally skipped when the movement exceeds the cap')
})
