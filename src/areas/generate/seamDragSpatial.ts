export type Point3 = readonly [number, number, number]

export interface FaceSpatialIndex {
  cellSize: number
  buckets: Map<string, Array<{ faceId: number; point: Point3 }>>
}

const cellKey = (x: number, y: number, z: number): string => `${x},${y},${z}`
const cellOf = (point: Point3, size: number): [number, number, number] => [
  Math.floor(point[0] / size), Math.floor(point[1] / size), Math.floor(point[2] / size),
]

/** Build once per seam-drag session; each face center is stored in one spatial bin. */
export function buildFaceSpatialIndex(faces: Iterable<readonly [number, Point3]>, cellSize: number): FaceSpatialIndex {
  if (!Number.isFinite(cellSize) || cellSize <= 0) throw new RangeError('cellSize must be positive and finite')
  const buckets = new Map<string, Array<{ faceId: number; point: Point3 }>>()
  for (const [faceId, point] of faces) {
    const cell = cellOf(point, cellSize)
    const key = cellKey(...cell)
    const bucket = buckets.get(key)
    const entry = { faceId, point }
    if (bucket) bucket.push(entry)
    else buckets.set(key, [entry])
  }
  return { cellSize, buckets }
}

function distanceSquaredToSegment(point: Point3, start: Point3, end: Point3): number {
  const dx = end[0] - start[0]; const dy = end[1] - start[1]; const dz = end[2] - start[2]
  const lengthSquared = dx * dx + dy * dy + dz * dz
  const t = lengthSquared === 0 ? 0 : Math.max(0, Math.min(1,
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy + (point[2] - start[2]) * dz) / lengthSquared))
  const x = start[0] + t * dx - point[0]
  const y = start[1] + t * dy - point[1]
  const z = start[2] + t * dz - point[2]
  return x * x + y * y + z * z
}

/**
 * Query only bins near one newly traversed segment. Sampling and bin visits are
 * bounded so pathological pointer jumps cannot monopolize the render thread.
 */
export function queryFacesNearSegment(
  index: FaceSpatialIndex,
  start: Point3,
  end: Point3,
  radius: number,
  maxSamples = 128,
): number[] {
  if (!Number.isFinite(radius) || radius < 0) throw new RangeError('radius must be finite and nonnegative')
  if (!Number.isInteger(maxSamples) || maxSamples < 1) throw new RangeError('maxSamples must be a positive integer')
  if (radius > index.cellSize) throw new RangeError('radius must not exceed the spatial cell size')
  const dx = end[0] - start[0]; const dy = end[1] - start[1]; const dz = end[2] - start[2]
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
  const samples = Math.min(maxSamples, Math.max(1, Math.ceil(distance / (index.cellSize * 0.5))))
  const neighboringCells = Math.max(1, Math.ceil(radius / index.cellSize))
  const visitedBuckets = new Set<string>()
  const candidateFaces = new Map<number, Point3>()
  for (let sampleIndex = 0; sampleIndex <= samples; sampleIndex++) {
    const t = sampleIndex / samples
    const sample: Point3 = [start[0] + dx * t, start[1] + dy * t, start[2] + dz * t]
    const [cx, cy, cz] = cellOf(sample, index.cellSize)
    for (let x = -neighboringCells; x <= neighboringCells; x++) {
      for (let y = -neighboringCells; y <= neighboringCells; y++) {
        for (let z = -neighboringCells; z <= neighboringCells; z++) {
          const key = cellKey(cx + x, cy + y, cz + z)
          if (visitedBuckets.has(key)) continue
          visitedBuckets.add(key)
          for (const entry of index.buckets.get(key) ?? []) candidateFaces.set(entry.faceId, entry.point)
        }
      }
    }
  }
  const radiusSquared = radius * radius
  const result: number[] = []
  for (const [faceId, point] of candidateFaces) {
    if (distanceSquaredToSegment(point, start, end) <= radiusSquared) result.push(faceId)
  }
  return result
}
