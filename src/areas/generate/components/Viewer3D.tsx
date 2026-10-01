import { Component, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode, ErrorInfo, MutableRefObject } from 'react'
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import { Environment, GizmoHelper, Lightformer, OrbitControls, useGizmoContext, useGLTF } from '@react-three/drei'
import { EffectComposer, Outline, Select, Selection } from '@react-three/postprocessing'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh'

// Patch THREE pour utiliser BVH sur tous les meshes — réduit le raycast O(N) → O(log N)
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree as any
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree as any
THREE.Mesh.prototype.raycast = acceleratedRaycast
import SplatViewer, { type SplatViewerHandle } from './SplatViewer'
import { useGeneration } from '@shared/hooks/useGeneration'
import { useAppStore } from '@shared/stores/appStore'
import { ViewerToolbar, type ViewMode } from './ViewerToolbar'
import type { LightSettings } from '@shared/stores/appStore'
import { DEFAULT_LIGHT_SETTINGS } from '@shared/stores/appStore'
import StructuredAssetReviewPanel from './StructuredAssetReviewPanel'
import { useWorkflowRunStore, type WorkflowStageTelemetry } from '@areas/workflows/workflowRunStore'
import { buildFaceSpatialIndex, queryFacesNearSegment, type FaceSpatialIndex } from '../seamDragSpatial'

export type GizmoMode = 'translate' | 'rotate' | 'scale'

function formatTelemetryBytes(value?: number): string {
  if (value === undefined) return 'Unknown'
  if (value === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  return `${(value / 1024 ** index).toFixed(index > 0 ? 1 : 0)} ${units[index]}`
}

function StageTelemetryPanel({ stages }: { stages: WorkflowStageTelemetry[] }): JSX.Element | null {
  const [open, setOpen] = useState(false)
  if (stages.length === 0) return null
  return (
    <section className="absolute left-3 top-3 z-20 w-[min(360px,calc(100%-24px))] overflow-hidden rounded-xl border border-zinc-700/80 bg-zinc-950/95 text-zinc-200 shadow-xl backdrop-blur">
      <button className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-semibold" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span>Workflow performance · {stages.length} stage{stages.length === 1 ? '' : 's'}</span>
        <span className="text-zinc-500">{open ? 'Hide' : 'Inspect'}</span>
      </button>
      {open && (
        <div className="max-h-[55vh] space-y-2 overflow-auto border-t border-zinc-800 px-3 py-2">
          {stages.map((stage, index) => (
            <article key={`${stage.nodeId}:${stage.stageId}:${index}`} className="rounded-lg border border-zinc-800 bg-zinc-900/70 p-2">
              <h3 className="truncate text-xs font-medium" title={stage.stageId}>{stage.stageId}</h3>
              <p className="mt-1 text-[11px] text-zinc-400">{stage.nodeLabel} · backend {stage.backend ?? 'Unknown'}</p>
              <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                <dt className="text-zinc-500">Stage time</dt><dd>{stage.latencyMs === undefined ? 'Unknown' : `${stage.latencyMs.toFixed(1)} ms`}</dd>
                <dt className="text-zinc-500">Peak VRAM used</dt><dd>{formatTelemetryBytes(stage.peakVramAllocatedBytes)}</dd>
                <dt className="text-zinc-500">Peak VRAM reserved</dt><dd>{formatTelemetryBytes(stage.peakVramReservedBytes)}</dd>
                <dt className="text-zinc-500">Device</dt><dd className="truncate" title={stage.device}>{stage.device ?? 'Unknown'}</dd>
              </dl>
              {stage.evidencePath && <p className="mt-1 truncate text-[10px] text-zinc-600" title={stage.evidencePath}>Evidence: {stage.evidencePath}</p>}
            </article>
          ))}
          <p className="text-[10px] text-zinc-500">Unknown means the stage did not provide that measurement; it is not treated as zero.</p>
        </div>
      )}
    </section>
  )
}

const SELECTION_OUTLINE_VISIBLE_COLOR = 0x8b5cf6
const SELECTION_OUTLINE_HIDDEN_COLOR = 0x5b21b6
const SELECTION_OUTLINE_EDGE_STRENGTH = 2.5
const SELECTION_OUTLINE_BLUR = false
const SELECTION_OUTLINE_MULTISAMPLING = 0
const SELECTION_OUTLINE_RESOLUTION_SCALE = 0.5

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

function createMatcapTexture(): THREE.CanvasTexture {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')!
  const grad = ctx.createRadialGradient(size * 0.35, size * 0.3, 0, size / 2, size / 2, size / 2)
  grad.addColorStop(0, '#ffffff')
  grad.addColorStop(0.45, '#aaaaaa')
  grad.addColorStop(1, '#222222')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)
  return new THREE.CanvasTexture(canvas)
}

function createCheckerTexture(): THREE.CanvasTexture {
  const size = 256
  const tileCount = 8
  const tileSize = size / tileCount
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')!
  for (let row = 0; row < tileCount; row++) {
    for (let col = 0; col < tileCount; col++) {
      ctx.fillStyle = (row + col) % 2 === 0 ? '#e0e0e0' : '#888888'
      ctx.fillRect(col * tileSize, row * tileSize, tileSize, tileSize)
    }
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  return tex
}

// ---------------------------------------------------------------------------
// CanvasCapture — exposes gl.domElement ref outside Canvas
// ---------------------------------------------------------------------------

function CanvasCapture({
  domRef,
}: {
  domRef: React.MutableRefObject<HTMLCanvasElement | null>
}): null {
  const { gl } = useThree()
  useEffect(() => {
    domRef.current = gl.domElement
    // eslint-disable-next-line react-hooks/exhaustive-deps -- domRef is a stable ref
  }, [gl])
  return null
}

// ---------------------------------------------------------------------------
// ModelErrorBoundary — catches useGLTF load failures (e.g. 404)
// ---------------------------------------------------------------------------

interface ErrorBoundaryProps {
  children: ReactNode
  fallback: ReactNode
  resetKey?: string | null
}

interface ErrorBoundaryState {
  hasError: boolean
}

class ModelErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn('[Viewer3D] Failed to load model:', error.message, info.componentStack)
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false })
    }
  }

  render(): ReactNode {
    return this.state.hasError ? this.props.fallback : this.props.children
  }
}

function ModelLoadError(): JSX.Element {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-600 pointer-events-none">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="10" />
        <line x1="15" y1="9" x2="9" y2="15" />
        <line x1="9" y1="9" x2="15" y2="15" />
      </svg>
      <p className="mt-3 text-sm">Model file not found</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// MeshModel
// ---------------------------------------------------------------------------

interface MeshModelProps {
  url: string
  jobId: string
  viewMode: ViewMode
  selected: boolean
  onStats: (stats: { vertices: number; triangles: number }) => void
  onSelect: () => void
  onObject: (obj: THREE.Object3D | null) => void
  facePickEnabled: boolean
  onPickFace: (faceId: number) => void
  pickedFaceIds: number[]
  highlightedRegionFaceIds: number[]
  seamDragEnabled?: boolean
  seamSourceFaceIds?: number[]
  seamDestinationFaceIds?: number[]
  onSeamDragFaces?: (faceIds: number[]) => void
  onSeamBoundaryAvailability?: (available: boolean) => void
  faceIndexOffsets?: WeakMap<THREE.Mesh, number>
}

function MeshModel({ url, jobId, viewMode, selected, onStats, onSelect, onObject, facePickEnabled, onPickFace, pickedFaceIds, highlightedRegionFaceIds, seamDragEnabled, seamSourceFaceIds, seamDestinationFaceIds, onSeamDragFaces, onSeamBoundaryAvailability }: MeshModelProps): JSX.Element {
  const extension = url.split('?')[0]?.split('.').pop()?.toLowerCase()
  const common = { url, jobId, viewMode, selected, onStats, onSelect, onObject, facePickEnabled, onPickFace, pickedFaceIds, highlightedRegionFaceIds, seamDragEnabled, seamSourceFaceIds, seamDestinationFaceIds, onSeamDragFaces, onSeamBoundaryAvailability }
  return extension === 'obj' ? <ObjMeshModel {...common} /> : <GltfMeshModel {...common} />
}

function GltfMeshModel(props: MeshModelProps): JSX.Element {
  const gltf = useGLTF(props.url)
  const faceIndexOffsets = useMemo(() => {
    gltf.scene.userData.modlyFaceIdMappingAvailable = false
    delete gltf.scene.userData.modlyFaceIdCount
    const associations = gltf.parser?.associations as Map<THREE.Object3D, { meshes?: number; primitives?: number }> | undefined
    if (!associations) return undefined
    const rows: Array<{ meshIndex: number; primitiveIndex: number; mesh: THREE.Mesh; faces: number }> = []
    gltf.scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      const association = associations.get(child)
      if (!Number.isInteger(association?.meshes) || !Number.isInteger(association?.primitives)) return
      const count = child.geometry.index?.count ?? child.geometry.attributes.position?.count ?? 0
      rows.push({ meshIndex: association!.meshes!, primitiveIndex: association!.primitives!, mesh: child, faces: Math.floor(count / 3) })
    })
    rows.sort((a, b) => a.meshIndex - b.meshIndex || a.primitiveIndex - b.primitiveIndex)
    const offsets = new WeakMap<THREE.Mesh, number>()
    const primitiveKeys = new Set<string>()
    let offset = 0
    for (const row of rows) {
      const primitiveKey = `${row.meshIndex}:${row.primitiveIndex}`
      if (primitiveKeys.has(primitiveKey)) return undefined
      primitiveKeys.add(primitiveKey)
      if (offsets.has(row.mesh)) return undefined
      offsets.set(row.mesh, offset)
      offset += row.faces
    }
    gltf.scene.userData.modlyFaceIdMappingAvailable = rows.length > 0
    gltf.scene.userData.modlyFaceIdCount = offset
    return rows.length > 0 ? offsets : undefined
  }, [gltf.scene, gltf.parser])
  return <SceneMeshModel {...props} faceIndexOffsets={faceIndexOffsets} scene={gltf.scene} loaderType="gltf" />
}

function ObjMeshModel(props: MeshModelProps): JSX.Element {
  const scene = useLoader(OBJLoader, props.url)
  return <SceneMeshModel {...props} scene={scene} loaderType="obj" />
}

function SceneMeshModel({
  url,
  viewMode,
  selected,
  onStats,
  onSelect,
  onObject,
  facePickEnabled,
  onPickFace,
  pickedFaceIds,
  highlightedRegionFaceIds,
  seamDragEnabled,
  seamSourceFaceIds = [],
  seamDestinationFaceIds = [],
  onSeamDragFaces,
  onSeamBoundaryAvailability,
  faceIndexOffsets,
  scene,
  loaderType,
}: MeshModelProps & {
  scene: THREE.Group | THREE.Scene
  loaderType: 'gltf' | 'obj'
}): JSX.Element {
  const captured = useRef(false)
  const edgeHelpers = useRef<THREE.LineSegments[]>([])
  const pickedFaceHelpers = useRef<THREE.LineSegments[]>([])
  const boundaryDrag = useRef<{ active: boolean; path: THREE.Vector3[]; lastPoint: THREE.Vector3 | null; source: Set<number>; moved: Set<number>; lastPublishedAt: number }>({ active: false, path: [], lastPoint: null, source: new Set(), moved: new Set(), lastPublishedAt: 0 })
  const { controls } = useThree() as { controls?: { enabled: boolean } }

  const seamHandleData = useMemo(() => {
    if (!faceIndexOffsets || seamSourceFaceIds.length === 0 || seamDestinationFaceIds.length === 0) return []
    const source = new Set(seamSourceFaceIds)
    const destination = new Set(seamDestinationFaceIds)
    const handles: THREE.Vector3[] = []
    scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      const offset = faceIndexOffsets.get(child)
      const position = child.geometry.getAttribute('position')
      if (offset === undefined || !position) return
      const index = child.geometry.index
      const faceCount = Math.floor((index?.count ?? position.count) / 3)
      const edges = new Map<string, { a: number; b: number; sourceFace?: number; destination: boolean }>()
      for (let face = 0; face < faceCount; face++) {
        const globalFace = offset + face
        const isSource = source.has(globalFace)
        const isDestination = destination.has(globalFace)
        if (!isSource && !isDestination) continue
        const vertices = [0, 1, 2].map((corner) => index ? index.getX(face * 3 + corner) : face * 3 + corner)
        for (const [left, right] of [[0, 1], [1, 2], [2, 0]]) {
          const a = Math.min(vertices[left], vertices[right]); const b = Math.max(vertices[left], vertices[right])
          const key = `${a}:${b}`
          const edge = edges.get(key) ?? { a, b, destination: false }
          if (isSource) edge.sourceFace = globalFace
          if (isDestination) edge.destination = true
          edges.set(key, edge)
        }
      }
      for (const edge of edges.values()) {
        if (edge.sourceFace === undefined || !edge.destination) continue
        const midpoint = new THREE.Vector3().fromBufferAttribute(position, edge.a)
          .add(new THREE.Vector3().fromBufferAttribute(position, edge.b)).multiplyScalar(0.5)
        child.localToWorld(midpoint)
        handles.push(midpoint)
      }
    })
    return handles
  }, [faceIndexOffsets, scene, seamDestinationFaceIds, seamSourceFaceIds])
  const sourceFaceSpatialIndex = useMemo<FaceSpatialIndex | null>(() => {
    const source = new Set(seamSourceFaceIds)
    if (!seamDragEnabled || !faceIndexOffsets || source.size === 0) return null
    const sphere = new THREE.Sphere()
    new THREE.Box3().setFromObject(scene).getBoundingSphere(sphere)
    const brushRadius = Math.max(sphere.radius * 0.035, 1e-5)
    const faces: Array<readonly [number, readonly [number, number, number]]> = []
    scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      const offset = faceIndexOffsets.get(child)
      const positions = child.geometry.getAttribute('position')
      if (offset === undefined || !positions) return
      const index = child.geometry.index
      const count = Math.floor((index?.count ?? positions.count) / 3)
      for (let face = 0; face < count; face++) {
        const globalFace = offset + face
        if (!source.has(globalFace)) continue
        const vertices = [0, 1, 2].map((corner) => index ? index.getX(face * 3 + corner) : face * 3 + corner)
        const center = new THREE.Vector3().fromBufferAttribute(positions, vertices[0])
          .add(new THREE.Vector3().fromBufferAttribute(positions, vertices[1]))
          .add(new THREE.Vector3().fromBufferAttribute(positions, vertices[2])).multiplyScalar(1 / 3)
        child.localToWorld(center)
        faces.push([globalFace, [center.x, center.y, center.z]])
      }
    })
    return buildFaceSpatialIndex(faces, brushRadius)
  }, [faceIndexOffsets, scene, seamDragEnabled, seamSourceFaceIds])
  useEffect(() => { onSeamBoundaryAvailability?.(seamHandleData.length > 0) }, [onSeamBoundaryAvailability, seamHandleData])

  const updateSeamDrag = useCallback((point: THREE.Vector3, hitFace?: number) => {
    const drag = boundaryDrag.current
    if (!drag.active) return
    const previousPoint = drag.lastPoint ?? point
    drag.lastPoint = point.clone()
    drag.path.push(point.clone())
    if (drag.path.length > 256) drag.path.shift()
    if (hitFace != null && drag.source.has(hitFace)) drag.moved.add(hitFace)
    // Query only bins near this new segment. Prior pointer samples are never
    // rescanned; the one-time index and sampling cap keep pointer work bounded.
    if (sourceFaceSpatialIndex) {
      const radius = sourceFaceSpatialIndex.cellSize
      const start: readonly [number, number, number] = [previousPoint.x, previousPoint.y, previousPoint.z]
      const end: readonly [number, number, number] = [point.x, point.y, point.z]
      for (const candidate of queryFacesNearSegment(sourceFaceSpatialIndex, start, end, radius)) drag.moved.add(candidate)
    }
    const now = performance.now()
    if (now - drag.lastPublishedAt >= 100) {
      drag.lastPublishedAt = now
      onSeamDragFaces?.([...drag.moved].sort((a, b) => a - b))
    }
  }, [onSeamDragFaces, sourceFaceSpatialIndex])

  useEffect(() => {
    if (seamDragEnabled) return
    boundaryDrag.current.active = false
    if (controls) controls.enabled = true
  }, [controls, seamDragEnabled])

  useEffect(() => {
    const finish = () => {
      const drag = boundaryDrag.current
      if (!drag.active) return
      drag.active = false
      if (controls) controls.enabled = true
      if (drag.moved.size > 0) onSeamDragFaces?.([...drag.moved].sort((a, b) => a - b))
    }
    window.addEventListener('pointerup', finish)
    window.addEventListener('blur', finish)
    return () => { window.removeEventListener('pointerup', finish); window.removeEventListener('blur', finish) }
  }, [controls, onSeamDragFaces])

  // Expose the scene object so Viewer3D can attach the transform gizmo to it.
  useEffect(() => {
    onObject(scene)
    return () => onObject(null)
  }, [scene, onObject])

  // Free GPU resources and loader cache when this model is replaced or unmounted
  useEffect(() => {
    return () => {
      if (loaderType === 'obj') {
        useLoader.clear(OBJLoader, url)
      } else {
        useGLTF.clear(url)
      }
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose()
          const materials = Array.isArray(child.material) ? child.material : [child.material]
          materials.forEach((m: THREE.Material) => m.dispose())
        }
      })
    }
  }, [loaderType, scene, url])

  // Compute BVH on all geometries for fast raycasting (O(log N) vs O(N)).
  // Also force DoubleSide on every material so faces with inverted normals
  // (a known artifact of the flexible-dual-grid mesh decoder) are still visible.
  useEffect(() => {
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        (child.geometry as any).computeBoundsTree()
        const mats = Array.isArray(child.material) ? child.material : [child.material]
        mats.forEach((m: THREE.Material) => { m.side = THREE.DoubleSide })
      }
    })
    return () => {
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          (child.geometry as any).disposeBoundsTree?.()
        }
      })
    }
  }, [scene])

  useEffect(() => {
    pickedFaceHelpers.current.forEach((lines) => {
      lines.parent?.remove(lines)
      lines.geometry.dispose()
      ;(lines.material as THREE.Material).dispose()
    })
    pickedFaceHelpers.current = []
    if (!faceIndexOffsets || (pickedFaceIds.length === 0 && highlightedRegionFaceIds.length === 0)) return
    scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      const offset = faceIndexOffsets.get(child)
      if (offset === undefined) return
      const position = child.geometry.getAttribute('position')
      if (!position) return
      const index = child.geometry.index
      const faceCount = Math.floor((index?.count ?? position.count) / 3)
      const picked = new Set<number>(pickedFaceIds)
      const region = new Set<number>(highlightedRegionFaceIds.filter((id) => !picked.has(id)))
      const drawFaces = (globalIds: Set<number>, color: number) => {
        const vertices: number[] = []
        for (const globalId of globalIds) {
          const localFace = globalId - offset
          if (localFace < 0 || localFace >= faceCount) continue
          const tri = [0, 1, 2].map((corner) => index ? index.getX(localFace * 3 + corner) : localFace * 3 + corner)
          const points = tri.map((vertex) => new THREE.Vector3().fromBufferAttribute(position, vertex))
          for (const edge of [[0, 1], [1, 2], [2, 0]]) {
            vertices.push(...points[edge[0]].toArray(), ...points[edge[1]].toArray())
          }
        }
        if (vertices.length === 0) return
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
        const helper = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }))
        helper.renderOrder = 1000
        child.add(helper)
        pickedFaceHelpers.current.push(helper)
      }
      drawFaces(region, 0x47d7ff)
      drawFaces(picked, 0xffc94a)
    })
    return () => {
      pickedFaceHelpers.current.forEach((lines) => {
        lines.parent?.remove(lines)
        lines.geometry.dispose()
        ;(lines.material as THREE.Material).dispose()
      })
      pickedFaceHelpers.current = []
    }
  }, [faceIndexOffsets, highlightedRegionFaceIds, pickedFaceIds, scene])

  // Centre the mesh on the grid. Runs only on first load / model change — never
  // on plain re-renders, so a live gizmo transform is not silently overwritten.
  useEffect(() => {
    // Clear any cached transform before measuring (useGLTF may reuse a scene
    // that still carries an earlier gizmo pose).
    scene.position.set(0, 0, 0)
    scene.rotation.set(0, 0, 0)
    scene.scale.set(1, 1, 1)
    const box = new THREE.Box3().setFromObject(scene)
    const center = new THREE.Vector3()
    box.getCenter(center)
    scene.position.set(-center.x, -box.min.y, -center.z)

    // Compute stats
    let vertices = 0
    let triangles = 0
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        vertices += child.geometry.attributes.position?.count ?? 0
        triangles += child.geometry.index
          ? child.geometry.index.count / 3
          : (child.geometry.attributes.position?.count ?? 0) / 3
      }
    })
    const roundedTriangles = Math.round(triangles)
    onStats({ vertices: Math.round(vertices), triangles: roundedTriangles })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- recompute on scene change only; onStats is a stable callback
  }, [scene])

  // Thumbnail capture (kept for future use)
  useEffect(() => {
    captured.current = false
  }, [url])

  // Material swapping based on viewMode
  useEffect(() => {
    // Remove any edge helpers from previous wireframe pass
    edgeHelpers.current.forEach((lines) => lines.parent?.remove(lines))
    edgeHelpers.current = []

    scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return

      // Save original material on first visit
      if (!child.userData.originalMaterial) {
        child.userData.originalMaterial = child.material
      }

      let next: THREE.Material
      switch (viewMode) {
        case 'wireframe': {
          next = new THREE.MeshBasicMaterial({ color: 0x4ade80, wireframe: true })
          break
        }
        case 'normals':
          // Ensure vertex normals exist — AI-generated meshes often skip this
          child.geometry.computeVertexNormals()
          next = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide })
          break
        case 'matcap':
          next = new THREE.MeshMatcapMaterial({ matcap: createMatcapTexture() })
          break
        case 'uv':
          next = new THREE.MeshBasicMaterial({ map: createCheckerTexture() })
          break
        default:
          next = child.userData.originalMaterial as THREE.Material
      }

      child.material = next
    })
  }, [scene, viewMode])

  return (
    <Select enabled={selected}>
      <primitive
        object={scene}
        onClick={(e: { stopPropagation: () => void }) => { e.stopPropagation(); if (!facePickEnabled) onSelect() }}
        onPointerDown={(event: ThreeEvent<PointerEvent>) => {
          if (!facePickEnabled || seamDragEnabled) return
          const mesh = event.object as THREE.Mesh
          const offset = faceIndexOffsets?.get(mesh)
      if (event.faceIndex != null && offset !== undefined) {
            event.stopPropagation()
            onPickFace(offset + event.faceIndex)
          }
        }}
        onPointerMove={(event: ThreeEvent<PointerEvent>) => {
          if (seamDragEnabled && boundaryDrag.current.active && event.faceIndex != null && faceIndexOffsets) {
            const mesh = event.object as THREE.Mesh
            const offset = faceIndexOffsets.get(mesh)
            if (offset !== undefined) { event.stopPropagation(); updateSeamDrag(event.point, offset + event.faceIndex) }
            return
          }
          if (!facePickEnabled || (event.nativeEvent.buttons & 1) === 0) return
          const mesh = event.object as THREE.Mesh
          const offset = faceIndexOffsets?.get(mesh)
      if (event.faceIndex != null && offset !== undefined) {
            event.stopPropagation()
            onPickFace(offset + event.faceIndex)
          }
        }}
        onPointerUp={() => {
          const drag = boundaryDrag.current
          if (!drag.active) return
          drag.active = false
          if (controls) controls.enabled = true
          if (drag.moved.size > 0) onSeamDragFaces?.([...drag.moved].sort((a, b) => a - b))
        }}
      />
      {seamDragEnabled && seamHandleData.map((handlePosition, index) => (
        <mesh key={`seam-handle-${index}`} position={handlePosition} onPointerDown={(event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation()
          boundaryDrag.current = { active: true, path: [event.point.clone()], lastPoint: event.point.clone(), source: new Set(seamSourceFaceIds), moved: new Set(), lastPublishedAt: 0 }
          if (controls) controls.enabled = false
        }} onPointerUp={(event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation()
          const drag = boundaryDrag.current
          if (!drag.active) return
          drag.active = false
          if (controls) controls.enabled = true
          if (drag.moved.size > 0) onSeamDragFaces?.([...drag.moved].sort((a, b) => a - b))
        }}>
          <sphereGeometry args={[0.014, 8, 6]} />
          <meshBasicMaterial color="#ff4fa3" depthTest={false} />
        </mesh>
      ))}
    </Select>
  )

}

// ---------------------------------------------------------------------------
// Orientation gizmo — coloured bubbles only (X/Y/Z)
// ---------------------------------------------------------------------------

function makeAxisLabelTexture(letter: string, bg: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 64
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.arc(32, 32, 16, 0, 2 * Math.PI)
  ctx.closePath()
  ctx.fillStyle = bg
  ctx.fill()
  ctx.font = '18px Arial, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillStyle = '#ffffff'
  ctx.fillText(letter, 32, 41)
  return new THREE.CanvasTexture(canvas)
}

const GIZMO_AXES: {
  letter: string
  color: string
  pos: [number, number, number]
  lineRotation: [number, number, number]
}[] = [
  { letter: 'X', color: '#f87171', pos: [1, 0, 0], lineRotation: [0, 0, 0] },
  { letter: 'Y', color: '#4ade80', pos: [0, 1, 0], lineRotation: [0, 0, Math.PI / 2] },
  { letter: 'Z', color: '#60a5fa', pos: [0, 0, 1], lineRotation: [0, -Math.PI / 2, 0] },
]

function AxisLine({ color, rotation }: { color: string; rotation: [number, number, number] }) {
  return (
    <group rotation={rotation}>
      <mesh position={[0.4, 0, 0]}>
        <boxGeometry args={[0.8, 0.05, 0.05]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
    </group>
  )
}

function AxisBubble({ letter, color, pos }: { letter: string; color: string; pos: [number, number, number] }) {
  const { tweenCamera } = useGizmoContext()
  const texture = useMemo(() => makeAxisLabelTexture(letter, color), [letter, color])
  const [hovered, setHovered] = useState(false)

  return (
    <sprite
      position={pos}
      scale={hovered ? 1.2 : 1}
      onPointerDown={(e) => { tweenCamera(e.object.position); e.stopPropagation() }}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true) }}
      onPointerOut={() => setHovered(false)}
    >
      <spriteMaterial map={texture} alphaTest={0.3} toneMapped={false} />
    </sprite>
  )
}

function GizmoBubbles() {
  return (
    <group scale={40}>
      {GIZMO_AXES.map((axis) => (
        <AxisLine key={`line-${axis.letter}`} color={axis.color} rotation={axis.lineRotation} />
      ))}
      {GIZMO_AXES.map((axis) => (
        <AxisBubble key={axis.letter} {...axis} />
      ))}
    </group>
  )
}

// ---------------------------------------------------------------------------
// Transform gizmos — custom move / rotate / scale handles (shared style)
// ---------------------------------------------------------------------------

type GizmoAxis = 'x' | 'y' | 'z'
type TranslateHandleId = GizmoAxis | 'xy' | 'yz' | 'xz'
type ScaleHandleId = GizmoAxis | 'xyz'

const AXIS_COLORS: Record<GizmoAxis, string> = {
  x: '#f87171',
  y: '#4ade80',
  z: '#60a5fa',
}

const AXIS_DIR: Record<GizmoAxis, [number, number, number]> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
}

// Orient a +Y cylinder/cone/box onto each axis.
const AXIS_ROTATION: Record<GizmoAxis, [number, number, number]> = {
  x: [0, 0, -Math.PI / 2],
  y: [0, 0, 0],
  z: [Math.PI / 2, 0, 0],
}

// Orient a default-XY torus so its ring spins around each axis.
const RING_ROTATION: Record<GizmoAxis, [number, number, number]> = {
  x: [0, Math.PI / 2, 0],
  y: [Math.PI / 2, 0, 0],
  z: [0, 0, 0],
}

// Two-axis plane handles, coloured by their locked (normal) axis.
const PLANE_HANDLES: {
  id: 'xy' | 'yz' | 'xz'
  normal: [number, number, number]
  color: string
  position: [number, number, number]
  rotation: [number, number, number]
}[] = [
  { id: 'xy', normal: [0, 0, 1], color: AXIS_COLORS.z, position: [0.26, 0.26, 0], rotation: [0, 0, 0] },
  { id: 'yz', normal: [1, 0, 0], color: AXIS_COLORS.x, position: [0, 0.26, 0.26], rotation: [0, -Math.PI / 2, 0] },
  { id: 'xz', normal: [0, 1, 0], color: AXIS_COLORS.y, position: [0.26, 0, 0.26], rotation: [Math.PI / 2, 0, 0] },
]

const GIZMO_SCREEN_SIZE = 0.12

function lightenColor(hex: string, amount = 0.5): string {
  return '#' + new THREE.Color(hex).lerp(new THREE.Color('#ffffff'), amount).getHexString()
}

function intersectPlane(ray: THREE.Ray, origin: THREE.Vector3, normal: THREE.Vector3): THREE.Vector3 | null {
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, origin)
  const hit = new THREE.Vector3()
  return ray.intersectPlane(plane, hit) ? hit : null
}

// Shared plumbing: follow the object, keep a constant on-screen size, and run
// the pointer-drag lifecycle (window listeners + OrbitControls locking).
function useGizmoBase(object: THREE.Object3D) {
  const camera = useThree((s) => s.camera)
  const gl = useThree((s) => s.gl)
  const raycaster = useThree((s) => s.raycaster)
  const controls = useThree((s) => s.controls) as { enabled: boolean } | null

  const groupRef = useRef<THREE.Group>(null)
  const ndc = useRef(new THREE.Vector2())
  const moveRef = useRef<((ev: PointerEvent) => void) | null>(null)
  const endRef = useRef<(() => void) | null>(null)

  useFrame(() => {
    const g = groupRef.current
    if (!g) return
    object.getWorldPosition(g.position)
    g.scale.setScalar(Math.max(camera.position.distanceTo(g.position) * GIZMO_SCREEN_SIZE, 0.001))
  })

  const pointerRay = useCallback((ev: PointerEvent): THREE.Ray => {
    const rect = gl.domElement.getBoundingClientRect()
    ndc.current.set(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -((ev.clientY - rect.top) / rect.height) * 2 + 1,
    )
    raycaster.setFromCamera(ndc.current, camera)
    return raycaster.ray
  }, [camera, gl, raycaster])

  const stop = useCallback(() => {
    if (!moveRef.current) return
    window.removeEventListener('pointermove', moveRef.current)
    window.removeEventListener('pointerup', stop)
    moveRef.current = null
    endRef.current?.()
    endRef.current = null
    if (controls) controls.enabled = true
    gl.domElement.style.cursor = ''
  }, [controls, gl])

  const start = useCallback((onMove: (ev: PointerEvent) => void, onEnd?: () => void) => {
    moveRef.current = onMove
    endRef.current = onEnd ?? null
    if (controls) controls.enabled = false
    gl.domElement.style.cursor = 'grabbing'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', stop)
  }, [controls, gl, stop])

  useEffect(() => stop, [stop])  // release the drag if unmounted mid-interaction

  return { camera, groupRef, pointerRay, start }
}

function hoverHandlers<T extends string>(
  id: T,
  setHovered: (value: T | null) => void,
  onDown: (e: ThreeEvent<PointerEvent>) => void,
) {
  return {
    onPointerOver: (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setHovered(id) },
    onPointerOut: () => setHovered(null),
    onPointerDown: onDown,
  }
}

function GizmoArrow({ color, active }: { color: string; active: boolean }): JSX.Element {
  const tint = active ? lightenColor(color) : color
  return (
    <group>
      {/* Invisible, fat hit target spanning the whole arm */}
      <mesh position={[0, 0.55, 0]}>
        <cylinderGeometry args={[0.09, 0.09, 1.1, 8]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {/* Shaft */}
      <mesh position={[0, 0.48, 0]} renderOrder={999}>
        <cylinderGeometry args={[0.014, 0.014, 0.66, 16]} />
        <meshBasicMaterial color={tint} toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>
      {/* Arrowhead */}
      <mesh position={[0, 0.9, 0]} renderOrder={999}>
        <coneGeometry args={[0.055, 0.2, 20]} />
        <meshBasicMaterial color={tint} toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>
    </group>
  )
}

function GizmoScaleArm({ color, active }: { color: string; active: boolean }): JSX.Element {
  const tint = active ? lightenColor(color) : color
  return (
    <group>
      {/* Invisible, fat hit target — starts above the centre cube so a
          centre click hits the uniform-scale handle, not an axis */}
      <mesh position={[0, 0.6, 0]}>
        <cylinderGeometry args={[0.09, 0.09, 0.8, 8]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {/* Shaft */}
      <mesh position={[0, 0.42, 0]} renderOrder={999}>
        <cylinderGeometry args={[0.014, 0.014, 0.7, 16]} />
        <meshBasicMaterial color={tint} toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>
      {/* Cube head */}
      <mesh position={[0, 0.84, 0]} renderOrder={999}>
        <boxGeometry args={[0.11, 0.11, 0.11]} />
        <meshBasicMaterial color={tint} toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>
    </group>
  )
}

function GizmoRing({ color, active }: { color: string; active: boolean }): JSX.Element {
  const tint = active ? lightenColor(color) : color
  return (
    <group>
      {/* Invisible, fat hit target */}
      <mesh>
        <torusGeometry args={[0.9, 0.06, 8, 48]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      <mesh renderOrder={999}>
        <torusGeometry args={[0.9, 0.012, 12, 64]} />
        <meshBasicMaterial color={tint} toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>
    </group>
  )
}

function GizmoPlane({ color, active }: { color: string; active: boolean }): JSX.Element {
  return (
    <mesh renderOrder={998}>
      <planeGeometry args={[0.26, 0.26]} />
      <meshBasicMaterial
        color={active ? lightenColor(color) : color}
        transparent
        opacity={active ? 0.6 : 0.28}
        side={THREE.DoubleSide}
        toneMapped={false}
        depthTest={false}
        depthWrite={false}
      />
    </mesh>
  )
}

function TranslateGizmo({ object, onDragStart, onDragEnd }: { object: THREE.Object3D; onDragStart?: () => void; onDragEnd?: () => void }): JSX.Element {
  const { camera, groupRef, pointerRay, start } = useGizmoBase(object)
  const [hovered, setHovered] = useState<TranslateHandleId | null>(null)
  const [activeId, setActiveId] = useState<TranslateHandleId | null>(null)
  const drag = useRef<{
    axisDir: THREE.Vector3 | null
    planeNormal: THREE.Vector3
    origin: THREE.Vector3
    startHit: THREE.Vector3
    startPos: THREE.Vector3
  } | null>(null)

  const beginDrag = useCallback((id: TranslateHandleId, e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    const origin = new THREE.Vector3()
    object.getWorldPosition(origin)
    const startPos = object.position.clone()

    let axisDir: THREE.Vector3 | null = null
    let planeNormal: THREE.Vector3
    if (id === 'x' || id === 'y' || id === 'z') {
      axisDir = new THREE.Vector3(...AXIS_DIR[id])
      // Drag plane: contains the axis and faces the camera as much as possible.
      const view = new THREE.Vector3().subVectors(camera.position, origin)
      planeNormal = view.sub(axisDir.clone().multiplyScalar(view.dot(axisDir)))
      if (planeNormal.lengthSq() < 1e-6) planeNormal.set(axisDir.y ? 1 : 0, axisDir.y ? 0 : 1, 0)
      planeNormal.normalize()
    } else {
      planeNormal = new THREE.Vector3(...PLANE_HANDLES.find((p) => p.id === id)!.normal)
    }

    const startHit = intersectPlane(e.ray, origin, planeNormal)
    if (!startHit) return
    drag.current = { axisDir, planeNormal, origin, startHit, startPos }
    setActiveId(id)
    onDragStart?.()
    start((ev) => {
      const d = drag.current
      if (!d) return
      const hit = intersectPlane(pointerRay(ev), d.origin, d.planeNormal)
      if (!hit) return
      const delta = new THREE.Vector3().subVectors(hit, d.startHit)
      if (d.axisDir) {
        object.position.copy(d.startPos).addScaledVector(d.axisDir, delta.dot(d.axisDir))
      } else {
        object.position.copy(d.startPos).add(delta)
      }
    }, () => { drag.current = null; setActiveId(null); onDragEnd?.() })
  }, [object, camera, pointerRay, start, onDragStart, onDragEnd])

  return (
    <group ref={groupRef} renderOrder={999}>
      {/* Central origin handle (decorative — never blocks picking) */}
      <mesh raycast={() => null} renderOrder={999}>
        <sphereGeometry args={[0.05, 20, 20]} />
        <meshBasicMaterial color="#e4e4e7" toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>

      {(['x', 'y', 'z'] as GizmoAxis[]).map((axis) => (
        <group key={axis} rotation={AXIS_ROTATION[axis]} {...hoverHandlers<TranslateHandleId>(axis, setHovered, (e) => beginDrag(axis, e))}>
          <GizmoArrow color={AXIS_COLORS[axis]} active={hovered === axis || activeId === axis} />
        </group>
      ))}

      {PLANE_HANDLES.map((plane) => (
        <group key={plane.id} position={plane.position} rotation={plane.rotation} {...hoverHandlers<TranslateHandleId>(plane.id, setHovered, (e) => beginDrag(plane.id, e))}>
          <GizmoPlane color={plane.color} active={hovered === plane.id || activeId === plane.id} />
        </group>
      ))}
    </group>
  )
}

function RotateGizmo({ object, onDragStart, onDragEnd }: { object: THREE.Object3D; onDragStart?: () => void; onDragEnd?: () => void }): JSX.Element {
  const { groupRef, pointerRay, start } = useGizmoBase(object)
  const [hovered, setHovered] = useState<GizmoAxis | null>(null)
  const [activeId, setActiveId] = useState<GizmoAxis | null>(null)
  const drag = useRef<{
    axisDir: THREE.Vector3
    origin: THREE.Vector3
    startVec: THREE.Vector3
    startQuat: THREE.Quaternion
  } | null>(null)

  const beginDrag = useCallback((axis: GizmoAxis, e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    const origin = new THREE.Vector3()
    object.getWorldPosition(origin)
    const axisDir = new THREE.Vector3(...AXIS_DIR[axis]).normalize()
    // Rotation happens in the plane perpendicular to the axis (the ring's plane).
    const startHit = intersectPlane(e.ray, origin, axisDir)
    if (!startHit) return
    const startVec = new THREE.Vector3().subVectors(startHit, origin)
    if (startVec.lengthSq() < 1e-9) return
    drag.current = { axisDir, origin, startVec, startQuat: object.quaternion.clone() }
    setActiveId(axis)
    onDragStart?.()
    start((ev) => {
      const d = drag.current
      if (!d) return
      const hit = intersectPlane(pointerRay(ev), d.origin, d.axisDir)
      if (!hit) return
      const cur = new THREE.Vector3().subVectors(hit, d.origin)
      // Signed angle between the start and current vectors, around the axis.
      const cross = new THREE.Vector3().crossVectors(d.startVec, cur)
      const angle = Math.atan2(cross.dot(d.axisDir), d.startVec.dot(cur))
      const q = new THREE.Quaternion().setFromAxisAngle(d.axisDir, angle)
      object.quaternion.copy(d.startQuat).premultiply(q)
    }, () => { drag.current = null; setActiveId(null); onDragEnd?.() })
  }, [object, pointerRay, start, onDragStart, onDragEnd])

  return (
    <group ref={groupRef} renderOrder={999}>
      {(['x', 'y', 'z'] as GizmoAxis[]).map((axis) => (
        <group key={axis} rotation={RING_ROTATION[axis]} {...hoverHandlers<GizmoAxis>(axis, setHovered, (e) => beginDrag(axis, e))}>
          <GizmoRing color={AXIS_COLORS[axis]} active={hovered === axis || activeId === axis} />
        </group>
      ))}
    </group>
  )
}

function ScaleGizmo({ object, onDragStart, onDragEnd }: { object: THREE.Object3D; onDragStart?: () => void; onDragEnd?: () => void }): JSX.Element {
  const { camera, groupRef, pointerRay, start } = useGizmoBase(object)
  const [hovered, setHovered] = useState<ScaleHandleId | null>(null)
  const [activeId, setActiveId] = useState<ScaleHandleId | null>(null)
  const drag = useRef<{
    axisDir: THREE.Vector3 | null
    planeNormal: THREE.Vector3
    origin: THREE.Vector3
    startProj: number
    startScale: THREE.Vector3
    armLength: number
  } | null>(null)

  const beginDrag = useCallback((id: ScaleHandleId, e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    const origin = new THREE.Vector3()
    object.getWorldPosition(origin)
    // World length of one local unit — maps drag distance to a sensible factor.
    const armLength = Math.max(groupRef.current?.scale.x ?? 1, 1e-4)

    let axisDir: THREE.Vector3 | null = null
    let planeNormal: THREE.Vector3
    if (id === 'xyz') {
      planeNormal = new THREE.Vector3().subVectors(camera.position, origin).normalize()
    } else {
      axisDir = new THREE.Vector3(...AXIS_DIR[id])
      const view = new THREE.Vector3().subVectors(camera.position, origin)
      planeNormal = view.sub(axisDir.clone().multiplyScalar(view.dot(axisDir)))
      if (planeNormal.lengthSq() < 1e-6) planeNormal.set(axisDir.y ? 1 : 0, axisDir.y ? 0 : 1, 0)
      planeNormal.normalize()
    }

    const startHit = intersectPlane(e.ray, origin, planeNormal)
    if (!startHit) return
    const startRel = new THREE.Vector3().subVectors(startHit, origin)
    const startProj = axisDir ? startRel.dot(axisDir) : startRel.length()
    drag.current = { axisDir, planeNormal, origin, startProj, startScale: object.scale.clone(), armLength }
    setActiveId(id)
    onDragStart?.()
    start((ev) => {
      const d = drag.current
      if (!d) return
      const hit = intersectPlane(pointerRay(ev), d.origin, d.planeNormal)
      if (!hit) return
      const rel = new THREE.Vector3().subVectors(hit, d.origin)
      const proj = d.axisDir ? rel.dot(d.axisDir) : rel.length()
      const factor = Math.max(0.01, 1 + (proj - d.startProj) / d.armLength)
      if (d.axisDir) {
        const s = d.startScale.clone()
        if (d.axisDir.x) s.x = Math.max(0.01, d.startScale.x * factor)
        if (d.axisDir.y) s.y = Math.max(0.01, d.startScale.y * factor)
        if (d.axisDir.z) s.z = Math.max(0.01, d.startScale.z * factor)
        object.scale.copy(s)
      } else {
        object.scale.copy(d.startScale).multiplyScalar(factor)
      }
    }, () => { drag.current = null; setActiveId(null); onDragEnd?.() })
  }, [object, camera, pointerRay, start, groupRef, onDragStart, onDragEnd])

  const uniformActive = hovered === 'xyz' || activeId === 'xyz'

  return (
    <group ref={groupRef} renderOrder={999}>
      {/* Central cube — uniform scale */}
      <mesh {...hoverHandlers<ScaleHandleId>('xyz', setHovered, (e) => beginDrag('xyz', e))} renderOrder={999}>
        <boxGeometry args={[0.12, 0.12, 0.12]} />
        <meshBasicMaterial color={uniformActive ? lightenColor('#e4e4e7') : '#e4e4e7'} toneMapped={false} transparent depthTest={false} depthWrite={false} />
      </mesh>

      {(['x', 'y', 'z'] as GizmoAxis[]).map((axis) => (
        <group key={axis} rotation={AXIS_ROTATION[axis]} {...hoverHandlers<ScaleHandleId>(axis, setHovered, (e) => beginDrag(axis, e))}>
          <GizmoScaleArm color={AXIS_COLORS[axis]} active={hovered === axis || activeId === axis} />
        </group>
      ))}
    </group>
  )
}

// ---------------------------------------------------------------------------
// EmptyState
// ---------------------------------------------------------------------------

function EmptyState(): JSX.Element {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-700 pointer-events-none">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="0.75">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
      <p className="mt-4 text-sm">3D model will appear here</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Viewer3D
// ---------------------------------------------------------------------------

type TransformSnapshot = { p: THREE.Vector3; q: THREE.Quaternion; s: THREE.Vector3 }

export default function Viewer3D({ lightSettings = DEFAULT_LIGHT_SETTINGS, gizmoMode = null, gizmoUndoRef }: { lightSettings?: LightSettings; gizmoMode?: GizmoMode | null; gizmoUndoRef?: MutableRefObject<(() => boolean) | null> }): JSX.Element {
  const { currentJob } = useGeneration()
  const apiUrl = useAppStore((s) => s.apiUrl)
  const workflowRunState = useWorkflowRunStore((s) => s.runState)
  const stageTelemetry = workflowRunState.stageTelemetryOutputUrl === currentJob?.outputUrl
    ? (workflowRunState.stageTelemetry ?? []) : []

  const setStoreMeshStats = useAppStore((s) => s.setMeshStats)
  const meshStats = useAppStore((s) => s.meshStats)
  const setCurrentJob = useAppStore((s) => s.setCurrentJob)

  const [viewMode, setViewMode] = useState<ViewMode>('solid')
  const [autoRotate, setAutoRotate] = useState(false)
  const selected = useAppStore((s) => s.meshSelected)
  const setSelected = useAppStore((s) => s.setMeshSelected)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const splatRef = useRef<SplatViewerHandle | null>(null)

  const [meshObject, setMeshObject] = useState<THREE.Object3D | null>(null)
  const [facePickEnabled, setFacePickEnabled] = useState(false)
  const [pickedFaceIds, setPickedFaceIds] = useState<number[]>([])
  const [highlightedRegionFaceIds, setHighlightedRegionFaceIds] = useState<number[]>([])
  const [seamDragEnabled, setSeamDragEnabled] = useState(false)
  const [seamBoundaryAvailable, setSeamBoundaryAvailable] = useState(false)
  const [seamDraftFaceIds, setSeamDraftFaceIds] = useState<number[]>([])
  const [seamSourceFaceIdsForViewer, setSeamSourceFaceIdsForViewer] = useState<number[]>([])
  const [seamDestinationFaceIdsForViewer, setSeamDestinationFaceIdsForViewer] = useState<number[]>([])
  const handlePickFace = useCallback((faceId: number) => {
    setPickedFaceIds((current) => current.includes(faceId) ? current : [...current, faceId].sort((a, b) => a - b))
  }, [])

  // Local gizmo-transform history (live TRS), undoable with Ctrl+Z. A snapshot
  // is taken when a drag starts and committed on release only if it changed.
  const transformHistory = useRef<TransformSnapshot[]>([])
  const pendingTransform = useRef<TransformSnapshot | null>(null)

  const outputUrl = currentJob?.outputUrl ?? ''
  const modelUrl =
    currentJob?.status === 'done' && currentJob.outputUrl
      ? `${apiUrl}${currentJob.outputUrl}`
      : null

  // A .ply/.splat reaching the viewer is always a Gaussian splat here: mesh
  // plys are converted to GLB on import and workflow mesh outputs are .glb.
  const isSplat = /\.(ply|splat)$/i.test(outputUrl)

  // The splat viewer needs binary .splat — route raw workspace .ply through the
  // conversion endpoint; import URLs already point at a .splat via serve-file.
  const splatUrl = outputUrl.startsWith('/workspace/')
    ? `${apiUrl}/optimize/ply-to-splat?path=${encodeURIComponent(outputUrl.slice('/workspace/'.length))}`
    : modelUrl

  // Reset view state when model changes
  useEffect(() => {
    setSelected(false)
    setViewMode('solid')
    setFacePickEnabled(false)
    setPickedFaceIds([])
    setStoreMeshStats(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reset only when the model changes; setters are stable
  }, [modelUrl])

  // Clear the shared selection when the viewer unmounts — the store would
  // otherwise keep it set and flash a stale selection on the next mount.
  useEffect(() => () => setSelected(false), [setSelected])

  // Delete key removes the model from the scene
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'Delete') return
      if (document.activeElement instanceof HTMLInputElement) return
      if (!selected) return
      setCurrentJob(null)
      setSelected(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- setSelected is a stable store setter
  }, [selected, setCurrentJob])

  const handleScreenshot = () => {
    const dataUrl = isSplat
      ? splatRef.current?.screenshot() ?? null
      : canvasRef.current?.toDataURL('image/png') ?? null
    if (!dataUrl) return
    const link = document.createElement('a')
    link.download = `modly-${Date.now()}.png`
    link.href = dataUrl
    link.click()
  }

  // Snapshot the pre-drag pose when a gizmo manipulation starts.
  const handleGizmoDragStart = useCallback(() => {
    if (meshObject) {
      pendingTransform.current = {
        p: meshObject.position.clone(),
        q: meshObject.quaternion.clone(),
        s: meshObject.scale.clone(),
      }
    }
  }, [meshObject])

  // Commit the snapshot on release, but only if the pose actually changed.
  const handleGizmoDragEnd = useCallback(() => {
    const before = pendingTransform.current
    pendingTransform.current = null
    if (!before || !meshObject) return
    const changed = !meshObject.position.equals(before.p)
      || !meshObject.quaternion.equals(before.q)
      || !meshObject.scale.equals(before.s)
    if (changed) transformHistory.current.push(before)
  }, [meshObject])

  // Revert the most recent gizmo manipulation. Returns false when there is
  // nothing to undo, so the caller can fall back to the mesh-history undo.
  const undoTransform = useCallback((): boolean => {
    const prev = transformHistory.current.pop()
    if (!prev || !meshObject) return false
    meshObject.position.copy(prev.p)
    meshObject.quaternion.copy(prev.q)
    meshObject.scale.copy(prev.s)
    return true
  }, [meshObject])

  // Expose transform-undo so the page's Ctrl+Z undoes gizmo edits first.
  useEffect(() => {
    if (!gizmoUndoRef) return
    gizmoUndoRef.current = undoTransform
    return () => { if (gizmoUndoRef.current === undoTransform) gizmoUndoRef.current = null }
  }, [gizmoUndoRef, undoTransform])

  // Drop the transform history when the model changes.
  useEffect(() => {
    transformHistory.current = []
    pendingTransform.current = null
  }, [modelUrl])

  // Memoise the post-processing stack so its children stay referentially stable.
  // @react-three/postprocessing rebuilds (recompiles) all EffectPasses whenever the
  // <EffectComposer> children identity changes; without this, every Viewer3D re-render
  // (e.g. dragging a Lighting slider) recompiles the outline shader. The Outline still
  // tracks selection through the <Selection> context, so nothing here needs to depend
  // on render state.
  const postProcessing = useMemo(() => (
    <EffectComposer
      autoClear={false}
      multisampling={SELECTION_OUTLINE_MULTISAMPLING}
      resolutionScale={SELECTION_OUTLINE_RESOLUTION_SCALE}
      frameBufferType={THREE.HalfFloatType}
    >
      <Outline
        blur={SELECTION_OUTLINE_BLUR}
        edgeStrength={SELECTION_OUTLINE_EDGE_STRENGTH}
        visibleEdgeColor={SELECTION_OUTLINE_VISIBLE_COLOR}
        hiddenEdgeColor={SELECTION_OUTLINE_HIDDEN_COLOR}
        xRay={false}
      />
    </EffectComposer>
  ), [])


  return (
    <ModelErrorBoundary resetKey={modelUrl} fallback={<ModelLoadError />}>
      <div className="relative w-full h-full bg-surface-400">
        {!modelUrl && <EmptyState />}

        {/* Splat path → fully isolated viewer (mkkellogg, outside R3F) */}
        {modelUrl && isSplat && splatUrl ? (
          <SplatViewer ref={splatRef} url={splatUrl} autoRotate={autoRotate} />
        ) : null}

        {/* Mesh path → original Canvas, unchanged */}
        {!isSplat && (
        <Canvas
          onPointerMissed={() => setSelected(false)}
          camera={{ position: [0, 1.5, 4], fov: 45 }}
          dpr={[1, 2]}
          gl={{
            antialias: true,
            preserveDrawingBuffer: true,
            outputColorSpace: THREE.SRGBColorSpace,
          }}
        >
          <color attach="background" args={['#18181b']} />
          <CanvasCapture domRef={canvasRef} />
          <ambientLight intensity={lightSettings.ambientIntensity ?? DEFAULT_LIGHT_SETTINGS.ambientIntensity} />
          <Environment background={false}>
            <Lightformer intensity={2 * (lightSettings.envIntensity ?? DEFAULT_LIGHT_SETTINGS.envIntensity)} position={[0, 4, 4]} scale={8} />
            <Lightformer intensity={0.5 * (lightSettings.envIntensity ?? DEFAULT_LIGHT_SETTINGS.envIntensity)} position={[-4, 2, -4]} scale={6} />
            <Lightformer intensity={0.3 * (lightSettings.envIntensity ?? DEFAULT_LIGHT_SETTINGS.envIntensity)} position={[4, 1, -4]} scale={6} />
          </Environment>

          <gridHelper args={[10, 20, '#3f3f46', '#27272a']} />

          {modelUrl && currentJob ? (
            <Selection enabled={selected}>
              {postProcessing}
              <Suspense fallback={null}>
                <directionalLight position={[5, 8, 5]} color={lightSettings.mainColor} intensity={lightSettings.mainIntensity} castShadow />
                <directionalLight position={[-4, 2, -4]} color={lightSettings.fillColor} intensity={lightSettings.fillIntensity} />
                <MeshModel
                  url={modelUrl}
                  jobId={currentJob.id}
                  viewMode={viewMode}
                  selected={selected}
                  onStats={setStoreMeshStats}
                  onSelect={() => setSelected(true)}
                  onObject={setMeshObject}
                  facePickEnabled={facePickEnabled}
                  onPickFace={handlePickFace}
                  pickedFaceIds={pickedFaceIds}
                  highlightedRegionFaceIds={highlightedRegionFaceIds}
                  seamDragEnabled={seamDragEnabled}
                  seamSourceFaceIds={seamSourceFaceIdsForViewer}
                  seamDestinationFaceIds={seamDestinationFaceIdsForViewer}
                  onSeamDragFaces={(faceIds) => { setSeamDraftFaceIds(faceIds); setPickedFaceIds(faceIds) }}
                  onSeamBoundaryAvailability={setSeamBoundaryAvailable}
                />
              </Suspense>
            </Selection>
          ) : null}

          {selected && meshObject && gizmoMode === 'translate' && (
            <TranslateGizmo object={meshObject} onDragStart={handleGizmoDragStart} onDragEnd={handleGizmoDragEnd} />
          )}
          {selected && meshObject && gizmoMode === 'rotate' && (
            <RotateGizmo object={meshObject} onDragStart={handleGizmoDragStart} onDragEnd={handleGizmoDragEnd} />
          )}
          {selected && meshObject && gizmoMode === 'scale' && (
            <ScaleGizmo object={meshObject} onDragStart={handleGizmoDragStart} onDragEnd={handleGizmoDragEnd} />
          )}

          <OrbitControls
            makeDefault
            enablePan
            enableZoom
            enableRotate
            minDistance={0.5}
            maxDistance={20}
            autoRotate={autoRotate}
            autoRotateSpeed={1.5}
            enableDamping
            dampingFactor={0.05}
          />

          <GizmoHelper alignment="top-right" margin={[72, 72]} renderPriority={modelUrl && currentJob ? 2 : 0}>
            <GizmoBubbles />
          </GizmoHelper>
        </Canvas>
        )}

        {/* Left toolbar — visible only when a model is loaded */}
        {modelUrl && (
          <ViewerToolbar
            viewMode={viewMode}
            autoRotate={autoRotate}
            onViewMode={setViewMode}
            onAutoRotate={() => setAutoRotate((v) => !v)}
            onScreenshot={handleScreenshot}
            showViewModes={!isSplat}
          />
        )}

        {modelUrl && currentJob?.structuredAssetPath && (
          <StructuredAssetReviewPanel
            apiUrl={apiUrl}
            sidecarPath={currentJob.structuredAssetPath}
            faceCount={meshStats?.triangles}
            canPickFaces={meshObject?.userData.modlyFaceIdMappingAvailable === true
              && meshObject.userData.modlyFaceIdCount === meshStats?.triangles}
            facePickEnabled={facePickEnabled}
            pickedFaceIds={pickedFaceIds}
            onHighlightFaces={setHighlightedRegionFaceIds}
            onPickModeChange={setFacePickEnabled}
            onClearPickedFaces={() => setPickedFaceIds([])}
            seamDragEnabled={seamDragEnabled}
            seamDraftFaceIds={seamDraftFaceIds}
            onSeamDragModeChange={setSeamDragEnabled}
            onClearSeamDraft={() => { setSeamDraftFaceIds([]); setPickedFaceIds([]) }}
            onHighlightSeamSource={setSeamSourceFaceIdsForViewer}
            onHighlightSeamDestination={setSeamDestinationFaceIdsForViewer}
            seamBoundaryAvailable={seamBoundaryAvailable}
          />
        )}
        {modelUrl && currentJob?.structuredAssetPath && <StageTelemetryPanel stages={stageTelemetry} />}

        {/* Bottom-left stats overlay */}
        {meshStats && (
          <div className="absolute bottom-4 left-4 pointer-events-none">
            <p className="text-xs text-zinc-500">
              {meshStats.triangles.toLocaleString()} tri &bull; {meshStats.vertices.toLocaleString()} verts
            </p>
          </div>
        )}

        {/* Bottom-right hint */}
        {modelUrl && (
          <div className="absolute bottom-4 right-4 pointer-events-none">
            <p className="text-xs text-zinc-600">
              {selected
                ? <>Click mesh to select &bull; <span className="text-zinc-500">Delete</span> to remove</>
                : 'Drag to rotate \u2022 Scroll to zoom'
              }
            </p>
          </div>
        )}
      </div>
    </ModelErrorBoundary>
  )
}
