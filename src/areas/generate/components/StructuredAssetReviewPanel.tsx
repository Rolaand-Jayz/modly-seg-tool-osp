import { useCallback, useEffect, useMemo, useState } from 'react'
import { useWorkflowRunStore, type CorrectionRerunPreview } from '@areas/workflows/workflowRunStore'

type TopologyMapping = {
  topology_revision: string
  state: 'valid' | 'invalid' | 'orphaned' | 'pending-remap'
  element_type: 'mesh' | 'primitive' | 'face' | 'vertex' | 'uv-region'
  element_ids: number[]
}

type Assertion = {
  assertion_id: string
  subject_id: string
  property: string
  value: unknown
  evidence_kind: string
  confidence: { state: string; score?: number | null }
  provenance: Record<string, unknown>
}

type FusedClaim = {
  subject_id: string
  property: string
  status: 'resolved' | 'conflict' | 'unknown'
  value?: unknown
  assertion_ids: string[]
  evidence: Assertion[]
  correction_id?: string | null
  discarded_assertion_ids: string[]
  stale_assertion_ids?: string[]
}

type Region = { region_id: string; mapping: TopologyMapping; [key: string]: unknown }
type StructuredAsset = {
  asset_id: string
  topology_revision: string
  part_segments: Region[]
  material_regions: Region[]
  assertions: Assertion[]
  corrections: Array<{ correction_id: string; property: string; value: unknown; target: TopologyMapping; status: string }>
  validation_state: string
  topology_counts: { face_count: number }
}

function statusColor(status: string): string {
  if (status === 'conflict' || status === 'orphaned' || status === 'invalid') return 'text-red-300 border-red-500/40 bg-red-500/10'
  if (status === 'unknown' || status === 'ambiguous' || status === 'pending-remap' || status === 'needs-review') return 'text-amber-200 border-amber-500/40 bg-amber-500/10'
  return 'text-emerald-200 border-emerald-500/30 bg-emerald-500/10'
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return `sha256:${Array.from(new Uint8Array(digest), (v) => v.toString(16).padStart(2, '0')).join('')}`
}

async function fetchFusedClaims(apiUrl: string, sidecarPath: string): Promise<FusedClaim[]> {
  const response = await fetch(`${apiUrl}/structured-assets/fuse`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sidecar_path: sidecarPath }),
  })
  if (!response.ok) throw new Error(`Could not resolve evidence states (${response.status})`)
  const value = await response.json() as { claims?: FusedClaim[] }
  if (!Array.isArray(value.claims)) throw new Error('Evidence fusion returned an invalid claim list')
  return value.claims
}

/** Review and correct a workflow-produced sidecar without hiding model evidence. */
export default function StructuredAssetReviewPanel({
  apiUrl, sidecarPath, faceCount, canPickFaces, facePickEnabled, pickedFaceIds, onPickModeChange, onClearPickedFaces, onHighlightFaces,
  seamDragEnabled, seamDraftFaceIds, onSeamDragModeChange, onClearSeamDraft, onHighlightSeamSource, onHighlightSeamDestination,
  seamBoundaryAvailable,
}: {
  apiUrl: string
  sidecarPath: string
  faceCount?: number
  canPickFaces: boolean
  facePickEnabled: boolean
  pickedFaceIds: number[]
  onPickModeChange: (enabled: boolean) => void
  onClearPickedFaces: () => void
  onHighlightFaces: (faceIds: number[]) => void
  seamDragEnabled: boolean
  seamDraftFaceIds: number[]
  onSeamDragModeChange: (enabled: boolean) => void
  onClearSeamDraft: () => void
  onHighlightSeamSource: (faceIds: number[]) => void
  onHighlightSeamDestination: (faceIds: number[]) => void
  seamBoundaryAvailable: boolean
}): JSX.Element {
  const [activeSidecarPath, setActiveSidecarPath] = useState(sidecarPath)
  const [asset, setAsset] = useState<StructuredAsset | null>(null)
  const [fusedClaims, setFusedClaims] = useState<FusedClaim[]>([])
  const [digest, setDigest] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [regionKey, setRegionKey] = useState('')
  const [property, setProperty] = useState('part.semantic-label')
  const [valueText, setValueText] = useState('"unknown"')
  const [saving, setSaving] = useState(false)
  const [open, setOpen] = useState(false)
  const [destinationRegionId, setDestinationRegionId] = useState('')
  const [movedFaceText, setMovedFaceText] = useState('')
  const [seamPreview, setSeamPreview] = useState<{ moved: number[]; sourceBefore: number[]; sourceAfter: number[]; destinationBefore: number[]; destinationAfter: number[] } | null>(null)
  const [seamCorrectionId, setSeamCorrectionId] = useState('')
  const [seamUndone, setSeamUndone] = useState(false)
  const [rerunPreview, setRerunPreview] = useState<CorrectionRerunPreview | null>(null)
  const [rerunProperty, setRerunProperty] = useState('')
  const [rerunError, setRerunError] = useState('')
  const [rerunNotice, setRerunNotice] = useState('')
  const [rerunning, setRerunning] = useState(false)

  const load = useCallback(async (path = activeSidecarPath) => {
    setLoading(true)
    setError('')
    try {
      const safePath = path.split('/').map(encodeURIComponent).join('/')
      const response = await fetch(`${apiUrl}/workspace/${safePath}`, { cache: 'no-store' })
      if (!response.ok) throw new Error(`Could not load Structured Asset (${response.status})`)
      const bytes = await response.arrayBuffer()
      const parsed = JSON.parse(new TextDecoder().decode(bytes)) as StructuredAsset
      if (!parsed || typeof parsed.asset_id !== 'string' || typeof parsed.topology_revision !== 'string') throw new Error('Sidecar is missing its asset or topology identity')
      const claims = await fetchFusedClaims(apiUrl, path)
      setAsset(parsed)
      setFusedClaims(claims)
      setDigest(await sha256(bytes))
      setRegionKey((current) => current || (parsed.part_segments[0] ? `part:${parsed.part_segments[0].region_id}` : parsed.material_regions[0] ? `material:${parsed.material_regions[0].region_id}` : ''))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      setAsset(null)
    } finally {
      setLoading(false)
    }
  }, [apiUrl, activeSidecarPath])

  useEffect(() => { setActiveSidecarPath(sidecarPath) }, [sidecarPath])
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    setRerunPreview(null)
    setRerunError('')
    setRerunNotice('')
  }, [activeSidecarPath])
  useEffect(() => {
    if (facePickEnabled) setMovedFaceText(pickedFaceIds.join(', '))
  }, [facePickEnabled, pickedFaceIds])

  const regions = useMemo(() => [
    ...((asset?.part_segments ?? []).map((region) => ({ ...region, kind: 'Part' as const, key: `part:${region.region_id}` }))),
    ...((asset?.material_regions ?? []).map((region) => ({ ...region, kind: 'Material' as const, key: `material:${region.region_id}` }))),
  ], [asset])
  const selectedRegion = regions.find((region) => region.key === regionKey)
  const partRegions = asset?.part_segments ?? []

  const previewCorrectionRerun = async (correctionProperty: string) => {
    setRerunProperty(correctionProperty)
    setRerunPreview(null)
    setRerunError('')
    setRerunNotice('')
    try {
      const preview = await useWorkflowRunStore.getState().previewCorrectionRerun(correctionProperty, activeSidecarPath)
      setRerunPreview(preview)
    } catch (cause) {
      setRerunPreview(null)
      setRerunError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  const runCorrectionStages = async () => {
    if (!rerunPreview?.ready || !rerunProperty || rerunning) return
    setRerunning(true)
    setRerunError('')
    setRerunNotice('')
    try {
      const result = await useWorkflowRunStore.getState().rerunForCorrection(rerunProperty, activeSidecarPath)
      if (result.status === 'done') {
        setRerunPreview(null)
        setRerunNotice('The affected workflow stages finished. The viewer will refresh from the workflow output.')
        const latestSidecarPath = useWorkflowRunStore.getState().runState.structuredAssetPath
        if (latestSidecarPath && latestSidecarPath !== activeSidecarPath) setActiveSidecarPath(latestSidecarPath)
        else await load(latestSidecarPath || activeSidecarPath)
        return
      }
      setRerunPreview(result.preview)
      setRerunError(result.status === 'error'
        ? `Rerun failed${result.error ? `: ${result.error}` : '.'} Affected stage outputs are unavailable until rerun succeeds.`
        : `Rerun was not started: ${result.preview.blockers.join(' ') || 'the workflow is no longer ready.'}`)
    } catch (cause) {
      setRerunError(cause instanceof Error ? `Rerun failed: ${cause.message}` : `Rerun failed: ${String(cause)}`)
    } finally {
      setRerunning(false)
    }
  }

  useEffect(() => {
    const mapping = selectedRegion?.mapping
    onHighlightFaces(mapping?.state === 'valid' && mapping.element_type === 'face' ? mapping.element_ids : [])
    return () => onHighlightFaces([])
  }, [onHighlightFaces, selectedRegion])

  const destinationPart = partRegions.find((part) => part.region_id === destinationRegionId)
  useEffect(() => {
    const source = selectedRegion?.kind === 'Part' && selectedRegion.mapping.state === 'valid' && selectedRegion.mapping.element_type === 'face'
      ? selectedRegion.mapping.element_ids : []
    const destination = destinationPart?.mapping.state === 'valid' && destinationPart.mapping.element_type === 'face'
      ? destinationPart.mapping.element_ids : []
    onHighlightSeamSource(source)
    onHighlightSeamDestination(destination)
    return () => { onHighlightSeamSource([]); onHighlightSeamDestination([]) }
  }, [destinationPart, onHighlightSeamDestination, onHighlightSeamSource, selectedRegion])

  const previewSeamEdit = (draftText = movedFaceText) => {
    setError('')
    if (!selectedRegion || selectedRegion.kind !== 'Part') { setError('Select a part region before editing a seam.'); return }
    const destination = partRegions.find((part) => part.region_id === destinationRegionId)
    if (!destination || destination.region_id === selectedRegion.region_id) { setError('Choose a different destination part.'); return }
    if (selectedRegion.mapping.state !== 'valid' || destination.mapping.state !== 'valid'
      || selectedRegion.mapping.element_type !== 'face' || destination.mapping.element_type !== 'face') {
      setError('Both regions need valid face mappings before seam edits.'); return
    }
    const moved = [...new Set(draftText.split(/[\s,]+/).filter(Boolean).map(Number))]
    if (moved.length === 0 || moved.some((face) => !Number.isSafeInteger(face) || face < 0)) {
      setError('Enter one or more nonnegative face IDs.'); return
    }
    const sourceBefore = selectedRegion.mapping.element_ids
    const destinationBefore = destination.mapping.element_ids
    const sourceFaces = new Set(sourceBefore)
    const destinationFaces = new Set(destinationBefore)
    const movedFaces = new Set(moved)
    if (!moved.every((face) => sourceFaces.has(face)) || moved.some((face) => destinationFaces.has(face))) {
      setError('Every moved face must belong to the selected source part and not the destination.'); return
    }
    const sourceAfter = sourceBefore.filter((face) => !movedFaces.has(face))
    if (sourceAfter.length === 0) { setError('A seam edit cannot remove every face from the source part.'); return }
    setSeamPreview({ moved: moved.sort((a, b) => a - b), sourceBefore, sourceAfter, destinationBefore, destinationAfter: [...new Set([...destinationBefore, ...moved])].sort((a, b) => a - b) })
  }

  const acceptViewerSeamDraft = (faceIds: number[]) => {
    const text = faceIds.join(', ')
    setMovedFaceText(text)
    if (faceIds.length > 0) previewSeamEdit(text)
  }
  useEffect(() => {
    if (seamDragEnabled && seamDraftFaceIds.length > 0) acceptViewerSeamDraft(seamDraftFaceIds)
    else if (seamDragEnabled) setSeamPreview(null)
  // The viewer publishes face IDs as the drag moves; each update is previewed
  // against the current topology-bound source and destination mappings.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seamDragEnabled, seamDraftFaceIds])

  const confirmSeamEdit = async () => {
    if (!asset || !selectedRegion || selectedRegion.kind !== 'Part' || !seamPreview || !digest || rerunning) return
    setSaving(true)
    setError('')
    try {
      const correctionId = crypto.randomUUID()
      const response = await fetch(`${apiUrl}/structured-assets/seam-edits`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sidecar_path: activeSidecarPath,
          expected_sidecar_digest: digest,
          correction_id: correctionId,
          source_region_id: selectedRegion.region_id,
          destination_region_id: destinationRegionId,
          moved_face_ids: seamPreview.moved,
        }),
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: { message?: string } } | null
        throw new Error(payload?.detail?.message ?? `Seam edit was rejected (${response.status})`)
      }
      const updated = await response.json() as StructuredAsset
      setAsset(updated)
      setFusedClaims(await fetchFusedClaims(apiUrl, activeSidecarPath))
      setSeamCorrectionId(correctionId)
      setSeamUndone(false)
      setSeamPreview(null)
      onSeamDragModeChange(false)
      setMovedFaceText('')
      onClearPickedFaces()
      onPickModeChange(false)
      const refreshed = await fetch(`${apiUrl}/workspace/${activeSidecarPath.split('/').map(encodeURIComponent).join('/')}`, { cache: 'no-store' })
      if (!refreshed.ok) throw new Error('Seam edit was saved, but the updated sidecar could not be reloaded.')
      setDigest(await sha256(await refreshed.arrayBuffer()))
      void previewCorrectionRerun('part.membership')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving(false)
    }
  }

  const changeSeamHistory = async (action: 'undo' | 'redo') => {
    if (!seamCorrectionId || !digest || saving || rerunning) return
    setSaving(true)
    setError('')
    try {
      const response = await fetch(`${apiUrl}/structured-assets/seam-edits/${action}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sidecar_path: activeSidecarPath, expected_sidecar_digest: digest,
          correction_id: seamCorrectionId, history_correction_id: crypto.randomUUID() }),
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: { message?: string } } | null
        throw new Error(payload?.detail?.message ?? `Could not ${action} the seam edit (${response.status})`)
      }
      const updated = await response.json() as StructuredAsset
      setAsset(updated)
      setFusedClaims(await fetchFusedClaims(apiUrl, activeSidecarPath))
      setSeamUndone(action === 'undo')
      const refreshed = await fetch(`${apiUrl}/workspace/${activeSidecarPath.split('/').map(encodeURIComponent).join('/')}`, { cache: 'no-store' })
      if (!refreshed.ok) throw new Error(`Seam edit was ${action}ne, but the updated sidecar could not be reloaded.`)
      setDigest(await sha256(await refreshed.arrayBuffer()))
      void previewCorrectionRerun('part.membership.history')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving(false)
    }
  }

  const saveCorrection = async () => {
    if (!asset || !selectedRegion || !digest || rerunning) return
    setSaving(true)
    setError('')
    try {
      const value = JSON.parse(valueText) as unknown
      const response = await fetch(`${apiUrl}/structured-assets/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sidecar_path: activeSidecarPath,
          expected_sidecar_digest: digest,
          correction_id: crypto.randomUUID(),
          subject_id: selectedRegion.region_id,
          property,
          value,
          target: selectedRegion.mapping,
        }),
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: { message?: string } } | null
        throw new Error(payload?.detail?.message ?? `Correction was rejected (${response.status})`)
      }
      const updated = await response.json() as StructuredAsset
      setAsset(updated)
      setFusedClaims(await fetchFusedClaims(apiUrl, activeSidecarPath))
      const bytes = await (await fetch(`${apiUrl}/workspace/${activeSidecarPath.split('/').map(encodeURIComponent).join('/')}`, { cache: 'no-store' })).arrayBuffer()
      setDigest(await sha256(bytes))
      void previewCorrectionRerun(property)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="absolute right-3 top-3 z-20 w-[min(360px,calc(100%-24px))] overflow-hidden rounded-xl border border-zinc-700/80 bg-zinc-950/95 text-zinc-200 shadow-xl backdrop-blur">
      <button className="flex w-full items-center gap-2 px-3 py-2.5 text-left" onClick={() => setOpen((v) => !v)}>
        <span className="text-xs font-semibold">Structured Asset review</span>
        {asset && <span className={`rounded border px-1.5 py-0.5 text-[9px] ${statusColor(asset.validation_state)}`}>{asset.validation_state}</span>}
        <span className="ml-auto text-[10px] text-zinc-500">{open ? 'Hide' : 'Review'}</span>
      </button>
      {open && <div className="max-h-[65vh] space-y-3 overflow-y-auto border-t border-zinc-800 p-3 text-[11px]">
        {loading && <p className="text-zinc-400">Loading saved asset…</p>}
        {error && <p role="alert" className="rounded border border-red-500/30 bg-red-500/10 p-2 text-red-200">{error}</p>}
        {rerunError && <p role="status" className="rounded border border-amber-500/30 bg-amber-500/10 p-2 text-amber-100">{rerunError}</p>}
        {rerunNotice && <p role="status" className="rounded border border-emerald-500/30 bg-emerald-500/10 p-2 text-emerald-100">{rerunNotice}</p>}
        {rerunPreview && <div className="space-y-2 rounded border border-sky-500/30 bg-sky-500/5 p-2" aria-live="polite">
          <div className="font-semibold text-sky-100">Workflow rerun preview</div>
          <p className="text-zinc-300">Saved correction: <span className="font-mono">{rerunProperty}</span></p>
          <div><span className="text-zinc-400">Would rerun:</span> {rerunPreview.plan.rerunNodes.length
            ? rerunPreview.plan.rerunNodes.map((node) => node.label).join(', ')
            : 'no configured stages'}</div>
          <div><span className="text-zinc-400">Old results to replace:</span> {rerunPreview.plan.invalidateNodeIds.length
            ? rerunPreview.plan.invalidateNodeIds.join(', ')
            : 'none'}</div>
          <div><span className="text-zinc-400">Unaffected results kept:</span> {rerunPreview.plan.preserveNodeIds.length
            ? rerunPreview.plan.preserveNodeIds.join(', ')
            : 'none'}</div>
          {rerunPreview.plan.notConfiguredCapabilities.length > 0 && <div className="text-amber-100">
            Not present in this workflow: {rerunPreview.plan.notConfiguredCapabilities.join(', ')}
          </div>}
          {rerunPreview.blockers.length > 0 && <ul className="list-disc space-y-1 pl-4 text-amber-100">
            {rerunPreview.blockers.map((blocker, index) => <li key={`${index}-${blocker}`}>{blocker}</li>)}
          </ul>}
          <p className="text-zinc-400">Running these stages uses the saved correction and current geometry. Review the list before continuing.</p>
          <div className="flex gap-2">
            <button className="rounded border border-sky-500/40 px-2 py-1 text-sky-100 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={!rerunPreview.ready || saving || rerunning} onClick={() => void runCorrectionStages()}>
              {rerunning ? 'Rerunning…' : 'Run listed stages'}
            </button>
            <button className="rounded border border-zinc-700 px-2 py-1 text-zinc-300 disabled:opacity-40"
              disabled={rerunning} onClick={() => { setRerunPreview(null); setRerunError(''); setRerunNotice('') }}>Dismiss</button>
          </div>
        </div>}
        {asset && <>
          <div className="grid grid-cols-2 gap-2">
            <div><div className="text-zinc-500">Asset</div><div className="truncate font-mono">{asset.asset_id}</div></div>
            <div><div className="text-zinc-500">Parts / materials</div><div>{asset.part_segments.length} / {asset.material_regions.length}</div></div>
            <div><div className="text-zinc-500">Revision</div><div className="truncate font-mono">{asset.topology_revision}</div></div>
            <div><div className="text-zinc-500">Face mapping</div><div>{faceCount ?? '…'} / {asset.topology_counts.face_count}</div></div>
            <div><div className="text-zinc-500">Corrections</div><div>{asset.corrections.length}</div></div>
          </div>
          <div className="space-y-1">
            <label className="block text-zinc-400" htmlFor="structured-review-region">Region to review</label>
            <select id="structured-review-region" value={regionKey} onChange={(event) => { setRegionKey(event.target.value); setSeamPreview(null); onPickModeChange(false); onSeamDragModeChange(false); onClearSeamDraft(); setProperty(event.target.value.startsWith('material:') ? 'material.identity' : 'part.semantic-label'); setValueText(event.target.value.startsWith('material:') ? '"unknown"' : '"unknown"') }} className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5">
              {regions.map((region) => <option key={region.key} value={region.key}>{region.kind}: {region.region_id} ({region.mapping.element_ids.length} {region.mapping.element_type}s, {region.mapping.state})</option>)}
              {regions.length === 0 && <option value="">No segmented regions yet</option>}
            </select>
          {selectedRegion && <div className={`rounded border px-2 py-1.5 ${statusColor(selectedRegion.mapping.state)}`}>
              {selectedRegion.mapping.element_type} mapping is {selectedRegion.mapping.state}; {selectedRegion.mapping.element_ids.length} elements, revision {selectedRegion.mapping.topology_revision}
          </div>}
          {selectedRegion?.kind === 'Part' && <div className="space-y-1 border-t border-zinc-800 pt-3">
            <div className="font-medium">Adjust part seam</div>
            <p className="text-zinc-500">Move selected face IDs between parts. Preview records exact old/new membership and does not move mesh points. Geometry remains unchanged.</p>
            <label className="block text-zinc-400" htmlFor="seam-destination">Move faces into</label>
            <select id="seam-destination" value={destinationRegionId} onChange={(event) => { setDestinationRegionId(event.target.value); setSeamPreview(null); onSeamDragModeChange(false); onClearSeamDraft() }} className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5">
              <option value="">Choose destination part</option>
              {partRegions.filter((part) => part.region_id !== selectedRegion.region_id).map((part) => <option key={part.region_id} value={part.region_id}>{part.region_id}</option>)}
            </select>
            <div className="flex gap-1.5">
              <button disabled={!canPickFaces || selectedRegion.mapping.state !== 'valid' || !destinationRegionId || facePickEnabled || seamDragEnabled} onClick={() => { onClearPickedFaces(); setMovedFaceText(''); onPickModeChange(true) }} className="flex-1 rounded border border-violet-500/30 px-2 py-1.5 text-violet-200 hover:bg-violet-500/10 disabled:opacity-40">Brush faces</button>
              <button disabled={!canPickFaces || selectedRegion.mapping.state !== 'valid' || !destinationRegionId || facePickEnabled || (!seamDragEnabled && !seamBoundaryAvailable)} onClick={() => { onClearSeamDraft(); setMovedFaceText(''); setSeamPreview(null); onSeamDragModeChange(!seamDragEnabled) }} className="flex-1 rounded border border-pink-500/40 px-2 py-1.5 text-pink-100 hover:bg-pink-500/10 disabled:opacity-40">{seamDragEnabled ? 'Finish seam drag' : 'Drag boundary point'}</button>
              {facePickEnabled && <button onClick={() => onPickModeChange(false)} className="rounded border border-emerald-500/30 px-2 py-1.5 text-emerald-200 hover:bg-emerald-500/10">Finish picking</button>}
            </div>
            {!canPickFaces && <p className="text-amber-200">Surface picking is disabled because this viewer cannot safely map its faces to the sidecar face order.</p>}
            {canPickFaces && destinationRegionId && !seamBoundaryAvailable && <p className="text-amber-200">No shared mesh edge joins these mapped parts, so a boundary point cannot be dragged for this pair.</p>}
            {facePickEnabled && <p className="text-violet-200">Drag across the source part on the model. Selected face IDs: {pickedFaceIds.length ? pickedFaceIds.join(', ') : 'none yet'}.</p>}
            {seamDragEnabled && <p className="text-pink-100">Drag a pink point on the shared boundary across the surface. The swept source faces are previewed for transfer; mesh vertices stay fixed.</p>}
            {seamDraftFaceIds.length > 0 && <p className="text-pink-100">Boundary drag preview: {seamDraftFaceIds.length} source faces selected for transfer.</p>}
            <label className="block text-zinc-400" htmlFor="seam-face-ids">Faces to transfer</label>
            <input id="seam-face-ids" value={movedFaceText} onChange={(event) => { setMovedFaceText(event.target.value); setSeamPreview(null) }} placeholder="Pick on surface or enter IDs" className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono" />
            {pickedFaceIds.length > 0 && !facePickEnabled && <button onClick={() => { onClearPickedFaces(); setMovedFaceText(''); setSeamPreview(null) }} className="w-full rounded px-2 py-1 text-zinc-400 hover:bg-zinc-800">Clear picked faces</button>}
            <button disabled={selectedRegion.mapping.state !== 'valid' || !destinationRegionId} onClick={() => previewSeamEdit()} className="w-full rounded border border-zinc-700 px-2 py-1.5 text-zinc-200 hover:bg-zinc-800 disabled:opacity-40">Preview face membership change</button>
            {seamPreview && <div className="space-y-1 rounded border border-amber-500/30 bg-amber-500/5 p-2">
              <div className="font-medium text-amber-100">Review before saving</div>
              <div>Faces transferred: {seamPreview.moved.length.toLocaleString()}{seamPreview.moved.length > 0 ? ` · ${seamPreview.moved.slice(0, 50).join(', ')}${seamPreview.moved.length > 50 ? ', …' : ''}` : ''}</div>
              <div>{selectedRegion.region_id}: {seamPreview.sourceBefore.length} → {seamPreview.sourceAfter.length} faces</div>
              <div>{destinationRegionId}: {seamPreview.destinationBefore.length} → {seamPreview.destinationAfter.length} faces</div>
              <button disabled={saving || rerunning} onClick={() => void confirmSeamEdit()} className="w-full rounded bg-amber-500/20 px-2 py-1.5 font-medium text-amber-100 hover:bg-amber-500/30 disabled:opacity-40">{saving ? 'Saving…' : 'Confirm seam correction'}</button>
              <button onClick={() => { setSeamPreview(null); onSeamDragModeChange(false); onClearSeamDraft(); setMovedFaceText('') }} className="w-full rounded px-2 py-1 text-zinc-400 hover:bg-zinc-800">Cancel preview</button>
            </div>}
          </div>}
          </div>
          <div className="space-y-1">
            <div className="text-zinc-400">Combined evidence states</div>
            {fusedClaims.filter((claim) => !selectedRegion || claim.subject_id === selectedRegion.region_id).map((claim) => <div key={`${claim.subject_id}:${claim.property}`} className={`rounded border p-2 ${statusColor(claim.status)}`}>
              <div className="flex items-center justify-between gap-2"><span className="font-medium">{claim.property}</span><span>{claim.status}</span></div>
              {claim.status === 'resolved' && <div className="mt-1 break-words">Value: {JSON.stringify(claim.value)}</div>}
              {claim.status === 'conflict' && <div className="mt-1">Conflicting model values: {claim.evidence.map((item) => JSON.stringify(item.value)).join(' · ')}</div>}
              {claim.status === 'unknown' && <div className="mt-1">No supported result is available.</div>}
              {claim.correction_id && <div className="mt-1 text-zinc-300">Resolved by saved user correction {claim.correction_id}; model assertions remain in the evidence below.</div>}
              {(claim.stale_assertion_ids?.length ?? 0) > 0 && <div className="mt-2 rounded border border-amber-500/30 bg-amber-500/5 p-2 text-amber-100">
                <div className="font-medium">Stale model evidence · excluded from current resolution</div>
                {asset.assertions.filter((item) => claim.stale_assertion_ids?.includes(item.assertion_id)).map((item) => <div key={item.assertion_id} className="mt-1 break-words text-amber-100/80">
                  {item.assertion_id}: {JSON.stringify(item.value)} · {item.evidence_kind} · {String(item.provenance.adapter_id ?? item.provenance.model_id ?? 'source unavailable')}
                </div>)}
              </div>}
            </div>)}
            <div className="border-t border-zinc-800 pt-2 text-zinc-400">Original evidence and model confidence</div>
            {(asset.assertions.filter((item) => !selectedRegion || item.subject_id === selectedRegion.region_id).slice(0, 16)).map((item) => <div key={item.assertion_id} className="rounded border border-zinc-800 p-2">
              <div className="flex items-center justify-between gap-2"><span className="font-medium">{item.property}</span><span className={`rounded border px-1.5 py-0.5 ${statusColor(item.confidence.state)}`}>{item.confidence.state}{item.confidence.score == null ? '' : ` · ${item.confidence.score.toFixed(2)}`}</span></div>
              <div className="mt-1 break-words text-zinc-300">{JSON.stringify(item.value)}</div>
              <div className="mt-1 text-zinc-500">{item.evidence_kind} · {String(item.provenance.adapter_id ?? item.provenance.model_id ?? item.provenance.evidence_source ?? 'source unavailable')}</div>
            </div>)}
            {asset.assertions.length === 0 && <p className="text-zinc-500">No assertions are available yet. The empty state is preserved as unknown.</p>}
          </div>
          <div className="space-y-1 border-t border-zinc-800 pt-3">
            <div className="font-medium">Save a correction</div>
            <p className="text-zinc-500">This records a user correction for this exact topology region. It does not train the model.</p>
            <label className="block text-zinc-400" htmlFor="structured-correction-property">What should change?</label>
            <select id="structured-correction-property" value={property} onChange={(event) => { setProperty(event.target.value); setValueText(event.target.value === 'pbr.base_color_linear' ? '[0.5, 0.5, 0.5]' : event.target.value === 'pbr.roughness' ? '0.5' : event.target.value === 'pbr.metallic' ? '0' : '"unknown"') }} className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5">
              {selectedRegion?.kind === 'Material' ? <>
                <option value="material.identity">Material identity</option>
                <option value="material.name">Material name</option>
                <option value="pbr.base_color_linear">Base color (linear RGB)</option>
                <option value="pbr.roughness">Roughness</option>
                <option value="pbr.metallic">Metallic</option>
              </> : <option value="part.semantic-label">Part name or meaning</option>}
            </select>
            {property === 'material.identity' ? <select aria-label="Corrected material identity" value={valueText} onChange={(event) => setValueText(event.target.value)} className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5">
              {['unknown', 'ambiguous', 'rubber_latex', 'glass', 'clear_plastic', 'paint_plaster_enamel', 'metal'].map((label) => <option key={label} value={JSON.stringify(label)}>{label.replaceAll('_', ' ')}</option>)}
            </select> : property === 'pbr.roughness' || property === 'pbr.metallic' ? <div>
              <label className="flex justify-between text-zinc-400"><span>{property === 'pbr.roughness' ? 'Roughness' : 'Metallic'}</span><span>{Number(valueText).toFixed(2)}</span></label>
              <input aria-label={property} type="range" min="0" max="1" step="0.01" value={Number(valueText)} onChange={(event) => setValueText(String(Number(event.target.value)))} className="w-full" />
            </div> : property === 'pbr.base_color_linear' ? <input aria-label="Base color linear RGB as JSON" value={valueText} onChange={(event) => setValueText(event.target.value)} placeholder="[0.4, 0.4, 0.4]" className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono" /> : <input aria-label="Corrected label or material name" value={valueText.replace(/^"|"$/g, '')} onChange={(event) => setValueText(JSON.stringify(event.target.value))} placeholder={property === 'material.name' ? 'e.g. Door paint' : 'e.g. front-left wheel'} className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5" />}
            <button disabled={!selectedRegion || selectedRegion.mapping.state !== 'valid' || saving || rerunning} onClick={() => void saveCorrection()} className="w-full rounded bg-violet-500/20 px-2 py-1.5 font-medium text-violet-200 hover:bg-violet-500/30 disabled:cursor-not-allowed disabled:opacity-40">{saving ? 'Saving…' : 'Save correction for selected region'}</button>
          </div>
          {asset.corrections.length > 0 && <div className="space-y-1 border-t border-zinc-800 pt-2"><div className="text-zinc-400">Saved correction precedence</div>{asset.corrections.map((item) => <div key={item.correction_id} className={`rounded border px-2 py-1.5 ${statusColor(item.status)}`}>{item.property}: {JSON.stringify(item.value)} · {item.status} · user-confirmed</div>)}</div>}
          {seamCorrectionId && <div className="flex gap-2 border-t border-zinc-800 pt-2">
            <button disabled={saving || rerunning || seamUndone} onClick={() => void changeSeamHistory('undo')} className="flex-1 rounded border border-zinc-700 px-2 py-1.5 disabled:opacity-40">Undo latest seam edit</button>
            <button disabled={saving || rerunning || !seamUndone} onClick={() => void changeSeamHistory('redo')} className="flex-1 rounded border border-zinc-700 px-2 py-1.5 disabled:opacity-40">Redo seam edit</button>
          </div>}
        </>}
      </div>}
    </section>
  )
}
