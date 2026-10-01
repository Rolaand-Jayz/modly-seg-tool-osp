import { Worker }      from 'worker_threads'
import { spawn }       from 'child_process'
import { existsSync }  from 'fs'
import { join }        from 'path'
import { performance } from 'node:perf_hooks'
import { mkdir, writeFile } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'

// ─── Worker code for JS process extensions ────────────────────────────────────

const WORKER_CODE = /* js */ `
const { workerData, parentPort } = require('worker_threads')
const path = require('path')
const Module = require('module')

// Resolve modules from the extension's own node_modules
const require_ext = Module.createRequire(path.join(workerData.extDir, '_'))

let processor
try {
  processor = require_ext(path.join(workerData.extDir, workerData.entry))
  if (typeof processor !== 'function') {
    throw new Error('processor.js must export a function as module.exports')
  }
} catch (err) {
  parentPort.postMessage({ type: 'error', message: 'Failed to load processor: ' + String(err) })
  process.exit(1)
}

parentPort.postMessage({ type: 'ready' })

parentPort.on('message', async (msg) => {
  if (msg.action !== 'run') return
  try {
    const context = {
      workspaceDir: workerData.workspaceDir,
      tempDir:      workerData.tempDir,
      nodeId:       msg.input?.nodeId ?? '',
      log:      (m)         => parentPort.postMessage({ type: 'log',      message: String(m) }),
      progress: (pct, label) => parentPort.postMessage({ type: 'progress', percent: pct, label }),
    }
    const result = await processor(msg.input, msg.params, context)
    parentPort.postMessage({ type: 'done', result })
  } catch (err) {
    parentPort.postMessage({ type: 'error', message: String(err) })
  }
})
`

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ProcessInput {
  filePath?: string
  text?:     string
  structuredAssetPath?: string
  /** Per-slot texts for multi-text-input nodes (index = target handle slot). */
  texts?:    (string | undefined)[]
  nodeId?:   string
}

export interface ProcessResult {
  filePath?: string
  text?:     string
  structuredAssetPath?: string
  stageOutputArtifact?: Record<string, unknown>
  qualitySummary?: Record<string, unknown>
}

export interface ProcessExecutionTelemetry {
  stage_id: string
  execution_backend: 'python_subprocess' | 'js_worker_thread'
  status: 'failed' | 'cancelled'
  host_wall_time_ms: number
  latency_ms: number
  latency_kind: 'host_wall_time'
  inference_backend: { state: 'unknown'; value: null }
  device: { state: 'unknown'; value: null }
  resources: { accelerator_vram_peak_bytes: { state: 'unknown'; value: null } }
  error_kind: 'process_error' | 'process_exit' | 'cancelled'
}

export class ProcessExecutionError extends Error {
  readonly telemetry: ProcessExecutionTelemetry
  constructor(message: string, telemetry: ProcessExecutionTelemetry) {
    super(message)
    this.name = 'ProcessExecutionError'
    this.telemetry = telemetry
  }
}

export function createProcessFailureTelemetry(
  extensionId: string,
  hostWallTimeMs: number,
  status: 'failed' | 'cancelled',
  errorKind: ProcessExecutionTelemetry['error_kind'],
  executorBackend: ProcessExecutionTelemetry['execution_backend'] = 'python_subprocess',
): ProcessExecutionTelemetry {
  return {
    stage_id: `${extensionId}:host-execution`, execution_backend: executorBackend, status,
    host_wall_time_ms: Math.max(0, hostWallTimeMs), latency_ms: Math.max(0, hostWallTimeMs),
    latency_kind: 'host_wall_time', inference_backend: { state: 'unknown', value: null },
    device: { state: 'unknown', value: null },
    resources: { accelerator_vram_peak_bytes: { state: 'unknown', value: null } },
    error_kind: errorKind,
  }
}

/** Store host-only failure evidence in the workspace so it remains inspectable after a failed workflow. */
export async function persistProcessFailureTelemetry(
  workspaceDir: string,
  telemetry: ProcessExecutionTelemetry,
): Promise<string> {
  const runId = randomUUID()
  const dir = join(workspaceDir, 'StructuredAssets', 'process-runs', runId)
  await mkdir(dir, { recursive: true })
  const reportPath = join(dir, 'execution-telemetry.json')
  await writeFile(reportPath, `${JSON.stringify({ schema_version: 1, run_id: runId, runtime_reports: [telemetry] }, null, 2)}\n`, { flag: 'wx' })
  return reportPath
}

/** Add host-observed run facts while leaving model/backend and GPU fields unknown unless reported. */
export function attachHostExecutionTelemetry(
  extensionId: string,
  executorBackend: 'python_subprocess' | 'js_worker_thread',
  hostWallTimeMs: number,
  result: ProcessResult,
): ProcessResult {
  const summary = result.qualitySummary ?? {}
  const existingReports = Array.isArray(summary.runtime_reports)
    ? summary.runtime_reports.filter((item): item is Record<string, unknown> =>
      !!item && typeof item === 'object' && !Array.isArray(item))
    : Object.keys(summary).some((key) => key !== 'runtime_reports') ? [summary] : []
  const processorLatency = typeof summary.latency_ms === 'number' && Number.isFinite(summary.latency_ms)
    ? summary.latency_ms : undefined
  const executionReport: Record<string, unknown> = {
    stage_id: `${extensionId}:host-execution`,
    execution_backend: executorBackend,
    status: 'done',
    host_wall_time_ms: Math.max(0, hostWallTimeMs),
    latency_ms: processorLatency ?? Math.max(0, hostWallTimeMs),
    latency_kind: processorLatency === undefined ? 'host_wall_time' : 'processor_reported',
    inference_backend: typeof summary.backend === 'string'
      ? { state: 'reported', value: summary.backend }
      : { state: 'unknown', value: null },
    resources: {
      accelerator_vram_peak_bytes: typeof summary.accelerator_vram_bytes === 'number'
        ? { state: 'reported', value: summary.accelerator_vram_bytes }
        : { state: 'unknown', value: null },
    },
  }
  return {
    ...result,
    qualitySummary: { ...summary, runtime_reports: [...existingReports, executionReport] },
  }
}

export interface IProcessRunner {
  run(
    input:       ProcessInput,
    params:      Record<string, unknown>,
    onProgress?: (percent: number, label: string) => void,
    onLog?:      (message: string) => void,
  ): Promise<ProcessResult>
  terminate(): void
}

/** API credentials are never inherited by arbitrary extension workers. */
export function extensionWorkerEnvironment(baseEnv: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const workerEnv = { ...baseEnv }
  delete workerEnv['OPENAI_API_KEY']
  return workerEnv
}

export function mayForwardOpenAIKey(
  isBuiltinExtension: boolean,
  extensionId: string,
  params: Record<string, unknown>,
): boolean {
  return isBuiltinExtension
    && extensionId === 'identify-part-semantics'
    && params['provider'] === 'openai_gpt6_luna'
    && params['remote_image_consent'] === true
}

// ─── JS ProcessRunner (Worker thread) ────────────────────────────────────────

export class ProcessRunner implements IProcessRunner {
  private extensionId: string
  private worker:   Worker | null = null
  private ready:    boolean       = false
  private extDir:   string
  private entry:    string
  private workspaceDir: string
  private tempDir:  string
  private activeRun: { reject: (error: Error) => void; startedAt: number } | null = null

  constructor(extensionId: string, extDir: string, entry: string, workspaceDir: string, tempDir: string) {
    this.extensionId  = extensionId
    this.extDir       = extDir
    this.entry        = entry
    this.workspaceDir = workspaceDir
    this.tempDir      = tempDir
  }

  private async ensureReady(): Promise<void> {
    if (this.ready && this.worker) return

    return new Promise((resolve, reject) => {
      const worker = new Worker(WORKER_CODE, {
        eval: true,
        env: extensionWorkerEnvironment(process.env),
        workerData: {
          extDir:       this.extDir,
          entry:        this.entry,
          workspaceDir: this.workspaceDir,
          tempDir:      this.tempDir,
        },
      })

      worker.once('message', (msg) => {
        if (msg.type === 'ready') {
          this.worker = worker
          this.ready  = true
          resolve()
        } else if (msg.type === 'error') {
          worker.terminate()
          reject(new Error(msg.message))
        }
      })

      worker.once('error', (err) => {
        reject(err)
      })
    })
  }

  async run(
    input:  ProcessInput,
    params: Record<string, unknown>,
    onProgress?: (percent: number, label: string) => void,
    onLog?:      (message: string) => void,
  ): Promise<ProcessResult> {
    const runStartedAt = performance.now()
    try {
      await this.ensureReady()
    } catch (error) {
      throw new ProcessExecutionError(String(error), createProcessFailureTelemetry(
        this.extensionId, performance.now() - runStartedAt, 'failed', 'process_error', 'js_worker_thread',
      ))
    }
    const worker = this.worker!

    return new Promise((resolve, reject) => {
      const startedAt = runStartedAt
      this.activeRun = { reject, startedAt }
      const fail = (message: string, status: 'failed' | 'cancelled' = 'failed') => {
        if (this.activeRun?.startedAt !== startedAt) return
        this.activeRun = null
        reject(new ProcessExecutionError(message, createProcessFailureTelemetry(
          this.extensionId, performance.now() - startedAt, status,
          status === 'cancelled' ? 'cancelled' : 'process_error', 'js_worker_thread',
        )))
      }
      const handler = (msg: { type: string; result?: ProcessResult; message?: string; percent?: number; label?: string }) => {
        if (msg.type === 'progress') {
          onProgress?.(msg.percent ?? 0, msg.label ?? '')
        } else if (msg.type === 'log') {
          onLog?.(msg.message ?? '')
        } else if (msg.type === 'done') {
          worker.off('message', handler)
          this.activeRun = null
          resolve(msg.result ?? {})
        } else if (msg.type === 'error') {
          worker.off('message', handler)
          fail(msg.message ?? 'Process worker failed')
        }
      }

      worker.on('message', handler)
      worker.postMessage({ action: 'run', input, params })
    })
  }

  terminate(): void {
    if (this.activeRun) {
      const active = this.activeRun
      this.activeRun = null
      active.reject(new ProcessExecutionError('Process extension cancelled', createProcessFailureTelemetry(
        this.extensionId, performance.now() - active.startedAt, 'cancelled', 'cancelled', 'js_worker_thread',
      )))
    }
    this.worker?.terminate()
    this.worker = null
    this.ready  = false
  }
}

// ─── Python ProcessRunner (subprocess, one process per run) ───────────────────
//
// Protocol — stdin:  one JSON line  { input, params, workspaceDir, tempDir }
// Protocol — stdout: JSON lines     { type: 'progress'|'log'|'done'|'error', ... }

export class PythonProcessRunner implements IProcessRunner {
  private extensionId: string
  private pythonExe:    string
  private scriptPath:   string
  private workspaceDir: string
  private tempDir:      string
  private modlyApiDir:  string
  private allowOpenAIKey: boolean
  private activeProcess: ReturnType<typeof spawn> | null = null
  private cancellationRequested = false

  constructor(extensionId: string, pythonExe: string, extDir: string, entry: string, workspaceDir: string, tempDir: string, modlyApiDir = '', allowOpenAIKey = false) {
    this.extensionId  = extensionId
    this.pythonExe    = pythonExe
    this.scriptPath   = join(extDir, entry)
    this.workspaceDir = workspaceDir
    this.tempDir      = tempDir
    this.modlyApiDir  = modlyApiDir
    this.allowOpenAIKey = allowOpenAIKey
  }

  canForwardOpenAIKey(params: Record<string, unknown>): boolean {
    return mayForwardOpenAIKey(this.allowOpenAIKey, this.extensionId, params)
  }

  async run(
    input:  ProcessInput,
    params: Record<string, unknown>,
    onProgress?: (percent: number, label: string) => void,
    onLog?:      (message: string) => void,
  ): Promise<ProcessResult> {
    const startedAt = performance.now()
    this.cancellationRequested = false
    return new Promise((resolve, reject) => {
      // The API key is forwarded only to this built-in semantic extension after
      // explicit provider selection and ephemeral per-run image-transfer consent.
      const remoteSemanticRun = this.canForwardOpenAIKey(params)
      const workerEnv = { ...process.env }
      if (!remoteSemanticRun) delete workerEnv['OPENAI_API_KEY']
      const proc = spawn(this.pythonExe, [this.scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        // Force UTF-8 stdio so Unicode prints from process extensions do not
        // crash under legacy Windows codepages (cp1252/cp932).
        env: {
          ...workerEnv,
          PYTHONUTF8: '1',
          ...(this.modlyApiDir ? { MODLY_API_DIR: this.modlyApiDir } : {}),
        },
      })
      this.activeProcess = proc

      // Send input as a single JSON line on stdin
      proc.stdin.write(JSON.stringify({
        input,
        params,
        nodeId:       input.nodeId ?? '',
        workspaceDir: this.workspaceDir,
        tempDir:      this.tempDir,
      }) + '\n')
      proc.stdin.end()

      let stdoutBuf = ''
      let resolved  = false

      proc.stdout.on('data', (chunk: Buffer) => {
        stdoutBuf += chunk.toString()
        const lines = stdoutBuf.split('\n')
        stdoutBuf = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue
          try {
            const msg = JSON.parse(trimmed) as { type: string; percent?: number; label?: string; message?: string; result?: ProcessResult }
            if (msg.type === 'progress') {
              onProgress?.(msg.percent ?? 0, msg.label ?? '')
            } else if (msg.type === 'log') {
              onLog?.(msg.message ?? '')
            } else if (msg.type === 'done') {
              resolved = true
              resolve(attachHostExecutionTelemetry(
                this.extensionId,
                'python_subprocess',
                performance.now() - startedAt,
                msg.result ?? {},
              ))
            } else if (msg.type === 'error') {
              resolved = true
              reject(new ProcessExecutionError(
                msg.message ?? 'Unknown error',
                createProcessFailureTelemetry(this.extensionId, performance.now() - startedAt, 'failed', 'process_error'),
              ))
            }
          } catch {
            // Non-JSON stdout line — treat as a log message
            onLog?.(trimmed)
          }
        }
      })

      let stderrBuf = ''
      proc.stderr.on('data', (chunk: Buffer) => {
        stderrBuf += chunk.toString()
      })

      proc.on('close', (code) => {
        if (this.activeProcess === proc) this.activeProcess = null
        if (!resolved) {
          if (code === 0) {
            resolve({})
          } else {
            const cancelled = this.cancellationRequested
            reject(new ProcessExecutionError(
              cancelled ? 'Process extension cancelled' : (stderrBuf.trim() || `Python process exited with code ${code}`),
              createProcessFailureTelemetry(this.extensionId, performance.now() - startedAt,
                cancelled ? 'cancelled' : 'failed', cancelled ? 'cancelled' : 'process_exit'),
            ))
          }
        }
      })

      proc.on('error', (err) => {
        if (!resolved) {
          resolved = true
          reject(new ProcessExecutionError(String(err), createProcessFailureTelemetry(
            this.extensionId, performance.now() - startedAt, 'failed', 'process_error',
          )))
        }
      })
    })
  }

  terminate(): void {
    if (!this.activeProcess) return
    this.cancellationRequested = true
    this.activeProcess.kill('SIGTERM')
  }
}

// ─── Helper: find Python executable for an extension ─────────────────────────

export function getExtPythonExe(extDir: string): string | null {
  const candidates = process.platform === 'win32'
    ? [join(extDir, 'venv', 'Scripts', 'python.exe')]
    : [join(extDir, 'venv', 'bin', 'python'), join(extDir, 'venv', 'bin', 'python3')]

  for (const p of candidates) {
    if (existsSync(p)) return p
  }
  return null
}

// ─── Registry (one runner per extension id, reused across calls) ──────────────

const registry = new Map<string, IProcessRunner>()

export function getProcessRunner(
  extensionId:  string,
  extDir:       string,
  entry:        string,
  workspaceDir: string,
  tempDir:      string,
): ProcessRunner {
  if (!registry.has(extensionId)) {
    registry.set(extensionId, new ProcessRunner(extensionId, extDir, entry, workspaceDir, tempDir))
  }
  return registry.get(extensionId)! as ProcessRunner
}

export function getPythonProcessRunner(
  extensionId:  string,
  pythonExe:    string,
  extDir:       string,
  entry:        string,
  workspaceDir: string,
  tempDir:      string,
  modlyApiDir?: string,
  allowOpenAIKey = false,
): PythonProcessRunner {
  if (!registry.has(extensionId)) {
    registry.set(extensionId, new PythonProcessRunner(extensionId, pythonExe, extDir, entry, workspaceDir, tempDir, modlyApiDir, allowOpenAIKey))
  }
  return registry.get(extensionId)! as PythonProcessRunner
}

export function terminateProcessRunner(extensionId: string): void {
  registry.get(extensionId)?.terminate()
  registry.delete(extensionId)
}

export function terminateAllProcessRunners(): void {
  for (const runner of registry.values()) runner.terminate()
  registry.clear()
}
