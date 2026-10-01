import assert from 'node:assert/strict'
import test from 'node:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { attachHostExecutionTelemetry, createProcessFailureTelemetry, extensionWorkerEnvironment, mayForwardOpenAIKey, persistProcessFailureTelemetry, ProcessRunner, PythonProcessRunner } = await import('./process-runner.ts')

test('host execution telemetry separates wall time from processor inference and keeps absent VRAM unknown', () => {
  const result = attachHostExecutionTelemetry('semantic-stage', 'python_subprocess', 52.25, {
    qualitySummary: { backend: 'cpu', latency_ms: 12.5, accuracy: 0.8 },
  })
  const summary = result.qualitySummary
  assert.equal(summary.backend, 'cpu')
  assert.equal(summary.latency_ms, 12.5)
  const report = summary.runtime_reports.at(-1)
  assert.equal(report.stage_id, 'semantic-stage:host-execution')
  assert.equal(report.execution_backend, 'python_subprocess')
  assert.equal(report.status, 'done')
  assert.equal(report.host_wall_time_ms, 52.25)
  assert.equal(report.latency_kind, 'processor_reported')
  assert.deepEqual(report.inference_backend, { state: 'reported', value: 'cpu' })
  assert.deepEqual(report.resources.accelerator_vram_peak_bytes, { state: 'unknown', value: null })
})

test('host wall time is labeled as such when the processor provides no inference telemetry', () => {
  const result = attachHostExecutionTelemetry('mesh-stage', 'python_subprocess', 18, {})
  const report = result.qualitySummary.runtime_reports[0]
  assert.equal(report.latency_ms, 18)
  assert.equal(report.latency_kind, 'host_wall_time')
  assert.deepEqual(report.inference_backend, { state: 'unknown', value: null })
  assert.deepEqual(report.resources.accelerator_vram_peak_bytes, { state: 'unknown', value: null })
})

test('JS extension worker environments never inherit the OpenAI credential', () => {
  const filtered = extensionWorkerEnvironment({ PATH: '/synthetic/path', OPENAI_API_KEY: 'synthetic-secret' })
  assert.equal(filtered.PATH, '/synthetic/path')
  assert.equal('OPENAI_API_KEY' in filtered, false)
})

test('only the selected and consented built-in semantic extension may receive the key', () => {
  assert.equal(mayForwardOpenAIKey(true, 'identify-part-semantics', {
    provider: 'openai_gpt6_luna', remote_image_consent: true,
  }), true)
  assert.equal(mayForwardOpenAIKey(false, 'identify-part-semantics', {
    provider: 'openai_gpt6_luna', remote_image_consent: true,
  }), false)
  assert.equal(mayForwardOpenAIKey(true, 'other-extension', {
    provider: 'openai_gpt6_luna', remote_image_consent: true,
  }), false)
  assert.equal(mayForwardOpenAIKey(true, 'identify-part-semantics', {
    provider: 'openai_gpt6_luna', remote_image_consent: false,
  }), false)
  assert.equal(mayForwardOpenAIKey(true, 'identify-part-semantics', {
    provider: 'local_north_micro', remote_image_consent: true,
  }), false)
})

test('Python runner retains its actual extension ID for key-forwarding policy', () => {
  const semanticRunner = new PythonProcessRunner(
    'identify-part-semantics', 'python', '/builtin/semantic', 'processor.py', '/workspace', '/worktmp', '', true,
  )
  const unrelatedRunner = new PythonProcessRunner(
    'other-extension', 'python', '/extension', 'processor.py', '/workspace', '/worktmp', '', true,
  )
  const consentedRemote = { provider: 'openai_gpt6_luna', remote_image_consent: true }
  assert.equal(semanticRunner.canForwardOpenAIKey(consentedRemote), true)
  assert.equal(unrelatedRunner.canForwardOpenAIKey(consentedRemote), false)
})

test('failure telemetry records host facts and leaves inference, device, and VRAM unknown', () => {
  const report = createProcessFailureTelemetry('geometry-stage', 31.5, 'failed', 'process_exit')
  assert.equal(report.status, 'failed')
  assert.equal(report.execution_backend, 'python_subprocess')
  assert.equal(report.host_wall_time_ms, 31.5)
  assert.deepEqual(report.inference_backend, { state: 'unknown', value: null })
  assert.deepEqual(report.device, { state: 'unknown', value: null })
  assert.deepEqual(report.resources.accelerator_vram_peak_bytes, { state: 'unknown', value: null })
})

test('failure telemetry is persisted as an inspectable workspace process-run artifact', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'modly-process-telemetry-'))
  try {
    const telemetry = createProcessFailureTelemetry('semantic-stage', 8, 'cancelled', 'cancelled')
    const reportPath = await persistProcessFailureTelemetry(workspace, telemetry)
    const document = JSON.parse(await readFile(reportPath, 'utf8'))
    assert.equal(document.schema_version, 1)
    assert.match(document.run_id, /^[0-9a-f-]{36}$/)
    assert.deepEqual(document.runtime_reports, [telemetry])
    assert.ok(reportPath.startsWith(join(workspace, 'StructuredAssets', 'process-runs')))
  } finally {
    await rm(workspace, { recursive: true, force: true })
  }
})

test('Python subprocess failures carry failed telemetry without claiming accelerator facts', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'modly-process-fail-'))
  try {
    await writeFile(join(dir, 'processor.py'), 'raise RuntimeError("synthetic processor failure")\n')
    const runner = new PythonProcessRunner('synthetic-stage', 'python3', dir, 'processor.py', dir, dir)
    await assert.rejects(runner.run({}, {}), (err) => {
      assert.match(String(err), /synthetic processor failure/)
      assert.equal(err.telemetry.status, 'failed')
      assert.deepEqual(err.telemetry.inference_backend, { state: 'unknown', value: null })
      assert.deepEqual(err.telemetry.resources.accelerator_vram_peak_bytes, { state: 'unknown', value: null })
      return true
    })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('terminating a running Python extension rejects as cancelled and records telemetry', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'modly-process-cancel-'))
  try {
    await writeFile(join(dir, 'processor.py'), 'import time\ntime.sleep(30)\n')
    const runner = new PythonProcessRunner('synthetic-stage', 'python3', dir, 'processor.py', dir, dir)
    const run = runner.run({}, {})
    await new Promise((resolve) => setTimeout(resolve, 150))
    runner.terminate()
    await assert.rejects(run, (err) => {
      assert.equal(err.telemetry.status, 'cancelled')
      assert.equal(err.telemetry.error_kind, 'cancelled')
      return true
    })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('JS worker failures report the worker execution backend and unknown device facts', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'modly-js-process-fail-'))
  try {
    await writeFile(join(dir, 'processor.js'), 'module.exports = async () => { throw new Error("synthetic worker failure") }\n')
    const runner = new ProcessRunner('synthetic-js-stage', dir, 'processor.js', dir, dir)
    await assert.rejects(runner.run({}, {}), (err) => {
      assert.equal(err.telemetry.execution_backend, 'js_worker_thread')
      assert.equal(err.telemetry.status, 'failed')
      assert.deepEqual(err.telemetry.device, { state: 'unknown', value: null })
      return true
    })
    runner.terminate()
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
