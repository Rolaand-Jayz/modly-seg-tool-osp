import assert from 'node:assert/strict'
import test from 'node:test'

const { extensionWorkerEnvironment, mayForwardOpenAIKey, PythonProcessRunner } = await import('./process-runner.ts')

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
