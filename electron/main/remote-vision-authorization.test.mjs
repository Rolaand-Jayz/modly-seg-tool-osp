import assert from 'node:assert/strict'
import test from 'node:test'

const { authorizeProcessRunParams, REMOTE_VISION_CONFIRMATION } = await import('./remote-vision-authorization.ts')

test('renderer-forged consent is stripped and a main-process decline returns no runnable params', async () => {
  const rendererParams = { provider: 'openai_gpt6_luna', remote_image_consent: true, prompt: 'frozen' }
  let confirmationCount = 0
  const result = await authorizeProcessRunParams('identify-part-semantics', true, rendererParams, async (message) => {
    confirmationCount += 1
    assert.equal(message, REMOTE_VISION_CONFIRMATION)
    return false
  })
  assert.equal(confirmationCount, 1)
  assert.equal(result, null)
  assert.equal(rendererParams.remote_image_consent, true)
})

test('only explicit main-process confirmation injects ephemeral remote consent', async () => {
  const rendererParams = { provider: 'openai_gpt6_luna', remote_image_consent: false }
  const result = await authorizeProcessRunParams('identify-part-semantics', true, rendererParams, async () => true)
  assert.equal(result?.remote_image_consent, true)
  assert.equal(rendererParams.remote_image_consent, false)
})

test('renderer consent is stripped for local and non-built-in providers without prompting', async () => {
  const local = await authorizeProcessRunParams(
    'identify-part-semantics', true,
    { provider: 'local_north_micro', remote_image_consent: true },
    async () => assert.fail('local provider must not request remote consent'),
  )
  const userExtension = await authorizeProcessRunParams(
    'identify-part-semantics', false,
    { provider: 'openai_gpt6_luna', remote_image_consent: true },
    async () => assert.fail('non-built-in extension cannot request the remote credential'),
  )
  assert.equal('remote_image_consent' in local, false)
  assert.equal('remote_image_consent' in userExtension, false)
})
