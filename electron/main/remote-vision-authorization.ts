const SEMANTIC_EXTENSION_ID = 'identify-part-semantics'
const REMOTE_PROVIDER_ID = 'openai_gpt6_luna'

export const REMOTE_VISION_CONFIRMATION =
  'GPT-6 Luna will send topology-bound part image crops to OpenAI for processing. The images leave this device. Continue?'

/** Strip renderer claims and return consent only after main-process confirmation. */
export async function authorizeProcessRunParams(
  extensionId: string,
  isBuiltinExtension: boolean,
  params: Record<string, unknown>,
  confirmRemoteTransfer: (message: string) => Promise<boolean>,
): Promise<Record<string, unknown> | null> {
  const sanitized = { ...params }
  delete sanitized['remote_image_consent']
  const isRemoteVisionRun = isBuiltinExtension
    && extensionId === SEMANTIC_EXTENSION_ID
    && params['provider'] === REMOTE_PROVIDER_ID
  if (!isRemoteVisionRun) return sanitized
  if (!await confirmRemoteTransfer(REMOTE_VISION_CONFIRMATION)) return null
  sanitized['remote_image_consent'] = true
  return sanitized
}
