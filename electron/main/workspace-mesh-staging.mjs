import { constants as fsConstants } from 'node:fs'
import { lstat, mkdir, open, realpath, rm } from 'node:fs/promises'
import path from 'node:path'

const ALLOWED_MESH_EXTENSIONS = new Set(['.glb', '.gltf'])
const MAX_GLTF_DOCUMENT_BYTES = 64 * 1024 * 1024
const MAX_EXTERNAL_RESOURCES = 256
const MAX_TOTAL_PACKAGE_BYTES = 512 * 1024 * 1024

function actionableError(message) {
  return new Error(`Could not stage mesh: ${message}`)
}

function sourceSnapshotChanged(before, after) {
  return ['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'].some((key) => {
    const beforeValue = before[key]
    const afterValue = after[key]
    if (beforeValue === undefined || afterValue === undefined) return false
    if ((key === 'dev' || key === 'ino') && (beforeValue === 0n || afterValue === 0n)) return false
    return beforeValue !== afterValue
  })
}

function assertContained(root, candidate) {
  const relative = path.relative(root, candidate)
  if (!relative || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))) return
  throw actionableError('a glTF resource resolves outside the selected mesh folder. Keep all referenced files inside that folder and select the glTF again.')
}

function decodeLocalResourceUri(uri, description) {
  if (typeof uri !== 'string' || !uri) {
    throw actionableError(`${description} has an invalid resource URI.`)
  }
  if (/^data:/i.test(uri)) return null
  if (/^[a-z][a-z\d+.-]*:/i.test(uri) || uri.startsWith('//') || uri.startsWith('\\\\')) {
    throw actionableError(`${description} uses a remote or absolute URI. Only local files in the selected mesh folder are supported.`)
  }
  if (uri.includes('?') || uri.includes('#')) {
    throw actionableError(`${description} uses a query or fragment in its resource URI. Select a glTF with plain relative file paths.`)
  }

  let decoded
  try {
    decoded = decodeURIComponent(uri)
  } catch {
    throw actionableError(`${description} has an invalid percent-encoded resource URI.`)
  }
  if (!decoded || decoded.startsWith('/') || decoded.startsWith('\\') || decoded.includes('\\') || decoded.includes('\0')) {
    throw actionableError(`${description} must reference a local relative file.`)
  }

  const segments = decoded.split('/')
  if (segments.some((segment) => segment === '..' || segment.includes(':') || segment === '')) {
    throw actionableError(`${description} contains traversal or an invalid path component. Keep resource paths relative to the selected glTF folder.`)
  }
  const relativePath = path.join(...segments.filter((segment) => segment !== '.'))
  if (!relativePath || path.isAbsolute(relativePath)) {
    throw actionableError(`${description} must reference a local relative file.`)
  }
  return relativePath
}

function getExternalGlbDependencies(document) {
  if (!document || typeof document !== 'object' || Array.isArray(document)) {
    throw actionableError('the glTF document is not a JSON object.')
  }
  const dependencies = new Map()
  let referenceCount = 0
  for (const [field, label] of [['buffers', 'buffer'], ['images', 'image']]) {
    const items = document[field] ?? []
    if (!Array.isArray(items)) throw actionableError(`the glTF ${field} field must be an array.`)
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index]
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        throw actionableError(`glTF ${label} ${index} metadata is malformed.`)
      }
      if (item.uri === undefined) continue
      const relativePath = decodeLocalResourceUri(item.uri, `glTF ${label} ${index}`)
      if (relativePath) {
        referenceCount += 1
        if (referenceCount > MAX_EXTERNAL_RESOURCES) {
          throw actionableError(`the glTF references more than ${MAX_EXTERNAL_RESOURCES} external resources. Reduce the resource count and try again.`)
        }
        dependencies.set(relativePath, { relativePath, label: `glTF ${label} ${index}` })
      }
    }
  }
  return [...dependencies.values()]
}

async function inspectLocalDependency(sourceDir, relativePath, label) {
  const segments = relativePath.split(path.sep)
  let candidate = sourceDir
  for (let index = 0; index < segments.length; index += 1) {
    candidate = path.join(candidate, segments[index])
    let stat
    try {
      stat = await lstat(candidate)
    } catch {
      throw actionableError(`${label} is missing (${relativePath}). Keep every referenced buffer and image beside the selected glTF and try again.`)
    }
    if (stat.isSymbolicLink()) {
      throw actionableError(`${label} uses a symbolic link (${relativePath}). Replace it with a regular file inside the selected glTF folder.`)
    }
    if (index < segments.length - 1 && !stat.isDirectory()) {
      throw actionableError(`${label} path contains a non-directory component (${relativePath}).`)
    }
    if (index === segments.length - 1 && !stat.isFile()) {
      throw actionableError(`${label} is not a regular file (${relativePath}).`)
    }
  }
  return { sourcePath: candidate, size: (await lstat(candidate)).size }
}

async function copyOpenFile(input, destinationPath, initialSnapshot, afterFirstChunk) {
  const output = await open(destinationPath, fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL | (fsConstants.O_NOFOLLOW ?? 0), 0o600)
  try {
    const buffer = Buffer.allocUnsafe(1024 * 1024)
    let position = 0
    let firstChunk = true
    while (true) {
      const { bytesRead } = await input.read(buffer, 0, buffer.length, position)
      if (bytesRead === 0) break
      let written = 0
      while (written < bytesRead) {
        const result = await output.write(buffer, written, bytesRead - written, position + written)
        if (result.bytesWritten <= 0) throw new Error('mesh staging write made no progress')
        written += result.bytesWritten
      }
      position += bytesRead
      if (firstChunk) {
        firstChunk = false
        await afterFirstChunk?.()
      }
    }
    await output.sync()
    const copiedSnapshot = await input.stat({ bigint: true })
    if (sourceSnapshotChanged(initialSnapshot, copiedSnapshot)) {
      throw actionableError('the selected mesh or one of its glTF resources changed while it was being copied. Select stable files and try again.')
    }
    await output.close()
  } catch (error) {
    await output.close().catch(() => undefined)
    await rm(destinationPath, { force: true }).catch(() => undefined)
    throw error
  }
}

async function copyStableFile(sourcePath, destinationPath, afterFirstChunk) {
  let input
  try {
    input = await open(sourcePath, fsConstants.O_RDONLY | (fsConstants.O_NOFOLLOW ?? 0))
    const initial = await input.stat({ bigint: true })
    if (!initial.isFile()) throw actionableError('a selected mesh resource is not a regular file.')
    const pathInfo = await lstat(sourcePath)
    if (pathInfo.isSymbolicLink() || (pathInfo.ino !== 0 && initial.ino !== 0n
      && (BigInt(pathInfo.dev) !== initial.dev || BigInt(pathInfo.ino) !== initial.ino))) {
      throw actionableError('a mesh resource changed or became a symbolic link while it was being opened. Select the files again.')
    }
    await copyOpenFile(input, destinationPath, initial, afterFirstChunk)
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('Could not stage mesh:')) throw error
    throw actionableError('a mesh file could not be copied. Check its permissions and available workspace space.')
  } finally {
    await input?.close().catch(() => undefined)
  }
}

async function ensureWorkspaceMeshDir(workspaceDir) {
  let root
  try {
    await mkdir(workspaceDir, { recursive: true })
    root = await realpath(workspaceDir)
  } catch {
    throw actionableError('the workspace folder is unavailable. Check Modly workspace settings and try again.')
  }
  const importsDir = path.join(root, 'imports')
  const meshesDir = path.join(importsDir, 'meshes')
  try {
    for (const directory of [importsDir, meshesDir]) {
      await mkdir(directory).catch((error) => {
        if (error?.code !== 'EEXIST') throw error
      })
      const stat = await lstat(directory)
      if (stat.isSymbolicLink() || !stat.isDirectory() || await realpath(directory) !== directory) {
        throw new Error('unsafe workspace import directory')
      }
    }
  } catch {
    throw actionableError('the workspace imports folder is unsafe or unwritable. Remove any symlink at workspace/imports or workspace/imports/meshes, then try again.')
  }
  return meshesDir
}

async function readGltfDependencies(sourcePath, openedSnapshot, sourceHandle) {
  if (openedSnapshot.size > BigInt(MAX_GLTF_DOCUMENT_BYTES)) {
    throw actionableError(`the glTF document exceeds ${MAX_GLTF_DOCUMENT_BYTES / (1024 * 1024)} MiB. Reduce its size and try again.`)
  }
  let document
  try {
    document = JSON.parse(await sourceHandle.readFile({ encoding: 'utf8' }))
  } catch {
    throw actionableError('the selected .gltf file is not readable JSON. Validate it and try again.')
  }
  const dependencies = getExternalGlbDependencies(document)
  const sourceDir = path.dirname(sourcePath)
  const resolved = []
  let totalBytes = Number(openedSnapshot.size)
  for (const dependency of dependencies) {
    const candidate = path.resolve(sourceDir, dependency.relativePath)
    assertContained(sourceDir, candidate)
    const inspected = await inspectLocalDependency(sourceDir, dependency.relativePath, dependency.label)
    totalBytes += inspected.size
    if (!Number.isSafeInteger(totalBytes) || totalBytes > MAX_TOTAL_PACKAGE_BYTES) {
      throw actionableError(`the glTF and its referenced files exceed ${MAX_TOTAL_PACKAGE_BYTES / (1024 * 1024)} MiB. Reduce the package size and try again.`)
    }
    resolved.push({ ...dependency, ...inspected })
  }
  return resolved
}

async function createUniquePackageDir(meshesDir, stem) {
  for (let index = 0; index < 10_000; index += 1) {
    const packageName = `${stem}${index === 0 ? '' : `-${index}`}`
    const packagePath = path.join(meshesDir, packageName)
    try {
      await mkdir(packagePath, { mode: 0o700 })
      if (await realpath(packagePath) !== packagePath) throw new Error('unsafe package directory')
      return { packageName, packagePath }
    } catch (error) {
      if (error?.code === 'EEXIST') continue
      await rm(packagePath, { recursive: true, force: true }).catch(() => undefined)
      throw actionableError('the workspace mesh package could not be created. Check workspace permissions and try again.')
    }
  }
  throw actionableError('too many staged packages share this name. Rename the source file or remove an older staged copy.')
}

/**
 * Copy a selected mesh into Modly's workspace. Standalone glTF files are
 * packaged with only their referenced local buffers and images so the JSON's
 * relative URIs keep resolving. GLB remains a single staged file.
 */
export async function stageWorkspaceMeshFile(sourcePath, workspaceDir, testHooks = {}) {
  if (typeof sourcePath !== 'string' || !sourcePath.trim()) {
    throw actionableError('select a GLB or glTF file and try again.')
  }
  const source = path.resolve(sourcePath)
  const extension = path.extname(source).toLowerCase()
  if (!ALLOWED_MESH_EXTENSIONS.has(extension)) {
    throw actionableError('only .glb and .gltf files can be imported by this workflow.')
  }

  let sourceInfo
  try {
    sourceInfo = await lstat(source)
  } catch {
    throw actionableError('the selected file is no longer available. Select it again.')
  }
  if (sourceInfo.isSymbolicLink() || !sourceInfo.isFile()) {
    throw actionableError('the selected path must be a regular file, not a symbolic link or directory.')
  }

  let sourceHandle
  let sourceSnapshot
  try {
    sourceHandle = await open(source, fsConstants.O_RDONLY | (fsConstants.O_NOFOLLOW ?? 0))
    const opened = await sourceHandle.stat()
    if (!opened.isFile() || (sourceInfo.ino !== 0 && opened.ino !== 0 && (sourceInfo.dev !== opened.dev || sourceInfo.ino !== opened.ino))) {
      throw actionableError('the selected path changed while it was being opened. Select the file again.')
    }
    sourceSnapshot = await sourceHandle.stat({ bigint: true })
  } catch (error) {
    await sourceHandle?.close().catch(() => undefined)
    if (error instanceof Error && error.message.startsWith('Could not stage mesh:')) throw error
    throw actionableError('the selected file could not be opened. Check its permissions and select it again.')
  }

  let packageDir
  try {
    const stem = path.basename(source, path.extname(source)).replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').trim() || 'mesh'
    const meshesDir = await ensureWorkspaceMeshDir(workspaceDir)
    if (await realpath(meshesDir) !== meshesDir) {
      throw actionableError('the workspace mesh folder changed while importing. Remove any symlink at workspace/imports/meshes and try again.')
    }

    if (extension === '.gltf') {
      const dependencies = await readGltfDependencies(source, sourceSnapshot, sourceHandle)
      const currentSnapshot = await sourceHandle.stat({ bigint: true })
      if (sourceSnapshotChanged(sourceSnapshot, currentSnapshot)) {
        throw actionableError('the selected glTF changed while it was being read. Select a stable file and try again.')
      }

      packageDir = await createUniquePackageDir(meshesDir, stem)
      const fileName = `${stem}.gltf`
      const filePath = path.join(packageDir.packagePath, fileName)
      await copyOpenFile(sourceHandle, filePath, sourceSnapshot, testHooks.afterFirstChunk)
      for (const dependency of dependencies) {
        const targetPath = path.resolve(packageDir.packagePath, dependency.relativePath)
        const relativeTarget = path.relative(packageDir.packagePath, targetPath)
        if (!relativeTarget || relativeTarget === '..' || relativeTarget.startsWith(`..${path.sep}`) || path.isAbsolute(relativeTarget)) {
          throw actionableError(`${dependency.label} resolves outside the staged package. Select a glTF with safe relative paths.`)
        }
        const parent = path.dirname(targetPath)
        await mkdir(parent, { recursive: true })
        await inspectLocalDependency(path.dirname(source), dependency.relativePath, dependency.label)
        await copyStableFile(dependency.sourcePath, targetPath)
      }
      await sourceHandle.close()
      sourceHandle = null
      return { filePath, fileName }
    }

    for (let index = 0; index < 10_000; index += 1) {
      const fileName = `${stem}${index === 0 ? '' : `-${index}`}${extension}`
      const filePath = path.join(meshesDir, fileName)
      try {
        await copyOpenFile(sourceHandle, filePath, sourceSnapshot, testHooks.afterFirstChunk)
        await sourceHandle.close()
        sourceHandle = null
        return { filePath, fileName }
      } catch (error) {
        if (error?.code === 'EEXIST') continue
        throw error
      }
    }
    throw actionableError('too many files with this name already exist in workspace/imports/meshes. Rename the source file or remove an older staged copy.')
  } catch (error) {
    if (packageDir) await rm(packageDir.packagePath, { recursive: true, force: true }).catch(() => undefined)
    if (error instanceof Error && error.message.startsWith('Could not stage mesh:')) throw error
    throw actionableError('the file could not be copied into workspace/imports/meshes. Check that the workspace has free space and write permission.')
  } finally {
    await sourceHandle?.close().catch(() => undefined)
  }
}
