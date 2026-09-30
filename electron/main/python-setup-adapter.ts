import { createHash } from 'crypto'
import { readdirSync, readFileSync, lstatSync } from 'fs'
import { dirname, join, relative, resolve } from 'path'

const EXECUTABLE_ADAPTER_SUFFIXES = new Set(['.py', '.pyw', '.so', '.pyd', '.dll', '.dylib'])

/** The Python distribution root is always the API directory beside requirements.txt. */
export function getSemanticAdapterDistributionPath(requirementsPath: string): string {
  return resolve(dirname(requirementsPath))
}

export function semanticAdapterInstallArgs(apiDir: string): string[] {
  return ['-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--disable-pip-version-check', resolve(apiDir)]
}

/** Hash source inputs that determine the local wheel installed into Modly's venv. */
export function hashSemanticAdapterSources(apiDir: string): string {
  const root = resolve(apiDir)
  const packageDir = join(root, 'runtime', 'adapters', 'parts')
  const files: string[] = [join(root, 'pyproject.toml')]
  const visit = (directory: string): void => {
    for (const name of readdirSync(directory).sort()) {
      const path = join(directory, name)
      const stat = lstatSync(path)
      if (stat.isSymbolicLink()) throw new Error(`semantic adapter source cannot be a symlink: ${path}`)
      if (stat.isDirectory()) visit(path)
      else if (stat.isFile() && EXECUTABLE_ADAPTER_SUFFIXES.has(path.slice(path.lastIndexOf('.')))) files.push(path)
      else if (!stat.isFile()) throw new Error(`semantic adapter source contains a special file: ${path}`)
    }
  }
  visit(packageDir)
  const digest = createHash('sha256')
  for (const path of files.sort()) {
    digest.update(relative(root, path).replaceAll('\\', '/'))
    digest.update('\0')
    digest.update(readFileSync(path))
    digest.update('\0')
  }
  return digest.digest('hex')
}
