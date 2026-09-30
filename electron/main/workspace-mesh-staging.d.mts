export function stageWorkspaceMeshFile(
  sourcePath: string,
  workspaceDir: string,
  testHooks?: { afterFirstChunk?: () => void | Promise<void> },
): Promise<{ filePath: string; fileName: string }>
