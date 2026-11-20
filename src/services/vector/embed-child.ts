import { LocalEmbedder } from './LocalEmbedder.js';
import { resolveEmbeddingModel } from './models.js';
import { runEmbeddingProcess } from './embed-protocol.js';

/**
 * The process a ChildProcessEmbedder starts: `embed-child <model name> <threads>`
 * (threads 0 means all). It loads the model on its first request and answers
 * until its stdin closes, which happens when the worker stops it or dies.
 */
async function main(): Promise<void> {
  const [modelName, threadsArg] = process.argv.slice(2);
  const threads = Number(threadsArg);
  const embedder = new LocalEmbedder(resolveEmbeddingModel(modelName), threads > 0 ? threads : undefined);
  await runEmbeddingProcess(embedder);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
