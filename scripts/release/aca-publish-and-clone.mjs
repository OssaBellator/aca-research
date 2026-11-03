// Explicitly invoked by ACA's fleet setup task. Publishes ONLY this audited project
// to the authenticated personal GitHub account, then clones it into PC's Code folder.
// Idempotent: refuses an unrelated existing repository or existing unrelated checkout.
// This code intentionally never reads/stores GH tokens or changes global auth settings.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const repo = 'OssaBellator/aca-research';
const url = 'https://github.com/' + repo + '.git';
const desc = 'Adaptive Cognitive Architecture research: resource-constrained latent reasoning, routed memory, and falsifiable tests';
const root = process.cwd();
const target = path.join(os.homedir(), 'Downloads', 'Code', 'aca-research');
const stages = [];

function run(executable, args, cwd = root, timeout = 90000, allowError = false) {
  const p = spawnSync(executable, args, {
    cwd, encoding: 'utf8', windowsHide: true, timeout, maxBuffer: 400_000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GH_PROMPT_DISABLED: '1' }
  });
  const result = { program: executable, exitCode: p.status, stderr: (p.stderr ?? '').trim().slice(0, 1800) };
  if (p.error || p.status !== 0) {
    if (allowError) return { ok: false, stdout: (p.stdout ?? '').trim(), ...result };
    throw new Error(JSON.stringify({ phase: executable + ' ' + args.slice(0,2).join(' '), ...result, error: p.error?.message }));
  }
  stages.push(executable + ' ' + args.slice(0,2).join(' '));
  return { ok: true, stdout: (p.stdout ?? '').trim(), ...result };
}

try {
  if (!process.argv.includes('--run')) throw new Error('Explicit --run flag required');
  for (const file of ['README.md','LICENSE','.gitignore','docs/hypotheses.md','experiments/hypothesis-index.csv']) {
    if (!fs.existsSync(path.join(root, file))) throw new Error('Missing project source file: ' + file);
  }
  const active = run('gh', ['api', 'user', '--jq', '.login'], root, 15000).stdout.trim();
  if (active.toLowerCase() !== 'ossabellator') throw new Error('GitHub account mismatch: refusing public publication.');

  const current = run('gh', ['repo', 'view', repo, '--json', 'name,description,visibility', '--jq', '.'], root, 20000, true);
  let exists = current.ok;
  if (exists) {
    const data = JSON.parse(current.stdout);
    if (data.visibility?.toLowerCase() !== 'public' ||
      !String(data.description ?? '').toLowerCase().includes('adaptive cognitive architecture')) {
      throw new Error('An existing repository has unexpected visibility or description; refusing changes');
    }
  }

  // This workspace snapshot is isolated from the source project's original .git history.
  // A single reproducible snapshot commit is therefore used for initial public import.
  if (!fs.existsSync(path.join(root,'.git'))) {
    run('git', ['init', '-b', 'main']);
    run('git', ['add', '.gitignore', 'LICENSE', 'README.md', 'docs', 'experiments', 'results/README.md', 'scripts']);
    run('git', ['-c', 'user.name=ACA Research', '-c',
      'user.email=141500316+OssaBellator@users.noreply.github.com',
      'commit', '-m', 'Initial public ACA research snapshot']);
  }
  if (!exists) {
    run('gh', ['repo', 'create', repo, '--public', '--description', desc], root, 30000);
    exists = true;
  }
  const origin = run('git', ['remote', 'get-url', 'origin'], root, 10000, true);
  if (!origin.ok) run('git', ['remote', 'add', 'origin', url]);
  else if (origin.stdout.toLowerCase() !== url.toLowerCase()) throw new Error('Unexpected origin remote; refusing push');

  const remoteHeads = run('git', ['ls-remote', '--heads', url, 'main'], root, 30000);
  if (!remoteHeads.stdout.trim()) {
    run('git', ['-c', 'credential.helper=!gh auth git-credential', 'push', '-u', 'origin', 'main'], root, 120000);
  } else {
    stages.push('remote already populated; no push/overwrite attempted');
  }

  if (fs.existsSync(target)) {
    const check = run('git', ['-C', target, 'remote', 'get-url', 'origin'], root, 15000, true);
    if (!check.ok || check.stdout.toLowerCase() !== url.toLowerCase())
      throw new Error('Existing Code/aca-research is unrelated; refusing overwrite');
    run('git', ['-C', target, 'fetch', '--ff-only', 'origin'], root, 30000, true);
    stages.push('existing ACA checkout retained');
  } else {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    run('git', ['clone', '--depth', '1', url, target], root, 120000);
  }

  const localHead = run('git', ['-C', target, 'rev-parse', 'HEAD'], root, 10000).stdout.trim();
  const result = {
    ok: true, operation: 'publish-and-clone', repoUrl: 'https://github.com/' + repo,
    visibility: 'public', checkout: target, head: localHead, stages
  };
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  console.error(JSON.stringify({ ok: false, operation: 'publish-and-clone', stages, error: String(error.message ?? error) }));
  process.exitCode = 1;
}
