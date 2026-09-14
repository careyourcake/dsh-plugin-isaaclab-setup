import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { BUNDLED_SKILL_RANK } from "@deepseek-ai/dsh-skill";

/**
 * dsh-plugin-isaaclab-setup
 *
 * A DeepSeek Harness plugin that bundles one skill behind one skill provider:
 *
 *   - isaaclab-setup: install and run Isaac Sim + Isaac Lab on a Linux/NVIDIA
 *     server, covering version pinning (isaaclab 0.47.2, rsl-rl-lib 3.0.1,
 *     CUDA sm_89 vs sm_120), the AppLauncher-first import pattern, vectorized
 *     DirectRLEnv, headless rendering, and remote-training ops.
 *
 * Follows the same borrowed-provider pattern as @deepseek-ai/dsh-skill-badge
 * and dsh-plugin-litsearch-zotero: one provider registered on ctx.skills,
 * reading skill bodies from the package's own `skills/` directory at load time.
 */

const PROVIDER_NAME = "isaaclab-setup";

const RESOURCE_BASE = {
  kind: "directory",
  path: fileURLToPath(new URL("../skills/", import.meta.url)),
};

const SKILL_DEFS = [
  {
    name: "isaaclab-setup",
    description:
      "Set up and run Isaac Sim + Isaac Lab on a Linux/NVIDIA server: version pinning (isaaclab 0.47.2, isaaclab_rl 0.4.4, rsl-rl-lib 3.0.1, CUDA sm_89 vs sm_120 Blackwell), the AppLauncher-before-import pattern, vectorized DirectRLEnv (regex prim path), headless rendering via the Camera sensor, and remote-training SSH ops. Use when installing Isaac Lab, writing a robot RL env, debugging Isaac Sim import/version/EULA errors, or training PPO/DreamerV3 on a server.",
    whenToUse:
      "When the user asks to install or set up Isaac Lab / Isaac Sim, write or debug a DirectRLEnv robot simulation, train PPO or a world model (DreamerV3) on a robot task, fix Isaac Sim version/EULA/CUDA/import errors, render a scene headlessly, or run long RL training on a remote GPU server.",
    file: "isaaclab-setup.md",
  },
];

const CANDIDATES = SKILL_DEFS.map((def) => ({
  name: def.name,
  description: def.description,
  whenToUse: def.whenToUse,
  invocation: { modelInvocable: true, userInvocable: true },
  provider: PROVIDER_NAME,
  source: "bundled",
  resourceBase: RESOURCE_BASE,
  rank: BUNDLED_SKILL_RANK,
  locator: new URL(`../skills/${def.file}`, import.meta.url),
}));

const provider = {
  name: PROVIDER_NAME,
  list: () => Promise.resolve(CANDIDATES),
  async get(candidate) {
    const def = SKILL_DEFS.find((entry) => entry.name === candidate.name);
    if (def === undefined) return undefined;
    return {
      name: candidate.name,
      description: candidate.description,
      whenToUse: candidate.whenToUse,
      invocation: candidate.invocation,
      provider: PROVIDER_NAME,
      source: "bundled",
      resourceBase: RESOURCE_BASE,
      content: await readFile(candidate.locator, "utf8"),
    };
  },
};

/** Cordis plugin name. */
const name = "isaaclab-setup";

/** The skill registry service required by this plugin. */
const inject = ["skills"];

/**
 * Register the bundled skill provider on ctx.skills.
 *
 * `registerProvider` returns the provider; the registry owns its lifecycle and
 * unregisters it when this plugin fiber is disposed.
 */
function apply(ctx) {
  ctx.skills.registerProvider(() => provider);
}

export { apply, inject, name };
