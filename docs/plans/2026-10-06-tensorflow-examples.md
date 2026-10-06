# TensorFlow.js Examples Refresh Implementation Plan
> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
**Goal:** Add the uploaded Ollama/TensorFlow.js embedding comparison script, improve the existing examples, and provide accurate setup and usage documentation.
**Architecture:** Keep each runnable learning example in its own root JavaScript file. Add a small npm manifest for explicit dependencies/scripts, make all long-lived TensorFlow tensors disposable, and document the optional Ollama and CSV prerequisites separately.
**Tech Stack:** Node.js 18+, CommonJS, TensorFlow.js, TensorFlow.js WASM backend, TensorFlow.js Node native backend, Ollama REST API, Node built-in test runner.
---

## Task 1: Project setup and uploaded embedding example
**Files:**
- Create: `package.json`
- Create: `index.js`
- Test: `tests/index.test.js`

- Add explicit dependencies and npm scripts.
- Organize the uploaded script into exported, testable functions and a guarded CLI entry point.
- Validate inputs, HTTP errors, Ollama response dimensions, and zero-length vectors; return cosine similarity and report angle cleanly.
- Test cosine similarity, invalid vectors, and angle calculation without requiring a live Ollama server.

## Task 2: Improve existing runnable examples
**Files:**
- Modify: `tf.js`
- Modify: `model.js`
- Modify: `house-price-predction.js`

- Remove duplicated/broken experiments and organize each script into named functions.
- Preserve each example's learning purpose and command-line behavior where practical.
- Dispose intermediate tensors/models, validate dataset/training inputs, and handle constant-value normalization safely.
- Keep the house-price CSV and TensorFlow Node backend requirements explicit.

## Task 3: Documentation and verification
**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-10-06-tensorflow-examples.md`

- Rewrite the README in Bengali with prerequisites, installation, commands, examples, environment variables, and common errors.
- Run syntax checks, built-in tests, and safe offline smoke checks; record any checks that need Ollama/network access.
- Review the diff for unintended changes and secrets, commit in reviewable increments, push the feature branch, and open a PR without merging it.


### Verified implementation notes

- The uploaded file is the Ollama embedding comparison utility and is added as root `index.js`.
- The old housing CSV endpoint returned HTTP 404. The current dataset path is `datasets/regression/housing_market_data.csv`, with headers `Size_sqft,Bedrooms,Year_Built,Price`; the house-price demo now targets this schema.
- Delivery is limited to a pushed feature branch and a reviewable Pull Request; `main` will not be changed directly and the PR will not be merged automatically.


## Completion and verification

- `npm run check`: passed.
- `npm test`: all 13 tests passed, including mocked Ollama flow, CSV parsing, and both regression equations.
- CLI smoke checks: `tf.js`, `model.js`, and `house-price-predction.js` ran successfully; the corrected housing CSV URL returned the documented headers.
- Live Ollama integration could not be exercised because no Ollama server is running in this environment; the API flow is covered with a mocked fetch response.
- `npm audit` reports three moderate findings from one transitive `sprintf-js` advisory below TensorFlow.js → `argparse`. The available forced fix downgrades TensorFlow.js to 2.1.0 (breaking); it was not applied to avoid an unsafe major downgrade.
