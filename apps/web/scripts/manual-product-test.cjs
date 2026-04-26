const path = require("path");
const { chromium } = require("@playwright/test");

const ROOT = path.resolve(__dirname, "../../..");
const SAMPLE_DIR = path.join(ROOT, "sample-data", "manual-test-suite");
const APP_URL = process.env.APRAG_APP_URL || "http://localhost:5173";
const API_URL = process.env.APRAG_API_URL || "http://localhost:8000";

const sampleFiles = [
  "01_vaccine_administration_event.jpg",
  "02_lakeside_dispatch_memo.wav",
  "03_service_tickets.csv",
  "04_incident_review.pdf",
].map((name) => path.join(SAMPLE_DIR, name));

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function api(pathname, options = {}) {
  const response = await fetch(`${API_URL}${pathname}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${pathname} failed with ${response.status}: ${text}`);
  }
  return response.json();
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(label, timeoutMs, intervalMs, callback) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    last = await callback();
    if (last.done) {
      return last.value;
    }
    await sleep(intervalMs);
  }
  throw new Error(`${label} timed out. Last state: ${JSON.stringify(last)}`);
}

async function latestProjectByName(name) {
  const projects = await api("/api/projects");
  return projects.find((project) => project.name === name);
}

async function latestReadyPlaywrightProject() {
  const projects = await api("/api/projects");
  for (const project of projects.filter((item) => item.name.startsWith("Playwright Full Product"))) {
    const sources = await api(`/api/projects/${project.id}/sources`);
    if (sources.length === sampleFiles.length && sources.every((source) => source.status === "processed")) {
      return { project, sources };
    }
  }
  return null;
}

async function run() {
  const headed = process.env.APRAG_PLAYWRIGHT_HEADLESS === "0";
  const browser = await chromium.launch({ headless: !headed, slowMo: headed ? 75 : 0 });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const consoleMessages = [];
  const pageErrors = [];
  const failedRequests = [];

  page.on("console", (message) => {
    consoleMessages.push({ type: message.type(), text: message.text() });
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });
  page.on("requestfailed", (request) => {
    failedRequests.push({ url: request.url(), failure: request.failure()?.errorText });
  });

  try {
    const health = await api("/health");
    assert(health.status === "ok", "API health is not ok");
    const providerHealth = await api("/api/settings/provider-health");
    assert(providerHealth.ready === true, "Provider health is not ready");
    assert(providerHealth.required_missing.length === 0, "Required provider models are missing");

    await page.goto(APP_URL, { waitUntil: "networkidle" });
    await page.getByRole("heading", { name: "APRAG-Lab" }).waitFor({ timeout: 30000 });
    await page.getByText("real mode").waitFor({ timeout: 30000 });
    await page.getByText("qwen3:8b:installed").waitFor({ timeout: 30000 });
    await page.getByText("qwen3-vl:4b:installed").waitFor({ timeout: 30000 });
    await page.getByText("bge-m3:installed").waitFor({ timeout: 30000 });

    let project;
    let sources;
    const reusable = process.env.APRAG_REUSE_LATEST === "1" ? await latestReadyPlaywrightProject() : null;
    if (reusable) {
      project = reusable.project;
      sources = reusable.sources;
      console.log("reusing processed project", project.id);
    } else {
      const projectName = `Playwright Full Product ${Date.now()}`;
      await page.locator("#source-files").setInputFiles(sampleFiles);
      await page.locator("#upload-action").selectOption("create_new");
      await page.locator("#new-project-name").fill(projectName);

      const uploadResponsePromise = page.waitForResponse(
        (response) => response.url().includes("/sources") && response.request().method() === "POST",
        { timeout: 120000 },
      );
      await page.getByRole("button", { name: /upload/i }).click();
      const uploadResponse = await uploadResponsePromise;
      assert(uploadResponse.ok(), `Upload failed with status ${uploadResponse.status()}`);

      project = await waitFor("project creation", 60000, 2000, async () => {
        const found = await latestProjectByName(projectName);
        return { done: Boolean(found), value: found };
      });

      sources = await waitFor("source processing", 20 * 60 * 1000, 5000, async () => {
        const rows = await api(`/api/projects/${project.id}/sources`);
        const counts = rows.reduce((acc, row) => {
          acc[row.status] = (acc[row.status] || 0) + 1;
          return acc;
        }, {});
        console.log("source status", counts);
        return {
          done: rows.length === sampleFiles.length && rows.every((row) => !["queued", "processing"].includes(row.status)),
          value: rows,
        };
      });
    }

    const failedSources = sources.filter((source) => source.status !== "processed");
    assert(failedSources.length === 0, `Some sources failed: ${JSON.stringify(failedSources, null, 2)}`);

    await page.getByRole("button", { name: /refresh/i }).click();
    await page.getByRole("heading", { name: "Project Sources" }).waitFor({ timeout: 30000 });
    await page.getByText(String(sampleFiles.length)).first().waitFor({ timeout: 30000 });

    await page.getByRole("button", { name: "View" }).first().click();
    await page.getByText("Extracted Evidence Blocks").waitFor({ timeout: 30000 });

    const question = [
      "What caused the Lakeside Clinic outage?",
      "What did the audio memo say Priya Shah did?",
      "Why did Harbor Market not qualify for an SLA credit?",
      "What does the image show?",
      "Answer with citations.",
    ].join(" ");

    await page.locator("#question").fill(question);
    let completedRun;
    if (process.env.APRAG_REUSE_LATEST_RUN === "1") {
      const runs = await api(`/api/projects/${project.id}/runs`);
      assert(runs.length > 0, "No existing runs found to reuse");
      completedRun = await api(`/api/runs/${runs[0].id}`);
      console.log("reusing completed run", completedRun.id, completedRun.job_status);
    } else {
      const runResponsePromise = page.waitForResponse(
        (response) => response.url().includes("/runs") && response.request().method() === "POST",
        { timeout: 120000 },
      );
      await page.getByRole("button", { name: /run all rag flows/i }).click();
      const runResponse = await runResponsePromise;
      assert(runResponse.ok(), `Run creation failed with status ${runResponse.status()}`);
      const queuedRun = await runResponse.json();

      completedRun = await waitFor("benchmark run", 20 * 60 * 1000, 5000, async () => {
        const runState = await api(`/api/runs/${queuedRun.id}`);
        console.log("run status", runState.job_status, runState.results?.length || 0);
        return {
          done: !["queued", "running"].includes(runState.job_status) && (runState.results || []).length >= 3,
          value: runState,
        };
      });
    }

    assert(completedRun.results.length === 3, "Expected three pipeline results");
    assert(completedRun.comparison, "Expected comparison payload");
    assert(completedRun.recommendation, "Expected recommendation payload");

    const combinedAnswers = completedRun.results.map((result) => result.answer).join("\n").toLowerCase();
    for (const expected of ["lakeside", "priya", "harbor", "sla", "vaccine"]) {
      assert(combinedAnswers.includes(expected), `Expected answer to mention ${expected}`);
    }
    assert(
      combinedAnswers.includes("184000") || combinedAnswers.includes("47") || combinedAnswers.includes("p1"),
      "Expected answer to mention the ticket metric evidence",
    );
    for (const result of completedRun.results) {
      assert(result.citations.length > 0, `${result.pipeline_type} produced no citations`);
      assert(!result.warnings.includes("pipeline_failed"), `${result.pipeline_type} failed: ${result.warnings.join(", ")}`);
    }

    await page.getByRole("button", { name: /refresh/i }).click();
    await page.locator(".history-row").filter({ hasText: question.slice(0, 30) }).first().click();
    await page.getByRole("tab", { name: /^Comparison/ }).click();
    await page.getByRole("heading", { name: "Recommendation" }).waitFor({ timeout: 30000 });
    await page.getByRole("tab", { name: /^Traditional/ }).click();
    await page.getByRole("heading", { name: "Answer" }).waitFor({ timeout: 30000 });
    await page.getByRole("tab", { name: /^Agentic/ }).click();
    await page.getByRole("heading", { name: "Trace Viewer" }).waitFor({ timeout: 30000 });
    await page.getByRole("tab", { name: /^Hybrid Graph/ }).click();
    await page.getByRole("heading", { name: "Citations" }).waitFor({ timeout: 30000 });

    const exportJsonResponse = await fetch(`${API_URL}/api/runs/${completedRun.id}/export.json`);
    assert(exportJsonResponse.ok, "JSON export failed");
    const exportMarkdownResponse = await fetch(`${API_URL}/api/runs/${completedRun.id}/export.md`);
    assert(exportMarkdownResponse.ok, "Markdown export failed");

    const logs = await api("/api/diagnostics/logs?limit=200");
    assert(Array.isArray(logs.entries), "Diagnostics logs did not return entries");

    const badConsole = consoleMessages.filter((message) => ["error"].includes(message.type) && !message.text.includes("favicon"));
    assert(pageErrors.length === 0, `Page errors: ${pageErrors.join("\n")}`);
    assert(failedRequests.length === 0, `Failed browser requests: ${JSON.stringify(failedRequests, null, 2)}`);
    assert(badConsole.length === 0, `Console errors: ${JSON.stringify(badConsole, null, 2)}`);

    console.log(
      JSON.stringify(
        {
          status: "passed",
          project_id: project.id,
          run_id: completedRun.id,
          sources: sources.length,
          recommendation: completedRun.recommendation.recommended_flow,
          best_pipeline: completedRun.comparison.best_pipeline,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
