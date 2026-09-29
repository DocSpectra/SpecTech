import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const GEMMA_DIR = path.join(ROOT, "outputs", "round2", "human_first_reranking");
const GPT_DIR = path.join(ROOT, "outputs", "round2", "dgx_gptoss120b_human_first");
const OUT = path.join(ROOT, "analysis", "round2_candidate_packet_novelty", "novelty_summary.json");

const files = {
  gemmaWorkbook: path.join(GEMMA_DIR, "human_first_review_packet.xlsx"),
  gemmaKey: path.join(GEMMA_DIR, "private_review_key.csv"),
  gptWorkbook: path.join(GPT_DIR, "dgx_gptoss120b_review_packet.xlsx"),
  gptKey: path.join(GPT_DIR, "private_review_key.csv"),
};

const expectedHashes = {
  gemmaWorkbook: "07d294645a7fbdb4a10c74cb4ce8b959d1db97c7cc8979922435a9dc159d82d9",
  gptWorkbook: "00fd77981f86d991c5a516072577c6532f26812c0ca72f4150f6432e7d43ba9c",
};

function sha256(data) {
  return crypto.createHash("sha256").update(data).digest("hex");
}

function normalize(text) {
  return String(text).normalize("NFKC").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
}

function tokens(text) {
  return normalize(text).match(/[\p{L}\p{M}\p{N}]+|[^\s]/gu) ?? [];
}

function levenshtein(a, b) {
  if (a.length > b.length) [a, b] = [b, a];
  let prior = Array.from({ length: a.length + 1 }, (_, i) => i);
  for (let j = 1; j <= b.length; j += 1) {
    const current = [j];
    for (let i = 1; i <= a.length; i += 1) {
      current[i] = Math.min(
        current[i - 1] + 1,
        prior[i] + 1,
        prior[i - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
    prior = current;
  }
  return prior[a.length];
}

function similarity(a, b) {
  const longest = Math.max(a.length, b.length);
  return longest === 0 ? 1 : 1 - levenshtein(a, b) / longest;
}

function quantile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const index = (sorted.length - 1) * p;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
}

function summarize(values) {
  return {
    n: values.length,
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    median: quantile(values, 0.5),
    q25: quantile(values, 0.25),
    q75: quantile(values, 0.75),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/u, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/u, ""));
    rows.push(row);
  }
  const header = rows.shift();
  header[0] = header[0].replace(/^\uFEFF/u, "");
  return rows.filter((r) => r.length === header.length).map((r) =>
    Object.fromEntries(header.map((key, index) => [key, r[index]])),
  );
}

async function loadCsv(file) {
  return parseCsv(await fs.readFile(file, "utf8"));
}

async function verifyWorkbook(file, keyRows) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const sheet = workbook.worksheets.getItem("Review");
  const values = sheet.getRange("A1:D181").values;
  if (JSON.stringify(values[0]) !== JSON.stringify(["Review ID", "A", "B", "Score"])) {
    throw new Error(`Unexpected review header: ${file}`);
  }
  const keyByReview = new Map(keyRows.map((row) => [row.review_id, row]));
  for (const row of values.slice(1)) {
    const [reviewId, a, b, score] = row;
    const key = keyByReview.get(reviewId);
    if (!key || score !== null) throw new Error(`Workbook/key/blank-score mismatch: ${file}`);
    if (sha256(String(a)) !== key.sentence_a_sha256 || sha256(String(b)) !== key.sentence_b_sha256) {
      throw new Error(`Workbook text hash mismatch: ${file}`);
    }
  }
  return { rows: values.length - 1, blankScores: values.slice(1).filter((r) => r[3] === null).length };
}

function pairMetrics(a, b) {
  const an = normalize(a);
  const bn = normalize(b);
  const at = tokens(a);
  const bt = tokens(b);
  return {
    charSimilarity: similarity([...an], [...bn]),
    tokenSimilarity: similarity(at, bt),
  };
}

function sourceMetrics(candidate, source) {
  const cn = normalize(candidate);
  const sn = normalize(source);
  const ct = tokens(candidate);
  const st = tokens(source);
  return {
    charEditRate: 1 - similarity([...cn], [...sn]),
    tokenEditRate: 1 - similarity(ct, st),
    charLengthRatio: cn.length / Math.max(1, sn.length),
    tokenLengthRatio: ct.length / Math.max(1, st.length),
  };
}

function groupBy(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    const value = row[key];
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value).push(row);
  }
  return groups;
}

function summarizeSourceMetrics(rows, sourceByCase) {
  const metrics = rows.map((row) => sourceMetrics(row.sentence_candidate, sourceByCase.get(row.case_id)));
  return Object.fromEntries(Object.keys(metrics[0]).map((key) => [key, summarize(metrics.map((m) => m[key]))]));
}

function withinSource(rows) {
  const char = [];
  const token = [];
  let exactDuplicatePairs = 0;
  let normalizedDuplicatePairs = 0;
  const uniqueCandidatesPerSource = { one: 0, two: 0, three: 0 };
  for (const candidates of groupBy(rows, "case_id").values()) {
    candidates.sort((a, b) => Number(a.candidate_slot) - Number(b.candidate_slot));
    const uniqueCount = new Set(candidates.map((row) => normalize(row.sentence_candidate))).size;
    uniqueCandidatesPerSource[["zero", "one", "two", "three"][uniqueCount]] += 1;
    for (let i = 0; i < candidates.length; i += 1) {
      for (let j = i + 1; j < candidates.length; j += 1) {
        const a = candidates[i].sentence_candidate;
        const b = candidates[j].sentence_candidate;
        exactDuplicatePairs += a === b ? 1 : 0;
        normalizedDuplicatePairs += normalize(a) === normalize(b) ? 1 : 0;
        const metric = pairMetrics(a, b);
        char.push(metric.charSimilarity);
        token.push(metric.tokenSimilarity);
      }
    }
  }
  return {
    pairCount: char.length,
    exactDuplicatePairs,
    normalizedDuplicatePairs,
    uniqueCandidatesPerSource,
    pairwiseCharSimilarity: summarize(char),
    pairwiseTokenSimilarity: summarize(token),
  };
}

function crossModel(gemmaRows, gptRows) {
  const gemmaByCase = groupBy(gemmaRows, "case_id");
  const gptByCase = groupBy(gptRows, "case_id");
  const sameSlotChar = [];
  const sameSlotToken = [];
  const bestChar = [];
  const bestToken = [];
  let sourceAlignedExactMatches = 0;
  let sourceAlignedNormalizedMatches = 0;
  for (const [caseId, gptCandidates] of gptByCase.entries()) {
    const gemmaCandidates = gemmaByCase.get(caseId);
    const gemmaBySlot = new Map(gemmaCandidates.map((r) => [r.candidate_slot, r]));
    for (const gpt of gptCandidates) {
      const same = pairMetrics(gpt.sentence_candidate, gemmaBySlot.get(gpt.candidate_slot).sentence_candidate);
      sameSlotChar.push(same.charSimilarity);
      sameSlotToken.push(same.tokenSimilarity);
      const all = gemmaCandidates.map((gemma) => pairMetrics(gpt.sentence_candidate, gemma.sentence_candidate));
      bestChar.push(Math.max(...all.map((m) => m.charSimilarity)));
      bestToken.push(Math.max(...all.map((m) => m.tokenSimilarity)));
      sourceAlignedExactMatches += gemmaCandidates.some((gemma) => gemma.sentence_candidate === gpt.sentence_candidate) ? 1 : 0;
      sourceAlignedNormalizedMatches += gemmaCandidates.some(
        (gemma) => normalize(gemma.sentence_candidate) === normalize(gpt.sentence_candidate),
      ) ? 1 : 0;
    }
  }
  return {
    sameSlotCharSimilarity: summarize(sameSlotChar),
    sameSlotTokenSimilarity: summarize(sameSlotToken),
    bestOfThreeCharSimilarity: summarize(bestChar),
    bestOfThreeTokenSimilarity: summarize(bestToken),
    sourceAlignedExactMatches,
    sourceAlignedNormalizedMatches,
    bestOfThreeTokenSimilarityThresholdCounts: {
      below050: bestToken.filter((value) => value < 0.5).length,
      below075: bestToken.filter((value) => value < 0.75).length,
      atLeast090: bestToken.filter((value) => value >= 0.9).length,
    },
  };
}

const [gemmaWorkbookBytes, gptWorkbookBytes, gemmaRows, gptRows] = await Promise.all([
  fs.readFile(files.gemmaWorkbook),
  fs.readFile(files.gptWorkbook),
  loadCsv(files.gemmaKey),
  loadCsv(files.gptKey),
]);

if (sha256(gemmaWorkbookBytes) !== expectedHashes.gemmaWorkbook || sha256(gptWorkbookBytes) !== expectedHashes.gptWorkbook) {
  throw new Error("Released workbook hash mismatch");
}
if (gemmaRows.length !== 180 || gptRows.length !== 180) throw new Error("Expected 180 rows per generator");

const gemmaCases = new Set(gemmaRows.map((row) => row.case_id));
const gptCases = new Set(gptRows.map((row) => row.case_id));
if (gemmaCases.size !== 60 || gptCases.size !== 60 || [...gemmaCases].some((id) => !gptCases.has(id))) {
  throw new Error("Case population mismatch");
}

const sourceByCase = new Map();
for (const row of gemmaRows) {
  if (sourceByCase.has(row.case_id) && sourceByCase.get(row.case_id) !== row.sentence_original) {
    throw new Error("Gemma source inconsistency");
  }
  sourceByCase.set(row.case_id, row.sentence_original);
}

const workbookVerification = {
  gemma: await verifyWorkbook(files.gemmaWorkbook, gemmaRows),
  gptOss120b: await verifyWorkbook(files.gptWorkbook, gptRows),
};

const gemmaExact = new Set(gemmaRows.map((row) => row.sentence_candidate));
const gptExact = new Set(gptRows.map((row) => row.sentence_candidate));
const gemmaNorm = new Set(gemmaRows.map((row) => normalize(row.sentence_candidate)));
const gptNorm = new Set(gptRows.map((row) => normalize(row.sentence_candidate)));
const exactOverlap = [...gptExact].filter((value) => gemmaExact.has(value)).length;
const normalizedOverlap = [...gptNorm].filter((value) => gemmaNorm.has(value)).length;

const summary = {
  auditType: "content-blind decision-support novelty audit; no ratings, metrics, policies, or validity judgments",
  workbookHashes: expectedHashes,
  workbookVerification,
  population: { sourceCases: 60, candidatesPerGenerator: 180, candidatesPerSource: 3 },
  crossGeneratorSetOverlap: {
    exactUniqueStrings: exactOverlap,
    normalizedUniqueStrings: normalizedOverlap,
    exactRateOfGptCandidates: exactOverlap / gptRows.length,
    normalizedRateOfGptCandidates: normalizedOverlap / gptRows.length,
  },
  crossModelSimilarity: crossModel(gemmaRows, gptRows),
  sourceRelative: {
    gemma: summarizeSourceMetrics(gemmaRows, sourceByCase),
    gptOss120b: summarizeSourceMetrics(gptRows, sourceByCase),
  },
  withinSource: {
    gemma: withinSource(gemmaRows),
    gptOss120b: withinSource(gptRows),
  },
  strata: {},
};

for (const stratum of ["corpus_id", "edit_type"]) {
  summary.strata[stratum] = {};
  const values = [...new Set(gemmaRows.map((row) => row[stratum]))].sort();
  for (const value of values) {
    const gemmaSubset = gemmaRows.filter((row) => row[stratum] === value);
    const gptSubset = gptRows.filter((row) => row[stratum] === value);
    summary.strata[stratum][value] = {
      cases: new Set(gemmaSubset.map((row) => row.case_id)).size,
      gemmaTokenEditRate: summarizeSourceMetrics(gemmaSubset, sourceByCase).tokenEditRate,
      gptOss120bTokenEditRate: summarizeSourceMetrics(gptSubset, sourceByCase).tokenEditRate,
      gptBestOfThreeTokenSimilarityToGemma: crossModel(gemmaSubset, gptSubset).bestOfThreeTokenSimilarity,
    };
  }
}

const privateValues = new Set();
for (const row of [...gemmaRows, ...gptRows]) {
  for (const field of ["case_id", "candidate_id", "remote_candidate_id", "review_id", "sentence_original", "sentence_candidate"]) {
    if (row[field]) privateValues.add(row[field]);
  }
}
const draft = JSON.stringify(summary);
const leakedPrivateValues = [...privateValues].filter((value) => draft.includes(value));
if (leakedPrivateValues.length) throw new Error("Private value leaked into aggregate summary");
summary.privacyCheck = { privateValuesChecked: privateValues.size, leakedPrivateValues: 0 };

await fs.mkdir(path.dirname(OUT), { recursive: true });
await fs.writeFile(OUT, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ output: OUT, sha256: sha256(await fs.readFile(OUT)), cases: 60, rowsPerGenerator: 180 }));
