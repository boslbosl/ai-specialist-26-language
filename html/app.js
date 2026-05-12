const tokens = ["Your", "journey", "starts", "with", "one", "step"];
const tokenVectors = [
  [0.43, 0.15, 0.89],
  [0.55, 0.87, 0.66],
  [0.57, 0.85, 0.64],
  [0.22, 0.58, 0.33],
  [0.77, 0.25, 0.10],
  [0.05, 0.80, 0.55],
];

const steps = [
  {
    label: "1. 입력 벡터",
    note: "입력은 토큰마다 숫자 벡터입니다. 예제 shape는 [batch, 6, 3]입니다.",
  },
  {
    label: "2. Q/K/V projection",
    note: "nn.Linear가 같은 입력을 Query, Key, Value 세 관점의 벡터로 바꿉니다.",
  },
  {
    label: "3. attention scores",
    note: "Query와 Key의 내적이 토큰 간 관련도 점수 행렬을 만듭니다.",
  },
  {
    label: "4. causal mask",
    note: "미래 토큰 위치는 -inf로 막고, softmax 후 확률이 0이 되게 만듭니다.",
  },
  {
    label: "5. context vector",
    note: "attention weight로 Value를 가중합하면 문맥이 반영된 토큰 표현이 됩니다.",
  },
];

const generationPlans = [
  {
    next: "a",
    candidates: [
      ["a", 0.46],
      ["the", 0.24],
      ["student", 0.18],
      ["model", 0.12],
    ],
  },
  {
    next: "student",
    candidates: [
      ["student", 0.52],
      ["developer", 0.22],
      ["model", 0.16],
      ["teacher", 0.10],
    ],
  },
  {
    next: "learning",
    candidates: [
      ["learning", 0.49],
      ["building", 0.21],
      ["reading", 0.18],
      ["testing", 0.12],
    ],
  },
  {
    next: "attention",
    candidates: [
      ["attention", 0.55],
      ["tokens", 0.19],
      ["PyTorch", 0.15],
      ["GPT", 0.11],
    ],
  },
  {
    next: "step",
    candidates: [
      ["step", 0.44],
      ["carefully", 0.27],
      ["now", 0.17],
      ["again", 0.12],
    ],
  },
];

const el = {
  stepRange: document.querySelector("#step-range"),
  tokenRange: document.querySelector("#token-range"),
  headRange: document.querySelector("#head-range"),
  maskToggle: document.querySelector("#mask-toggle"),
  stepLabel: document.querySelector("#step-label"),
  tokenLabel: document.querySelector("#token-label"),
  headLabel: document.querySelector("#head-label"),
  shapeList: document.querySelector("#shape-list"),
  attentionNote: document.querySelector("#attention-note"),
  tokenRow: document.querySelector("#token-row"),
  matrix: document.querySelector("#attention-matrix"),
  lines: document.querySelector("#attention-lines"),
  layerRange: document.querySelector("#layer-range"),
  layerLabel: document.querySelector("#layer-label"),
  embeddingRange: document.querySelector("#embedding-range"),
  embeddingLabel: document.querySelector("#embedding-label"),
  blockStack: document.querySelector("#block-stack"),
  gptShapes: document.querySelector("#gpt-shapes"),
  promptInput: document.querySelector("#prompt-input"),
  generatedTokens: document.querySelector("#generated-tokens"),
  logitBars: document.querySelector("#logit-bars"),
  generationNote: document.querySelector("#generation-note"),
  loopState: document.querySelector("#loop-state"),
  playButton: document.querySelector("#play-button"),
  stepButton: document.querySelector("#step-button"),
  resetButton: document.querySelector("#reset-button"),
  temperatureRange: document.querySelector("#temperature-range"),
  temperatureLabel: document.querySelector("#temperature-label"),
  topkRange: document.querySelector("#topk-range"),
  topkLabel: document.querySelector("#topk-label"),
  lensModeButtons: document.querySelectorAll("[data-lens-mode]"),
  lensMap: document.querySelector("#lens-map"),
  lensNote: document.querySelector("#lens-note"),
  pipelineStages: document.querySelectorAll("[data-pipeline-stage]"),
  trainStepRange: document.querySelector("#train-step-range"),
  trainStepLabel: document.querySelector("#train-step-label"),
  epochRange: document.querySelector("#epoch-range"),
  epochLabel: document.querySelector("#epoch-label"),
  batchRange: document.querySelector("#batch-range"),
  batchLabel: document.querySelector("#batch-label"),
  trainingFlow: document.querySelector("#training-flow"),
  lossChart: document.querySelector("#loss-chart"),
  trainingNote: document.querySelector("#training-note"),
  trainingShapes: document.querySelector("#training-shapes"),
  finetuneModeButtons: document.querySelectorAll("[data-finetune-mode]"),
  classRange: document.querySelector("#class-range"),
  classLabel: document.querySelector("#class-label"),
  rankRange: document.querySelector("#rank-range"),
  rankLabel: document.querySelector("#rank-label"),
  freezeToggle: document.querySelector("#freeze-toggle"),
  finetuneFlow: document.querySelector("#finetune-flow"),
  adapterVisual: document.querySelector("#adapter-visual"),
  finetuneNote: document.querySelector("#finetune-note"),
  finetuneShapes: document.querySelector("#finetune-shapes"),
  alignModeButtons: document.querySelectorAll("[data-align-mode]"),
  responseMaskRange: document.querySelector("#response-mask-range"),
  responseMaskLabel: document.querySelector("#response-mask-label"),
  betaRange: document.querySelector("#beta-range"),
  betaLabel: document.querySelector("#beta-label"),
  referenceToggle: document.querySelector("#reference-toggle"),
  alignmentFlow: document.querySelector("#alignment-flow"),
  alignmentVisual: document.querySelector("#alignment-visual"),
  alignmentNote: document.querySelector("#alignment-note"),
  alignmentShapes: document.querySelector("#alignment-shapes"),
};

let generationIndex = 0;
let loopStep = 0;
let generated = [];
let playTimer = null;
let finetuneMode = "classification";
let alignMode = "sft";
let lensMode = "overview";

const lensModes = {
  overview: {
    activeStages: ["token", "attention", "model", "logits"],
    note: "큰 그림에서는 텍스트가 token id가 되고, GPTModel을 지나 logits와 다음 토큰으로 이어지는 흐름만 잡습니다.",
    nodes: [
      ["Text", "문자열"],
      ["Token IDs", "정수 시퀀스"],
      ["GPTModel", "문맥 계산"],
      ["Logits", "후보 점수"],
      ["Next Token", "반복 생성"],
    ],
  },
  shape: {
    activeStages: ["token", "attention", "model", "logits"],
    note: "Shape 렌즈에서는 각 단계의 tensor 차원이 맞는지 확인합니다. PyTorch 코드를 읽을 때 가장 먼저 볼 기준입니다.",
    nodes: [
      ["input_ids", "[batch, seq_len]"],
      ["embeddings", "[batch, seq_len, emb_dim]"],
      ["attention", "[batch, heads, seq, seq]"],
      ["hidden", "[batch, seq_len, emb_dim]"],
      ["logits", "[batch, seq_len, vocab]"],
    ],
  },
  training: {
    activeStages: ["token", "model", "logits", "training"],
    note: "학습 렌즈에서는 logits와 target을 비교해 loss를 만들고, backward와 optimizer step이 파라미터를 바꾸는 위치를 봅니다.",
    nodes: [
      ["Dataset", "input/target"],
      ["Forward", "logits"],
      ["Loss", "target 비교"],
      ["Backward", "gradient"],
      ["Update", "optimizer.step"],
    ],
  },
  alignment: {
    activeStages: ["model", "training", "alignment"],
    note: "정렬 렌즈에서는 instruction 응답 토큰 또는 chosen/rejected 선호 쌍이 어떤 loss로 모델 행동을 조정하는지 봅니다.",
    nodes: [
      ["Instruction", "SFT labels"],
      ["Preference", "chosen/rejected"],
      ["Policy", "업데이트 대상"],
      ["Reference", "DPO 기준"],
      ["Aligned Output", "응답 품질"],
    ],
  },
};

const trainingSteps = [
  ["Text Corpus", "원문 텍스트를 모아 next-token prediction용 재료로 사용합니다."],
  ["DataLoader", "Dataset이 input_ids와 target_ids를 만들고 DataLoader가 batch로 묶습니다."],
  ["GPT Forward", "모델은 각 위치마다 vocabulary 전체에 대한 logits를 냅니다."],
  ["CrossEntropyLoss", "logits와 한 칸 뒤 target token을 비교해 scalar loss를 만듭니다."],
  ["Backward", "loss.backward()가 학습 가능한 파라미터의 gradient를 채웁니다."],
  ["Optimizer Step", "optimizer.step()이 gradient 방향으로 모델 파라미터를 업데이트합니다."],
  ["Eval + Sample", "validation loss와 샘플 생성을 보며 학습 상태를 확인합니다."],
];

const finetuneFlows = {
  classification: [
    ["CSV Text", "분류 라벨이 붙은 텍스트 데이터"],
    ["Tokenizer + Padding", "문장 길이를 batch 안에서 맞춤"],
    ["GPT Backbone", "문장 표현을 계산"],
    ["Last Hidden State", "마지막 토큰 표현을 문장 표현으로 사용"],
    ["Linear Class Head", "class logits 출력"],
    ["Loss + Accuracy", "라벨과 비교"],
  ],
  lora: [
    ["Pretrained GPT", "이미 학습된 backbone"],
    ["Freeze Base W", "큰 기존 가중치는 고정"],
    ["LoRA A", "낮은 rank로 down projection"],
    ["LoRA B", "원래 차원으로 up projection"],
    ["Add to Linear", "원래 Linear 출력에 LoRA 업데이트를 더함"],
    ["Train Adapters", "작은 파라미터만 업데이트"],
  ],
};

const alignFlows = {
  sft: [
    ["Instruction JSON", "instruction/input/output 데이터"],
    ["Prompt Format", "학습용 프롬프트 템플릿"],
    ["Tokenize", "prompt와 response를 토큰화"],
    ["Label Mask", "응답 토큰만 loss 대상"],
    ["SFT Loss", "응답 next-token loss"],
    ["Generate + Eval", "튜닝 결과 확인"],
  ],
  dpo: [
    ["Prompt", "동일 질문"],
    ["Chosen", "선호 응답"],
    ["Rejected", "비선호 응답"],
    ["Policy Logprob", "현재 모델 확률"],
    ["Reference Logprob", "고정 기준 모델 확률"],
    ["DPO Loss", "선호 차이를 키움"],
  ],
};

function deterministicWeight(head, kind, inputDim, outputDim) {
  const weights = [];
  const offset = kind === "q" ? 0.19 : kind === "k" ? 0.31 : 0.43;
  for (let i = 0; i < inputDim; i += 1) {
    const row = [];
    for (let j = 0; j < outputDim; j += 1) {
      const raw = Math.sin((head + 1) * (i + 2) * (j + 3) + offset);
      row.push(Number((raw * 0.42 + 0.08 * (j + 1)).toFixed(4)));
    }
    weights.push(row);
  }
  return weights;
}

function matMulVector(vector, matrix) {
  return matrix[0].map((_, col) =>
    vector.reduce((sum, value, row) => sum + value * matrix[row][col], 0)
  );
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function softmax(values) {
  const finite = values.filter(Number.isFinite);
  const max = finite.length ? Math.max(...finite) : 0;
  const exps = values.map((value) => (Number.isFinite(value) ? Math.exp(value - max) : 0));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

function computeAttention() {
  const headCount = Number(el.headRange.value);
  const selectedHead = Math.min(0, headCount - 1);
  const modelDim = 8;
  const headDim = modelDim / headCount;
  const qWeight = deterministicWeight(selectedHead, "q", 3, headDim);
  const kWeight = deterministicWeight(selectedHead, "k", 3, headDim);
  const vWeight = deterministicWeight(selectedHead, "v", 3, headDim);
  const queries = tokenVectors.map((vector) => matMulVector(vector, qWeight));
  const keys = tokenVectors.map((vector) => matMulVector(vector, kWeight));
  const values = tokenVectors.map((vector) => matMulVector(vector, vWeight));
  const useMask = el.maskToggle.checked;
  const scores = queries.map((query, row) =>
    keys.map((key, col) => {
      const score = dot(query, key) / Math.sqrt(headDim);
      return useMask && col > row ? -Infinity : score;
    })
  );
  const weights = scores.map((row) => softmax(row));
  const context = weights.map((rowWeights) =>
    values[0].map((_, dim) =>
      rowWeights.reduce((sum, weight, tokenIndex) => sum + weight * values[tokenIndex][dim], 0)
    )
  );

  return { headCount, modelDim, headDim, scores, weights, context };
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-inf";
  return value.toFixed(2);
}

function setShapeList(target, rows) {
  target.innerHTML = rows
    .map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderLens() {
  const mode = lensModes[lensMode];
  el.lensModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.lensMode === lensMode);
  });
  el.pipelineStages.forEach((stage) => {
    stage.classList.toggle("active", mode.activeStages.includes(stage.dataset.pipelineStage));
  });
  el.lensMap.innerHTML = mode.nodes
    .map(([title, detail], index) => `
      <div class="lens-node${index === 2 ? " active" : ""}">
        ${title}<br><small>${detail}</small>
      </div>
    `)
    .join("");
  el.lensNote.innerHTML = `<strong>${lensMode.toUpperCase()}</strong><br>${mode.note}`;
}

function renderTokens(weights) {
  const selectedToken = Number(el.tokenRange.value);
  const step = Number(el.stepRange.value);
  el.tokenRow.innerHTML = tokens
    .map((token, index) => {
      const selected = index === selectedToken ? " selected" : "";
      const blocked = el.maskToggle.checked && index > selectedToken && step >= 3 ? " blocked" : "";
      const weight = step >= 4 ? weights[selectedToken][index] : 0;
      return `
        <div class="token${selected}${blocked}">
          ${token}
          <small>x${index + 1}${step >= 4 ? ` · ${weight.toFixed(2)}` : ""}</small>
        </div>
      `;
    })
    .join("");
}

function renderLines(weights) {
  const selectedToken = Number(el.tokenRange.value);
  const step = Number(el.stepRange.value);
  const width = 780;
  const startY = 188;
  const topY = 54;
  const gap = width / tokens.length;
  const selectedX = gap * selectedToken + gap / 2;
  const lines = [];

  tokens.forEach((token, index) => {
    const x = gap * index + gap / 2;
    const isFuture = el.maskToggle.checked && index > selectedToken;
    const visibleWeight = step >= 4 ? weights[selectedToken][index] : index === selectedToken ? 1 : 0.16;
    const opacity = isFuture && step >= 3 ? 0.12 : Math.max(0.18, visibleWeight);
    const stroke = isFuture && step >= 3 ? "#c04444" : index === selectedToken ? "#2468d1" : "#087f8c";
    const widthValue = step >= 4 ? 1 + visibleWeight * 9 : 2.5;
    lines.push(`
      <path d="M ${selectedX} ${startY} C ${selectedX} ${topY}, ${x} ${topY}, ${x} ${startY}"
        fill="none" stroke="${stroke}" stroke-width="${widthValue.toFixed(2)}"
        stroke-opacity="${opacity.toFixed(2)}" />
    `);
    lines.push(`
      <circle cx="${x}" cy="${startY}" r="${index === selectedToken ? 7 : 5}"
        fill="${isFuture && step >= 3 ? "#f6c7c7" : "#ffffff"}"
        stroke="${stroke}" stroke-width="2" />
    `);
  });

  el.lines.innerHTML = lines.join("");
}

function renderMatrix(attention) {
  const selectedToken = Number(el.tokenRange.value);
  const step = Number(el.stepRange.value);
  const showWeights = step >= 4;
  const values = showWeights ? attention.weights : attention.scores;
  const cells = [`<div class="matrix-cell axis">Q \\ K</div>`];
  tokens.forEach((token) => cells.push(`<div class="matrix-cell axis">${token}</div>`));

  tokens.forEach((rowToken, row) => {
    cells.push(`<div class="matrix-cell axis">${rowToken}</div>`);
    tokens.forEach((_, col) => {
      const masked = el.maskToggle.checked && col > row && step >= 3;
      const activeRow = row === selectedToken ? " active-row" : "";
      const className = `matrix-cell${masked ? " masked" : ""}${activeRow}`;
      const value = masked ? "-inf" : showWeights ? values[row][col].toFixed(2) : formatNumber(values[row][col]);
      const intensity = showWeights && !masked ? values[row][col] : 0;
      const background = showWeights && !masked
        ? `style="background: rgba(8, 127, 140, ${Math.min(0.12 + intensity * 0.85, 0.9).toFixed(2)}); color: ${intensity > 0.55 ? "#ffffff" : "inherit"}"`
        : "";
      cells.push(`<div class="${className}" ${background}>${value}</div>`);
    });
  });

  el.matrix.innerHTML = cells.join("");
}

function renderAttention() {
  const attention = computeAttention();
  const step = Number(el.stepRange.value);
  const selectedToken = Number(el.tokenRange.value);
  const headCount = attention.headCount;

  el.stepLabel.textContent = steps[step].label;
  el.tokenLabel.textContent = `${tokens[selectedToken]} · index ${selectedToken}`;
  el.headLabel.textContent = `${headCount} heads · head_dim ${attention.headDim}`;
  el.attentionNote.innerHTML = `<strong>${steps[step].label}</strong><br>${steps[step].note}`;

  renderTokens(attention.weights);
  renderLines(attention.weights);
  renderMatrix(attention);

  setShapeList(el.shapeList, [
    ["input x", "[2, 6, 3]"],
    ["Q/K/V", `[2, ${headCount}, 6, ${attention.headDim}]`],
    ["scores", `[2, ${headCount}, 6, 6]`],
    ["context", `[2, 6, ${attention.modelDim}]`],
  ]);
}

function renderBlocks() {
  const layers = Number(el.layerRange.value);
  const emb = Number(el.embeddingRange.value);
  el.layerLabel.textContent = `${layers} layers`;
  el.embeddingLabel.textContent = `emb_dim ${emb}`;

  const maxVisible = 12;
  el.blockStack.innerHTML = Array.from({ length: maxVisible }, (_, index) => {
    const active = index < layers;
    return `<div class="mini-block${active ? "" : " dimmed"}">Block ${index + 1}</div>`;
  }).join("");

  const headCount = Math.max(1, Math.min(12, emb / 64));
  setShapeList(el.gptShapes, [
    ["input ids", "[batch, seq_len]"],
    ["embeddings", `[batch, seq_len, ${emb}]`],
    ["attention heads", `${headCount} heads`],
    ["block output", `[batch, seq_len, ${emb}]`],
    ["logits", "[batch, seq_len, 50257]"],
  ]);
}

function resetGeneration() {
  generated = el.promptInput.value.trim().split(/\s+/).filter(Boolean);
  generationIndex = 0;
  loopStep = 0;
  stopPlaying();
  renderGeneration();
}

function stopPlaying() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  el.playButton.textContent = "▶";
  el.playButton.title = "재생";
  el.playButton.setAttribute("aria-label", "재생");
}

function advanceLoop() {
  loopStep += 1;
  if (loopStep > 4) {
    const plan = generationPlans[generationIndex % generationPlans.length];
    generated.push(plan.next);
    generationIndex += 1;
    loopStep = 0;
  }
  renderGeneration();
}

function renderGeneratedTokens() {
  const lastNewIndex = generated.length - 1;
  el.generatedTokens.innerHTML = generated
    .map((token, index) => `<span class="generated-token${index === lastNewIndex && generationIndex > 0 ? " new" : ""}">${token}</span>`)
    .join("");
}

function renderLoopNodes() {
  document.querySelectorAll(".loop-node").forEach((node) => {
    node.classList.toggle("active", Number(node.dataset.loopStep) === loopStep);
  });
}

function getCandidateDistribution(plan) {
  const temperature = Number(el.temperatureRange.value) / 10;
  const topK = Number(el.topkRange.value);
  const ranked = [...plan.candidates]
    .map(([name, base]) => ({ name, base, logit: Math.log(base) }))
    .sort((a, b) => b.logit - a.logit);
  const kept = new Set(ranked.slice(0, topK).map((item) => item.name));
  const adjusted = ranked.map((item) => ({
    ...item,
    kept: kept.has(item.name),
    score: kept.has(item.name) ? Math.exp(item.logit / temperature) : 0,
  }));
  const total = adjusted.reduce((sum, item) => sum + item.score, 0) || 1;
  return adjusted
    .map((item) => ({ ...item, probability: item.score / total }))
    .sort((a, b) => b.base - a.base);
}

function renderLogits() {
  const plan = generationPlans[generationIndex % generationPlans.length];
  const distribution = getCandidateDistribution(plan);
  const maxScore = Math.max(...distribution.map((candidate) => candidate.probability));
  el.temperatureLabel.textContent = `temperature ${(Number(el.temperatureRange.value) / 10).toFixed(1)}`;
  el.topkLabel.textContent = `top-${Number(el.topkRange.value)}`;
  el.logitBars.innerHTML = distribution
    .map(({ name, probability, kept }) => {
      const width = Math.round((probability / maxScore) * 100);
      const winner = kept && name === plan.next ? " winner" : "";
      const muted = kept ? "" : " filtered";
      return `
        <div class="bar-row${winner}${muted}">
          <div class="bar-name">${name}</div>
          <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
          <div class="bar-score">${kept ? probability.toFixed(2) : "off"}</div>
        </div>
      `;
    })
    .join("");
}

function renderGeneration() {
  const plan = generationPlans[generationIndex % generationPlans.length];
  const notes = [
    "현재 입력이 context window 크기를 넘지 않도록 마지막 토큰들만 남깁니다.",
    "모델은 [batch, seq_len, vocab_size] 형태의 logits를 계산합니다.",
    "다음 토큰은 마지막 위치 logits인 logits[:, -1, :]에서 고릅니다. temperature와 top-k는 이 후보 분포를 바꿉니다.",
    `현재 설정에서 가장 큰 후보는 "${plan.next}"입니다.`,
    "선택한 토큰 ID를 기존 입력 뒤에 붙이고 같은 과정을 반복합니다.",
  ];

  renderGeneratedTokens();
  renderLoopNodes();
  renderLogits();
  el.generationNote.innerHTML = `<strong>Step ${loopStep + 1}</strong><br>${notes[loopStep]}`;
  setShapeList(el.loopState, [
    ["tokens", `${generated.length}`],
    ["next token", plan.next],
    ["loop step", `${loopStep + 1} / 5`],
    ["max_new_tokens", `${generationPlans.length}`],
  ]);
}

function renderTraining() {
  const step = Number(el.trainStepRange.value);
  const epoch = Number(el.epochRange.value);
  const batch = Number(el.batchRange.value);
  el.trainStepLabel.textContent = `${trainingSteps[step][0]} · ${step + 1} / ${trainingSteps.length}`;
  el.epochLabel.textContent = `${epoch} epochs`;
  el.batchLabel.textContent = `batch ${batch}`;

  el.trainingFlow.innerHTML = trainingSteps
    .map(([label], index) => {
      const state = index === step ? " active" : index < step ? " done" : "";
      return `<div class="train-node${state}" data-train-step="${index}">${label}</div>`;
    })
    .join("");

  const trainLoss = Math.max(0.92, 4.2 * Math.exp(-epoch * 0.18));
  const valLoss = Math.max(1.05, 4.35 * Math.exp(-epoch * 0.15) + epoch * 0.015);
  const maxLoss = 4.4;
  el.lossChart.innerHTML = [
    ["train", trainLoss, ""],
    ["validation", valLoss, " validation"],
  ].map(([label, loss, className]) => {
    const width = Math.round((loss / maxLoss) * 100);
    return `
      <div class="loss-row${className}">
        <div class="loss-label">${label}</div>
        <div class="loss-track"><div class="loss-fill" style="width: ${width}%"></div></div>
        <div class="loss-value">${loss.toFixed(2)}</div>
      </div>
    `;
  }).join("");

  el.trainingNote.innerHTML = `<strong>${trainingSteps[step][0]}</strong><br>${trainingSteps[step][1]}`;
  setShapeList(el.trainingShapes, [
    ["input_ids", `[${batch}, 256]`],
    ["target_ids", `[${batch}, 256]`],
    ["logits", `[${batch}, 256, 50257]`],
    ["loss", "scalar"],
  ]);
}

function renderFinetuning() {
  const numClasses = Number(el.classRange.value);
  const rank = Number(el.rankRange.value);
  const frozen = el.freezeToggle.checked;
  const flow = finetuneFlows[finetuneMode];

  el.finetuneModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.finetuneMode === finetuneMode);
  });

  el.classLabel.textContent = `${numClasses} classes`;
  el.rankLabel.textContent = `rank ${rank}`;
  el.rankRange.disabled = finetuneMode !== "lora";
  el.freezeToggle.disabled = finetuneMode !== "lora";

  el.finetuneFlow.innerHTML = flow
    .map(([label], index) => `<div class="finetune-node${index <= 2 ? " done" : index === 3 ? " active" : ""}">${label}</div>`)
    .join("");

  if (finetuneMode === "classification") {
    el.adapterVisual.innerHTML = `
      <div class="adapter-box">GPT Backbone<br>[batch, seq_len, emb_dim]</div>
      <div class="adapter-box trainable">Classification Head<br>emb_dim -> ${numClasses}</div>
      <div class="adapter-box">Class Logits<br>[batch, ${numClasses}]</div>
    `;
    el.finetuneNote.innerHTML = "<strong>분류 미세튜닝</strong><br>사전학습 GPT의 문장 표현 위에 작은 Linear head를 붙이고 class logits를 라벨과 비교합니다.";
    setShapeList(el.finetuneShapes, [
      ["input_ids", "[batch, seq_len]"],
      ["hidden state", "[batch, seq_len, emb_dim]"],
      ["class logits", `[batch, ${numClasses}]`],
      ["loss", "CrossEntropyLoss"],
    ]);
    return;
  }

  el.adapterVisual.innerHTML = `
    <div class="adapter-box${frozen ? " frozen" : ""}">Base Linear W<br>${frozen ? "frozen" : "trainable"}</div>
    <div class="adapter-box trainable">LoRA A/B<br>rank ${rank}</div>
    <div class="adapter-box">W(x) + LoRA(x)<br>scaled update</div>
  `;
  el.finetuneNote.innerHTML = "<strong>LoRA 미세튜닝</strong><br>기존 큰 가중치는 그대로 두고, 낮은 rank의 작은 A/B 행렬만 학습해 업데이트 경로를 추가합니다.";
  setShapeList(el.finetuneShapes, [
    ["base W", frozen ? "frozen" : "trainable"],
    ["LoRA A", `[emb_dim, ${rank}]`],
    ["LoRA B", `[${rank}, emb_dim]`],
    ["trainable params", "small adapter only"],
  ]);
}

function renderAlignment() {
  const responseRatio = Number(el.responseMaskRange.value);
  const beta = Number(el.betaRange.value) / 10;
  const referenceFrozen = el.referenceToggle.checked;
  const flow = alignFlows[alignMode];

  el.alignModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.alignMode === alignMode);
  });

  el.responseMaskLabel.textContent = `${responseRatio}% response tokens`;
  el.betaLabel.textContent = `beta ${beta.toFixed(1)}`;
  el.responseMaskRange.disabled = alignMode !== "sft";
  el.betaRange.disabled = alignMode !== "dpo";
  el.referenceToggle.disabled = alignMode !== "dpo";

  el.alignmentFlow.innerHTML = flow
    .map(([label], index) => {
      const state = index <= 2 ? " done" : index === 3 ? " active" : "";
      return `<div class="align-node${state}">${label}</div>`;
    })
    .join("");

  if (alignMode === "sft") {
    const ignoredRatio = 100 - responseRatio;
    el.alignmentVisual.innerHTML = `
      <div class="align-box ignored">Instruction + Input<br>label = -100<br>${ignoredRatio}% ignored</div>
      <div class="align-box focus">Output Response<br>loss 대상<br>${responseRatio}% labels</div>
      <div class="align-box">CrossEntropyLoss<br>응답 토큰만 계산</div>
    `;
    el.alignmentNote.innerHTML = "<strong>SFT</strong><br>프롬프트 부분은 label을 -100으로 두어 loss에서 제외하고, 모델이 생성해야 할 응답 토큰만 학습합니다.";
    setShapeList(el.alignmentShapes, [
      ["input_ids", "[batch, seq_len]"],
      ["labels", "[batch, seq_len]"],
      ["ignored labels", `${ignoredRatio}% -> -100`],
      ["loss target", "response tokens only"],
    ]);
    return;
  }

  const chosenScore = Math.min(0.92, 0.52 + beta * 0.13);
  const rejectedScore = Math.max(0.08, 0.48 - beta * 0.11);
  el.alignmentVisual.innerHTML = `
    <div class="align-box focus">Chosen Response<br>policy logprob ↑</div>
    <div class="align-box rejected">Rejected Response<br>policy logprob ↓</div>
    <div class="align-box${referenceFrozen ? " ignored" : ""}">Reference Model<br>${referenceFrozen ? "frozen" : "trainable"}</div>
    <div class="preference-meter">
      <div class="meter-row">
        <div class="meter-label">chosen</div>
        <div class="meter-track"><div class="meter-fill" style="width: ${Math.round(chosenScore * 100)}%"></div></div>
        <div class="meter-value">${chosenScore.toFixed(2)}</div>
      </div>
      <div class="meter-row rejected">
        <div class="meter-label">rejected</div>
        <div class="meter-track"><div class="meter-fill" style="width: ${Math.round(rejectedScore * 100)}%"></div></div>
        <div class="meter-value">${rejectedScore.toFixed(2)}</div>
      </div>
    </div>
  `;
  el.alignmentNote.innerHTML = "<strong>DPO</strong><br>chosen과 rejected의 상대 logprob 차이를 policy와 reference에서 비교하고, policy만 선호 방향으로 업데이트합니다.";
  setShapeList(el.alignmentShapes, [
    ["prompt", "[batch, prompt_len]"],
    ["chosen/rejected", "[batch, response_len]"],
    ["beta", beta.toFixed(1)],
    ["updated model", "policy only"],
  ]);
}

function togglePlay() {
  if (playTimer) {
    stopPlaying();
    return;
  }
  el.playButton.textContent = "Ⅱ";
  el.playButton.title = "일시정지";
  el.playButton.setAttribute("aria-label", "일시정지");
  playTimer = setInterval(() => {
    advanceLoop();
    if (generationIndex >= generationPlans.length && loopStep === 0) {
      stopPlaying();
    }
  }, 900);
}

["input", "change"].forEach((eventName) => {
  el.stepRange.addEventListener(eventName, renderAttention);
  el.tokenRange.addEventListener(eventName, renderAttention);
  el.headRange.addEventListener(eventName, renderAttention);
  el.layerRange.addEventListener(eventName, renderBlocks);
  el.embeddingRange.addEventListener(eventName, renderBlocks);
  el.trainStepRange.addEventListener(eventName, renderTraining);
  el.epochRange.addEventListener(eventName, renderTraining);
  el.batchRange.addEventListener(eventName, renderTraining);
  el.classRange.addEventListener(eventName, renderFinetuning);
  el.rankRange.addEventListener(eventName, renderFinetuning);
  el.responseMaskRange.addEventListener(eventName, renderAlignment);
  el.betaRange.addEventListener(eventName, renderAlignment);
  el.temperatureRange.addEventListener(eventName, renderGeneration);
  el.topkRange.addEventListener(eventName, renderGeneration);
});

el.maskToggle.addEventListener("change", renderAttention);
el.playButton.addEventListener("click", togglePlay);
el.stepButton.addEventListener("click", advanceLoop);
el.resetButton.addEventListener("click", resetGeneration);
el.promptInput.addEventListener("change", resetGeneration);
el.freezeToggle.addEventListener("change", renderFinetuning);
el.finetuneModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    finetuneMode = button.dataset.finetuneMode;
    renderFinetuning();
  });
});
el.referenceToggle.addEventListener("change", renderAlignment);
el.alignModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    alignMode = button.dataset.alignMode;
    renderAlignment();
  });
});
el.lensModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    lensMode = button.dataset.lensMode;
    renderLens();
  });
});

renderLens();
renderAttention();
renderBlocks();
resetGeneration();
renderTraining();
renderFinetuning();
renderAlignment();
