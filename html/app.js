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
    note: "각 토큰은 계산하기 쉬운 숫자 벡터로 표현됩니다. 여기서는 토큰 6개, 차원 3개짜리 예제로 봅니다.",
  },
  {
    label: "2. Q/K/V 만들기",
    note: "같은 입력 벡터를 질문(Q), 비교 기준(K), 가져올 내용(V) 세 역할의 벡터로 바꿉니다.",
  },
  {
    label: "3. 관련도 점수",
    note: "Q와 K를 비교해 토큰끼리 얼마나 관련 있는지 점수 행렬로 만듭니다.",
  },
  {
    label: "4. 미래 토큰 가리기",
    note: "GPT는 다음 토큰을 예측해야 하므로 아직 나오지 않은 미래 토큰은 참고하지 못하게 막습니다.",
  },
  {
    label: "5. 문맥 벡터",
    note: "참고 비율에 따라 V를 섞으면 주변 문맥이 반영된 새 토큰 표현이 됩니다.",
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
  boxPopover: document.querySelector("#box-popover"),
  boxPopoverTitle: document.querySelector("#box-popover-title"),
  boxPopoverBody: document.querySelector("#box-popover-body"),
  boxPopoverTerm: document.querySelector("#box-popover-term"),
  boxPopoverExample: document.querySelector("#box-popover-example"),
  exampleModal: document.querySelector("#example-modal"),
  exampleModalTitle: document.querySelector("#example-modal-title"),
  exampleModalSource: document.querySelector("#example-modal-source"),
  exampleModalNote: document.querySelector("#example-modal-note"),
  exampleCode: document.querySelector("#example-code"),
  exampleClose: document.querySelector("#example-close"),
};

let generationIndex = 0;
let loopStep = 0;
let generated = [];
let playTimer = null;
let finetuneMode = "classification";
let alignMode = "sft";
let lensMode = "overview";
let detailRefreshPending = false;
let activeDetailBox = null;
let hidePopoverTimer = null;

const lensModes = {
  overview: {
    activeStages: ["token", "attention", "model", "logits"],
    note: "큰 그림에서는 문장이 토큰 번호가 되고, GPTModel을 지나 다음 토큰 후보 점수로 바뀌는 순서만 잡습니다.",
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
    note: "Shape 렌즈에서는 각 단계의 텐서 차원이 맞는지 확인합니다. PyTorch 코드를 읽을 때 가장 먼저 볼 기준입니다.",
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
    note: "학습 렌즈에서는 예측 점수와 정답을 비교해 loss를 만들고, 그 결과가 어디에서 모델 값을 바꾸는지 봅니다.",
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
    note: "정렬 렌즈에서는 지시문 답변 학습과 선호 답변 비교가 어떻게 모델의 말투와 선택을 바꾸는지 봅니다.",
    nodes: [
      ["Instruction", "SFT labels"],
      ["Preference Pair", "선호/비선호 답변"],
      ["Policy", "업데이트 대상"],
      ["Reference", "DPO 기준"],
      ["Aligned Output", "응답 품질"],
    ],
  },
};

const trainingSteps = [
  ["Text Corpus", "원문 텍스트를 모아 다음 토큰 맞히기 학습의 재료로 사용합니다."],
  ["DataLoader", "Dataset이 입력 토큰과 한 칸 뒤 정답 토큰을 만들고, DataLoader가 여러 문장을 batch로 묶습니다."],
  ["GPT Forward", "모델은 각 위치마다 다음 토큰 후보 전체에 대한 점수를 냅니다."],
  ["CrossEntropyLoss", "예측 점수와 정답 토큰을 비교해 틀린 정도를 하나의 loss 숫자로 만듭니다."],
  ["Backward", "loss.backward()가 어떤 파라미터를 어느 방향으로 바꿔야 할지 계산합니다."],
  ["Optimizer Step", "optimizer.step()이 계산된 방향을 사용해 모델 파라미터 값을 실제로 수정합니다."],
  ["Eval + Sample", "검증 loss와 샘플 생성을 보며 모델이 외운 것인지, 실제로 나아진 것인지 확인합니다."],
];

const finetuneFlows = {
  classification: [
    ["CSV Text", "분류 라벨이 붙은 텍스트 데이터"],
    ["Tokenizer + Padding", "문장 길이를 batch 안에서 맞춤"],
    ["GPT Backbone", "문장 표현을 계산"],
    ["Last Hidden State", "마지막 토큰 표현을 문장 표현으로 사용"],
    ["Linear Class Head", "클래스별 점수 출력"],
    ["Loss + Accuracy", "정답 라벨과 비교"],
  ],
  lora: [
    ["Pretrained GPT", "이미 학습된 기본 모델"],
    ["Freeze Base W", "큰 기존 가중치는 고정"],
    ["LoRA A", "작은 차원으로 줄이는 행렬"],
    ["LoRA B", "원래 차원으로 되돌리는 행렬"],
    ["Add to Linear", "기존 출력에 작은 보정값을 더함"],
    ["Train Adapters", "작은 파라미터만 업데이트"],
  ],
};

const alignFlows = {
  sft: [
    ["Instruction JSON", "instruction/input/output 데이터"],
    ["Prompt Format", "학습용 프롬프트 템플릿"],
    ["Tokenize", "질문과 답변을 토큰화"],
    ["Label Mask", "응답 토큰만 loss 대상"],
    ["SFT Loss", "답변의 다음 토큰만 학습"],
    ["Generate + Eval", "튜닝 결과 확인"],
  ],
  dpo: [
    ["Prompt", "동일 질문"],
    ["Chosen", "선호 응답"],
    ["Rejected", "비선호 응답"],
    ["Policy Logprob", "학습 중인 모델의 답변 점수"],
    ["Reference Logprob", "비교용 고정 모델의 답변 점수"],
    ["DPO Loss", "선호 차이를 키움"],
  ],
};

const codeExamples = {
  dataset: {
    title: "Next-token dataset",
    source: "Chapter_2_Exercise_Dataset.ipynb",
    note: "문장을 토큰 번호로 바꾼 뒤, 입력은 현재 구간으로 두고 정답은 한 칸 뒤 구간으로 맞춥니다.",
    code: [
      "token_ids = tokenizer.encode(txt, allowed_special={\"<|endoftext|>\"})",
      "",
      "for i in range(0, len(token_ids) - max_length, stride):",
      "    input_chunk = token_ids[i:i + max_length]",
      "    target_chunk = token_ids[i + 1:i + max_length + 1]",
      "    self.input_ids.append(torch.tensor(input_chunk))",
      "    self.target_ids.append(torch.tensor(target_chunk))",
    ].join("\n"),
  },
  attention: {
    title: "Causal self-attention",
    source: "Chapter_3_Excercise_Attention.ipynb",
    note: "입력 벡터를 Q/K/V 세 역할로 바꾸고, 미래 토큰을 가린 뒤, 참고할 토큰들의 값을 섞어 새 표현을 만듭니다.",
    code: [
      "queries = self.W_query(x)",
      "keys = self.W_key(x)",
      "values = self.W_value(x)",
      "",
      "attn_scores = queries @ keys.transpose(1, 2)",
      "attn_scores.masked_fill_(mask.bool(), -torch.inf)",
      "attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)",
      "context_vec = attn_weights @ values",
    ].join("\n"),
  },
  gpt: {
    title: "GPTModel forward",
    source: "Chapter_4_Excercise_GPT.ipynb",
    note: "토큰 번호와 위치 정보를 벡터로 만든 뒤, TransformerBlock들을 지나 다음 토큰 점수로 바꿉니다.",
    code: [
      "tok_embeds = self.tok_emb(in_idx)",
      "pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))",
      "x = tok_embeds + pos_embeds",
      "x = self.drop_emb(x)",
      "x = self.trf_blocks(x)",
      "x = self.final_norm(x)",
      "logits = self.out_head(x)",
    ].join("\n"),
  },
  generation: {
    title: "Autoregressive generation",
    source: "Chapter_4_Excercise_GPT.ipynb",
    note: "문장을 한 번에 만들지 않고, 마지막 위치의 다음 토큰 점수에서 하나를 고른 뒤 다시 입력 뒤에 붙입니다.",
    code: [
      "idx_cond = idx[:, -context_size:]",
      "with torch.no_grad():",
      "    logits = model(idx_cond)",
      "",
      "logits = logits[:, -1, :]",
      "idx_next = torch.argmax(logits, dim=-1, keepdim=True)",
      "idx = torch.cat((idx, idx_next), dim=1)",
    ].join("\n"),
  },
  pretraining: {
    title: "Pretraining loss step",
    source: "Chapter_5_Excercise_Pretraining.ipynb",
    note: "각 위치마다 다음 토큰을 맞히도록 점수를 펼쳐서 비교하고, 틀린 정도를 하나의 loss로 모읍니다.",
    code: [
      "logits = model(input_batch)",
      "loss = torch.nn.functional.cross_entropy(",
      "    logits.flatten(0, 1),",
      "    target_batch.flatten()",
      ")",
      "",
      "optimizer.zero_grad()",
      "loss.backward()",
      "optimizer.step()",
    ].join("\n"),
  },
  classification: {
    title: "Classification head",
    source: "Chapter_6_Excercise_Finetuning_Classification.ipynb",
    note: "GPT가 만든 문장 표현 중 대표 위치를 골라, 분류할 클래스 개수만큼의 점수로 바꿉니다.",
    code: [
      "hidden_states = model(input_ids)",
      "last_token_state = hidden_states[torch.arange(batch_size), last_token_pos]",
      "class_logits = classification_head(last_token_state)",
      "",
      "loss = torch.nn.functional.cross_entropy(class_logits, labels)",
      "predictions = torch.argmax(class_logits, dim=-1)",
    ].join("\n"),
  },
  lora: {
    title: "LoRA adapter",
    source: "Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb",
    note: "기존 큰 Linear 층은 그대로 두고, 작은 두 행렬이 만든 보정값만 더해서 적은 파라미터로 학습합니다.",
    code: [
      "class LinearWithLoRA(nn.Module):",
      "    def __init__(self, linear, rank, alpha):",
      "        super().__init__()",
      "        self.linear = linear",
      "        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)",
      "",
      "    def forward(self, x):",
      "        return self.linear(x) + self.lora(x)",
    ].join("\n"),
  },
  sft: {
    title: "Instruction label masking",
    source: "Chapter_7_Exercise_Follow_Instructions.ipynb",
    note: "질문과 지시문 부분은 조건으로만 쓰고, 모델이 직접 생성해야 하는 답변 부분에만 loss를 겁니다.",
    code: [
      "instruction_plus_input = format_input(entry)",
      "response_text = \"\\n\\n### Response:\\n\" + entry[\"output\"]",
      "full_text = instruction_plus_input + response_text",
      "",
      "labels = input_ids.clone()",
      "labels[:prompt_length] = -100",
      "labels[labels == pad_token_id] = -100",
    ].join("\n"),
  },
  dpo: {
    title: "DPO preference loss",
    source: "Chapter_7_Exercise_Follow_Instructions_dpo.ipynb",
    note: "좋은 답변과 나쁜 답변의 점수 차이를 비교해, 학습 중인 모델만 좋은 답변 쪽으로 더 기울게 만듭니다.",
    code: [
      "model_logratios = model_chosen_logprobs - model_rejected_logprobs",
      "reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs",
      "",
      "logits = model_logratios - reference_logratios",
      "losses = -torch.nn.functional.logsigmoid(beta * logits)",
      "loss = losses.mean()",
    ].join("\n"),
  },
};

const pipelineDetails = {
  token: {
    title: "Token 단계",
    body: "문장을 모델이 읽을 수 있는 토큰 번호 목록으로 바꿉니다. PyTorch에서는 보통 [batch, seq_len] 모양의 정수 텐서가 됩니다.",
    codeKey: "dataset",
  },
  attention: {
    title: "Attention 단계",
    body: "각 토큰이 앞의 어떤 토큰을 얼마나 참고할지 계산합니다. GPT는 다음 단어를 맞히는 모델이라 미래 토큰은 보지 못하게 가립니다.",
    codeKey: "attention",
  },
  model: {
    title: "Model 단계",
    body: "토큰 번호를 벡터로 바꾼 뒤 attention과 FFN을 여러 번 반복해 문맥이 반영된 표현을 만듭니다.",
    codeKey: "gpt",
  },
  logits: {
    title: "Logits 단계",
    body: "모델의 중간 표현을 전체 토큰 후보 점수로 바꾼 결과입니다. 확률로 바꾸기 전 점수라서 아직 softmax 전입니다.",
    codeKey: "generation",
  },
  training: {
    title: "Training 단계",
    body: "예측 점수와 정답을 비교해 loss를 만들고, backward와 optimizer step으로 틀린 정도가 줄어들게 파라미터를 바꿉니다.",
    codeKey: "pretraining",
  },
  alignment: {
    title: "Alignment 단계",
    body: "기본 언어 모델을 지시를 잘 따르고 사람이 선호하는 답변을 내도록 마지막으로 조정하는 학습 단계입니다.",
    codeKey: "dpo",
  },
};

function shortText(node) {
  return node.innerText.replace(/\s+/g, " ").trim();
}

function sectionCodeKey(node, label) {
  if (node.closest("#attention-lab")) return "attention";
  if (node.closest("#gpt-structure")) return "gpt";
  if (node.closest("#generation-loop")) return "generation";
  if (node.closest("#pretraining-loop")) return "pretraining";
  if (node.closest("#finetuning-lab")) {
    return /lora|rank|adapter|base linear|freeze/i.test(label) ? "lora" : "classification";
  }
  if (node.closest("#alignment-lab")) {
    return /dpo|chosen|rejected|reference|beta|preference/i.test(label) ? "dpo" : "sft";
  }
  if (node.closest("#lens-map")) {
    if (/dataset|input_ids|token/i.test(label)) return "dataset";
    if (/attention|q\/k\/v/i.test(label)) return "attention";
    if (/gpt|hidden|embedding/i.test(label)) return "gpt";
    if (/logits|next/i.test(label)) return "generation";
    if (/loss|backward|update|forward/i.test(label)) return "pretraining";
    if (/instruction|sft|label/i.test(label)) return "sft";
    if (/preference|policy|reference|aligned|chosen|rejected/i.test(label)) return "dpo";
  }
  return "gpt";
}

function simpleTerm(label, codeKey) {
  const text = label.toLowerCase();
  if (/q\/k\/v|query|key|value|attention/.test(text) || codeKey === "attention") {
    return "용어 풀이: Q는 찾는 기준, K는 비교할 이름표, V는 실제로 가져올 내용입니다.";
  }
  if ((/logit|vocab|candidate|next token|last logits/.test(text) && !["pretraining", "dpo"].includes(codeKey)) || codeKey === "generation") {
    return "용어 풀이: logits는 확률이 되기 전의 후보 점수이고, softmax를 거치면 확률처럼 해석할 수 있습니다.";
  }
  if (/embedding|hidden|block|gpt|transformer|model/.test(text) || codeKey === "gpt") {
    return "용어 풀이: embedding은 토큰 번호를 벡터로 바꾼 값이고, hidden state는 모델 내부에서 계속 갱신되는 중간 표현입니다.";
  }
  if (/loss|backward|optimizer|epoch|batch|target|train|validation/.test(text) || codeKey === "pretraining") {
    return "용어 풀이: loss는 틀린 정도, backward는 바꿀 방향 계산, optimizer step은 실제 값 수정입니다.";
  }
  if (/lora|rank|adapter|linear|freeze|frozen/.test(text) || codeKey === "lora") {
    return "용어 풀이: LoRA는 큰 가중치를 직접 바꾸지 않고 작은 보정 경로만 학습하는 방법입니다.";
  }
  if (/class|classification|accuracy/.test(text) || codeKey === "classification") {
    return "용어 풀이: class score는 각 라벨일 가능성을 나타내는 점수이고, 가장 큰 점수가 예측 라벨이 됩니다.";
  }
  if (/dpo|chosen|rejected|policy|reference|beta|preference/.test(text) || codeKey === "dpo") {
    return "용어 풀이: policy는 학습 중인 모델, reference는 비교용 고정 모델, chosen/rejected는 선호/비선호 답변입니다.";
  }
  if (/sft|instruction|prompt|response|label|-100/.test(text) || codeKey === "sft") {
    return "용어 풀이: prompt는 조건으로 주는 부분, response는 모델이 배워야 하는 답변 부분입니다.";
  }
  if (/token|dataset|input_ids/.test(text) || codeKey === "dataset") {
    return "용어 풀이: token은 문장을 모델이 다루기 쉬운 작은 조각으로 나눈 단위입니다.";
  }
  return "";
}

function detailForBox(node) {
  const label = shortText(node);

  if (node.matches(".pipeline span")) {
    const info = pipelineDetails[node.dataset.pipelineStage] || null;
    if (!info) return null;
    return { ...info, term: simpleTerm(label, info.codeKey) };
  }

  if (node.matches(".reference-card")) {
    return {
      title: label.split(" ").slice(0, 4).join(" "),
      body: "참고한 외부 시각화 자료입니다. 그림을 그대로 가져온 것이 아니라, 설명 순서와 상호작용 아이디어를 이 강의 흐름에 맞게 다시 구성했습니다.",
    };
  }

  if (node.matches(".token")) {
    return {
      title: `${label} token`,
      body: "이 박스는 문장 속 토큰 하나입니다. 선택된 토큰은 자기보다 앞에 있는 토큰들을 참고해, 문맥이 섞인 새 벡터로 바뀝니다.",
      codeKey: "attention",
    };
  }

  if (node.matches(".matrix-cell:not(.axis)")) {
    return {
      title: "Attention matrix cell",
      body: "행의 토큰이 열의 토큰을 얼마나 참고하는지 보여주는 칸입니다. 미래 토큰 칸은 가려져서 최종 참고 비율이 0이 됩니다.",
      codeKey: "attention",
    };
  }

  if (node.matches(".flow-node, .mini-block, .transformer-card, .block-steps li")) {
    return {
      title: label,
      body: "GPT가 입력을 처리하는 한 위치입니다. 토큰 번호가 벡터가 되고, 블록을 지나, 다음 토큰 후보 점수로 바뀌는 흐름을 보여줍니다.",
      codeKey: "gpt",
    };
  }

  if (node.matches(".generated-token, .loop-node, .bar-row")) {
    return {
      title: label,
      body: "생성 루프의 한 단계입니다. 현재 문맥에서 다음 토큰 후보 점수를 만들고, 설정에 따라 하나를 골라 문장 뒤에 붙입니다.",
      codeKey: "generation",
    };
  }

  if (node.matches(".train-node, .loss-row")) {
    return {
      title: label,
      body: "사전학습 루프의 한 단계입니다. 모델 예측과 다음 토큰 정답을 비교해 loss를 만들고, 그 loss가 줄어들도록 모델 값을 바꿉니다.",
      codeKey: "pretraining",
    };
  }

  if (node.matches(".finetune-node, .adapter-box")) {
    const codeKey = sectionCodeKey(node, label);
    return {
      title: label,
      body: codeKey === "lora"
        ? "LoRA는 기존 큰 가중치는 그대로 두고, 작은 두 행렬이 만든 보정값만 학습합니다. 그래서 적은 메모리로 모델을 조정할 수 있습니다."
        : "분류 미세튜닝은 GPT가 만든 문장 표현을 클래스별 점수로 바꾸고, 정답 라벨과 비교해 head를 학습합니다.",
      codeKey,
    };
  }

  if (node.matches(".align-node, .align-box, .preference-meter, .meter-row")) {
    const codeKey = sectionCodeKey(node, label);
    return {
      title: label,
      body: codeKey === "dpo"
        ? "DPO는 같은 질문에 대한 좋은 답변과 나쁜 답변을 비교해, 좋은 답변이 더 높은 점수를 받도록 학습 중인 모델을 조정합니다."
        : "SFT는 질문과 지시문은 조건으로만 사용하고, 모델이 실제로 써야 하는 답변 부분만 정답으로 두고 학습합니다.",
      codeKey,
    };
  }

  if (node.matches(".lens-node")) {
    return {
      title: label,
      body: "전체 흐름을 보는 관점을 바꾸는 노드입니다. 큰 그림, 텐서 모양, 학습 과정, 지시/선호 조정 중 어디에 집중할지 정합니다.",
      codeKey: sectionCodeKey(node, label),
    };
  }

  if (node.matches(".shape-box")) {
    return {
      title: label.split(" ").slice(0, 4).join(" "),
      body: "현재 섹션에서 먼저 확인해야 할 텐서 모양입니다. PyTorch 코드를 읽을 때 차원이 맞으면 연산 흐름을 훨씬 빨리 이해할 수 있습니다.",
      codeKey: sectionCodeKey(node, label),
    };
  }

  if (node.matches(".concept-note")) {
    return {
      title: "현재 상태 해설",
      body: "현재 슬라이더와 토글 값이 어떤 의미인지 풀어 쓴 설명입니다. 조작을 바꿨을 때 여기 문장이 같이 바뀌는지 보면 됩니다.",
      codeKey: sectionCodeKey(node, label),
    };
  }

  return null;
}

function refreshBoxDetails() {
  detailRefreshPending = false;
  const selector = [
    ".pipeline span",
    ".lens-node",
    ".token",
    ".matrix-cell:not(.axis)",
    ".flow-node",
    ".mini-block",
    ".transformer-card",
    ".block-steps li",
    ".generated-token",
    ".loop-node",
    ".bar-row",
    ".train-node",
    ".loss-row",
    ".finetune-node",
    ".adapter-box",
    ".align-node",
    ".align-box",
    ".preference-meter",
    ".meter-row",
    ".shape-box",
    ".concept-note",
    ".reference-card",
  ].join(", ");

  document.querySelectorAll(selector).forEach((node) => {
    const info = detailForBox(node);
    if (!info) return;
    node.classList.add("detail-enabled");
    node.dataset.infoTitle = info.title;
    node.dataset.infoBody = info.body;
    node.dataset.infoTerm = info.term || simpleTerm(node.innerText || "", info.codeKey || "");
    node.dataset.codeKey = info.codeKey || "";
    if (!node.matches("a, button, input, textarea, select") && !node.hasAttribute("tabindex")) {
      node.tabIndex = 0;
    }
  });
}

function scheduleDetailRefresh() {
  if (detailRefreshPending) return;
  detailRefreshPending = true;
  requestAnimationFrame(refreshBoxDetails);
}

function positionPopover(target) {
  const gap = 10;
  const rect = target.getBoundingClientRect();
  el.boxPopover.hidden = false;
  const popRect = el.boxPopover.getBoundingClientRect();
  let top = rect.top - popRect.height - gap;
  if (top < 8) top = rect.bottom + gap;
  let left = rect.left + rect.width / 2 - popRect.width / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - popRect.width - 8));
  el.boxPopover.style.top = `${Math.round(top)}px`;
  el.boxPopover.style.left = `${Math.round(left)}px`;
}

function showPopover(target) {
  if (!target?.classList?.contains("detail-enabled")) return;
  clearTimeout(hidePopoverTimer);
  activeDetailBox = target;
  el.boxPopoverTitle.textContent = target.dataset.infoTitle || "설명";
  el.boxPopoverBody.textContent = target.dataset.infoBody || "";
  el.boxPopoverTerm.textContent = target.dataset.infoTerm || "";
  el.boxPopoverTerm.hidden = !target.dataset.infoTerm;
  const hasExample = Boolean(target.dataset.codeKey && codeExamples[target.dataset.codeKey]);
  el.boxPopoverExample.hidden = !hasExample;
  el.boxPopoverExample.dataset.codeKey = hasExample ? target.dataset.codeKey : "";
  el.boxPopoverExample.title = hasExample ? "이 박스와 연결된 노트북 코드 일부를 봅니다." : "";
  positionPopover(target);
}

function schedulePopoverHide() {
  clearTimeout(hidePopoverTimer);
  hidePopoverTimer = setTimeout(() => {
    el.boxPopover.hidden = true;
    activeDetailBox = null;
  }, 120);
}

function openExample(codeKey) {
  const example = codeExamples[codeKey];
  if (!example) return;
  el.exampleModalTitle.textContent = example.title;
  el.exampleModalSource.textContent = example.source;
  el.exampleModalNote.textContent = example.note;
  el.exampleCode.textContent = example.code;
  el.exampleModal.hidden = false;
  document.body.classList.add("modal-open");
}

function closeExample() {
  el.exampleModal.hidden = true;
  document.body.classList.remove("modal-open");
}

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
  scheduleDetailRefresh();
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
  scheduleDetailRefresh();
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
  scheduleDetailRefresh();
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
    "모델은 각 위치마다 다음 토큰 후보 전체에 대한 점수를 계산합니다.",
    "다음 토큰은 마지막 위치의 점수에서 고릅니다. temperature는 후보 간 차이를 부드럽게 하거나 날카롭게 만들고, top-k는 남길 후보 수를 정합니다.",
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
  scheduleDetailRefresh();
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
    ["loss", "숫자 1개"],
  ]);
  scheduleDetailRefresh();
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
      <div class="adapter-box">GPT Backbone<br>문장 표현 계산</div>
      <div class="adapter-box trainable">Classification Head<br>emb_dim -> ${numClasses}</div>
      <div class="adapter-box">Class Scores<br>[batch, ${numClasses}]</div>
    `;
    el.finetuneNote.innerHTML = "<strong>분류 미세튜닝</strong><br>사전학습 GPT가 만든 문장 표현 위에 작은 Linear head를 붙이고, 클래스별 점수를 정답 라벨과 비교합니다.";
    setShapeList(el.finetuneShapes, [
      ["input_ids", "[batch, seq_len]"],
      ["문장 표현", "[batch, seq_len, emb_dim]"],
      ["class scores", `[batch, ${numClasses}]`],
      ["loss", "CrossEntropyLoss"],
    ]);
    scheduleDetailRefresh();
    return;
  }

  el.adapterVisual.innerHTML = `
    <div class="adapter-box${frozen ? " frozen" : ""}">Base Linear W<br>${frozen ? "frozen" : "trainable"}</div>
    <div class="adapter-box trainable">LoRA A/B<br>rank ${rank}</div>
    <div class="adapter-box">W(x) + LoRA(x)<br>작은 보정값 추가</div>
  `;
  el.finetuneNote.innerHTML = "<strong>LoRA 미세튜닝</strong><br>기존 큰 가중치는 그대로 두고, 작은 A/B 행렬만 학습해 출력에 보정값을 더합니다.";
  setShapeList(el.finetuneShapes, [
    ["base W", frozen ? "frozen" : "trainable"],
    ["LoRA A", `[emb_dim, ${rank}]`],
    ["LoRA B", `[${rank}, emb_dim]`],
    ["trainable params", "small adapter only"],
  ]);
  scheduleDetailRefresh();
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
    el.alignmentNote.innerHTML = "<strong>SFT</strong><br>질문과 지시문 부분은 label을 -100으로 두어 loss에서 빼고, 모델이 직접 써야 하는 답변 토큰만 학습합니다.";
    setShapeList(el.alignmentShapes, [
      ["input_ids", "[batch, seq_len]"],
      ["labels", "[batch, seq_len]"],
      ["ignored labels", `${ignoredRatio}% -> -100`],
      ["loss target", "response tokens only"],
    ]);
    scheduleDetailRefresh();
    return;
  }

  const chosenScore = Math.min(0.92, 0.52 + beta * 0.13);
  const rejectedScore = Math.max(0.08, 0.48 - beta * 0.11);
  el.alignmentVisual.innerHTML = `
    <div class="align-box focus">Chosen Response<br>좋은 답변 점수 ↑</div>
    <div class="align-box rejected">Rejected Response<br>나쁜 답변 점수 ↓</div>
    <div class="align-box${referenceFrozen ? " ignored" : ""}">Reference Model<br>${referenceFrozen ? "비교용 고정" : "함께 학습"}</div>
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
  el.alignmentNote.innerHTML = "<strong>DPO</strong><br>좋은 답변과 나쁜 답변의 상대 점수 차이를 비교해, 학습 중인 모델만 선호 답변 쪽으로 업데이트합니다.";
  setShapeList(el.alignmentShapes, [
    ["prompt", "[batch, prompt_len]"],
    ["chosen/rejected", "[batch, response_len]"],
    ["beta", beta.toFixed(1)],
    ["updated model", "학습 중인 모델만"],
  ]);
  scheduleDetailRefresh();
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

document.addEventListener("pointerover", (event) => {
  const target = event.target.closest(".detail-enabled");
  if (target) showPopover(target);
});

document.addEventListener("mouseover", (event) => {
  const target = event.target.closest(".detail-enabled");
  if (target) showPopover(target);
});

document.addEventListener("pointerout", (event) => {
  if (!activeDetailBox) return;
  const related = event.relatedTarget;
  if (activeDetailBox.contains(related) || el.boxPopover.contains(related)) return;
  schedulePopoverHide();
});

document.addEventListener("mouseout", (event) => {
  if (!activeDetailBox) return;
  const related = event.relatedTarget;
  if (activeDetailBox.contains(related) || el.boxPopover.contains(related)) return;
  schedulePopoverHide();
});

document.addEventListener("focusin", (event) => {
  const target = event.target.closest(".detail-enabled");
  if (target) showPopover(target);
});

document.addEventListener("focusout", (event) => {
  if (!activeDetailBox) return;
  const related = event.relatedTarget;
  if (activeDetailBox.contains(related) || el.boxPopover.contains(related)) return;
  schedulePopoverHide();
});

document.addEventListener("click", (event) => {
  const target = event.target.closest(".detail-enabled");
  if (!target || target.matches("a")) return;
  showPopover(target);
});

el.boxPopover.addEventListener("pointerenter", () => clearTimeout(hidePopoverTimer));
el.boxPopover.addEventListener("pointerleave", schedulePopoverHide);
el.boxPopoverExample.addEventListener("click", () => openExample(el.boxPopoverExample.dataset.codeKey));
el.exampleClose.addEventListener("click", closeExample);
el.exampleModal.addEventListener("click", (event) => {
  if (event.target.matches("[data-close-example]")) closeExample();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !el.exampleModal.hidden) closeExample();
});
window.addEventListener("scroll", () => {
  if (!el.boxPopover.hidden) schedulePopoverHide();
}, { passive: true });

renderLens();
renderAttention();
renderBlocks();
resetGeneration();
renderTraining();
renderFinetuning();
renderAlignment();
scheduleDetailRefresh();
