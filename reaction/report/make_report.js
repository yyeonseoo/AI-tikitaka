const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, LevelFormat,
  Footer, PageNumber, TableOfContents, PageBreak,
} = require('docx');

const FIG = 'C:/Users/USER/AI-tikitaka/reaction/figures/deck/';
const DEMO = 'C:/Users/USER/AppData/Local/Temp/claude/23a530e4af602772/30201f31-3981-4632-8cc0-0c28c862a7fe/scratchpad/demo_shot.png';
const OUT = process.argv[2];
const FONT = 'Malgun Gothic';
const W = 9638; // A4 내용 폭 (DXA, 여백 2cm)
const sizes = {
  'chem_bh.png': [2972, 728], 'interaction.png': [1310, 894], 'recommend.png': [1772, 722],
  'sequential.png': [1435, 844], 'shap_bh.png': [2190, 1338], 'shap_sm.png': [2263, 1338],
  'substrates.png': [1447, 945], 'topk.png': [1744, 908], 'demo': [1440, 1040],
};

const t = (text, o = {}) => new TextRun({ text, font: FONT, ...o });
const p = (text, o = {}) => new Paragraph({
  children: Array.isArray(text) ? text : [t(text)],
  spacing: { after: 120, line: 320 }, alignment: AlignmentType.JUSTIFIED, ...o,
});
const h1 = (s, brk = true) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: brk, children: [t(s)] });
const h2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(s)] });
const h3 = (s) => new Paragraph({ children: [t(s, { bold: true })], spacing: { before: 160, after: 80 }, keepNext: true });
const b = (s) => new Paragraph({ numbering: { reference: 'bul', level: 0 }, children: [t(s)], spacing: { after: 60, line: 300 } });
let figNo = 0, tabNo = 0;
function fig(name, caption, widthPx = 600) {
  const [w0, h0] = sizes[name];
  const data = fs.readFileSync(name === 'demo' ? DEMO : FIG + name);
  figNo++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data, transformation: { width: widthPx, height: Math.round(widthPx * h0 / w0) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [t(`그림 ${figNo}. `, { bold: true, size: 18 }), t(caption, { size: 18 })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, head, rows, ratio) {
  tabNo++;
  const tot = ratio.reduce((a, c) => a + c, 0);
  const cols = ratio.map((r) => Math.floor(W * r / tot));
  cols[cols.length - 1] += W - cols.reduce((a, c) => a + c, 0);
  const cell = (s, i, isHead) => new TableCell({
    borders, width: { size: cols[i], type: WidthType.DXA },
    shading: isHead ? { fill: 'E7E7E7', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [t(s, { size: 18, bold: isHead })], spacing: { line: 276 } })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, keepNext: true,
      children: [t(`표 ${tabNo}. `, { bold: true, size: 18 }), t(caption, { size: 18 })] }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: cols,
      rows: [new TableRow({ tableHeader: true, children: head.map((s, i) => cell(s, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((s, i) => cell(s, i, false)) }))] }),
    new Paragraph({ children: [], spacing: { after: 120 } }),
  ];
}
const br = () => new Paragraph({ children: [new PageBreak()] });

const cover = [
  new Paragraph({ children: [], spacing: { before: 2800 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [t('HTE 데이터 기반 반응조건 추천 연구 보고서', { size: 40, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1600 }, children: [t('팔라듐 교차 짝지음 반응의 수율 예측 및 반응 조건 우선순위 추천', { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t('팀명  윤멘토님과아이들', { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t('윤연서 (멘토) · 이경은 (멘티) · 김탁우 (멘티)', { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t('2026년 11월', { size: 22 })] }),
  br(),
  new Paragraph({ children: [t('목차', { size: 28, bold: true })], spacing: { after: 200 } }),
  new TableOfContents('목차', { hyperlink: true, headingStyleRange: '1-2' }),
];

const body = [
  h1('요약'),
  p('신약 공정 개발에서는 새 기질(substrate, 결합을 형성하는 두 반응물)이 도입될 때마다 리간드, 염기, 용매로 구성된 수백 가지 반응 조건 중 최적 조건을 찾기 위해 반복적인 스크리닝이 필요하다. 본 연구는 공개 고처리량 실험(HTE) 데이터 두 종(Buchwald–Hartwig 3,955건, Suzuki–Miyaura 5,760건)을 이용해, 미학습 기질(unseen substrate)에 대한 수율 예측과 반응 조건 우선순위 추천 모델을 개발하였다.'),
  p('모델은 시약별 분자 표현(RDKit 2D 기술자 + Morgan 지문)을 입력으로 하는 LightGBM 회귀 모델이며, 기질 단위 그룹 교차검증으로 평가하였다. 주요 결과는 다음과 같다.'),
  b('Suzuki–Miyaura 제로샷 조건 추천(대상 반응의 실험 데이터 없이 384개 조건 순위화)에서 상위 20개 안에 준최적 조건이 포함되는 비율(hit@20)은 87%로, 무작위(38%) 대비 크게 높았다. 동일한 분자 표현을 쓰는 능형 회귀 대비 regret@8이 평균 10.4%p 낮았다(95% 신뢰구간 [−20.3, −2.7]).'),
  b('Buchwald–Hartwig 미학습 기질에서 준최적 조건 도달까지의 기대 실험 수가 무작위 3.9회에서 1.3회로 약 65% 감소하였다.'),
  b('순차 모델 기반 최적화(SMBO)에서 준최적 조건 도달까지의 실험 수가 무작위 27.1회에서 13.7회로 감소하였고, 대상 반응의 실험만 사용하는 표준 베이지안 최적화(21.1회)보다 약 35% 적었다.'),
  b('SHAP 분석 결과, 모델이 의존한 특징은 산화적 첨가, 촉매 피독, 붕소 종의 활성화 등 알려진 반응 메커니즘과 일치하였다.'),
  p('이 모델을 추천 엔진으로 하는 웹 프로토타입 ReactPilot을 구현하였다.'),

  h1('1. 문제 정의'),
  h2('1.1 배경'),
  p('팔라듐 촉매 교차 짝지음 반응(cross-coupling)은 의약품 합성에서 C–N, C–C 결합 형성에 가장 널리 쓰이는 반응군이다. 동일한 반응이라도 기질이 바뀌면 최적의 리간드, 염기, 용매가 달라지며, 수율은 0%에서 90% 이상까지 분포한다.'),
  p('조건 공간(condition space)은 각 변수의 조합으로 증가한다. 본 연구에서 다룬 Suzuki–Miyaura 데이터는 리간드 12종, 염기 8종, 용매 4종으로 반응당 384개 조건을 갖는다. 제약 산업에서는 이를 96-웰 또는 384-웰 플레이트 기반 HTE로 스크리닝하지만, 기질 확보와 분석 비용이 발생하고 신규 기질마다 스크리닝을 반복해야 한다.'),
  h2('1.2 문제 정의'),
  p('본 연구의 핵심 질문은 다음과 같다. 미학습 기질이 주어졌을 때, 축적된 HTE 데이터만으로 어떤 반응 조건부터 실험해야 하는가. 이를 세 가지 과제로 정의하였다.'),
  ...table('과제 정의', ['과제', '입력', '출력', '적용 단계'], [
    ['수율 예측 (회귀)', '기질과 반응 조건', '예측 수율(%)', '실험 전 반응 가능성 판단'],
    ['제로샷 조건 추천', '미학습 기질 쌍', '전체 조건의 순위', '1차 스크리닝 플레이트 설계'],
    ['순차 최적화', '누적 실험 결과', '다음 배치 조건 4개', '후속 스크리닝 플레이트 설계'],
  ], [2.4, 2.6, 2.4, 3.2]),
  h2('1.3 평가 기준'),
  p('추천 모델의 가치는 실험 횟수 절감으로 측정된다. 따라서 수율 예측 오차(MAE)와 함께, 추천 상위 k개만 실험했을 때의 수율 손실(regret@k), 최적 조건 포함 확률(hit@k), 준최적 조건 도달까지의 실험 수를 주요 지표로 사용하였다. 비교 기준 모델(baseline)은 무작위 선택, 이력 기반 순위, 선형 회귀 모델, 표준 베이지안 최적화이다.'),

  h1('2. 관련 연구 및 기존 서비스'),
  h2('2.1 선행 연구'),
  p('Ahneman 등(2018)은 Buchwald–Hartwig 반응 4,608건을 HTE로 측정하고, 시약의 양자화학 기술자를 입력으로 한 랜덤 포레스트로 수율을 예측하였다. 이 데이터는 이후 반응 수율 예측의 표준 벤치마크가 되었다. Chuang과 Keiser(2018)는 동일 데이터에서 기술자를 무작위 특성으로 대체해도 유사한 성능이 나온다는 점을 근거로, 모델이 화학 정보를 실제로 활용하는지 검증해야 한다고 지적하였다.'),
  p('Perera 등(2018)은 Pfizer의 유동 반응기 기반 HTE 플랫폼으로 Suzuki–Miyaura 반응 5,760건을 측정하였다. Schwaller 등(2021)은 반응 SMILES를 입력으로 하는 트랜스포머 모델(Yield-BERT)로 두 데이터의 수율을 예측하였고, Shields 등(2021)은 베이지안 최적화가 숙련 화학자보다 적은 실험으로 최적 조건에 도달함을 보였다. Strieth-Kalthoff 등(2022)은 실패 반응이 보고되지 않는 문헌 데이터의 편향을 지적하며, 실패 결과까지 포함한 HTE 데이터의 가치를 강조하였다.'),
  p('최근에는 자동화 합성·분석 장비와 실험 선택 알고리즘을 결합해, 지금까지의 결과를 바탕으로 다음 실험을 스스로 결정하는 자율 실험실(Self-Driving Laboratory, SDL)이 화학·소재 연구에서 확산되고 있다(Coley 등, 2019; Burger 등, 2020). SDL은 실험을 수행하는 하드웨어 계층과 다음 실험을 고르는 의사결정 계층으로 구성되며, 의사결정 계층에는 베이지안 최적화(Bayesian optimization, BO)가 대표적으로 쓰인다. BO는 대리 모델(주로 가우시안 과정)로 미실험 조건의 결과와 불확실성을 예측하고, 획득 함수(acquisition function)로 다음 실험을 고른 뒤, 결과를 반영해 대리 모델을 갱신하는 과정을 반복한다. 본 연구의 순차 최적화는 이 의사결정 계층에 해당한다.'),
  ...table('선행 연구 비교', ['연구', '모델', '평가 방식', '본 연구와의 차이'], [
    ['Ahneman 등, 2018', '랜덤 포레스트 + 양자화학 기술자', '무작위 분할, 첨가제 단위 분할', '기질 단위 그룹 교차검증, 추천 지표 평가'],
    ['Schwaller 등, 2021', '트랜스포머(Yield-BERT)', '무작위 분할 중심', '추천 성능과 실험 절감량 직접 측정'],
    ['Shields 등, 2021', '베이지안 최적화(GP)', '순차 최적화', '과거 데이터 기반 제로샷 추천 포함'],
    ['본 연구', 'LightGBM + 분자 표현', '그룹 교차검증, 부트스트랩 신뢰구간', '수율 예측, 제로샷 추천, 순차 최적화를 단일 모델로 평가'],
  ], [2.2, 2.6, 2.6, 3.4]),
  p('선행 연구 역시 기존 알고리즘(랜덤 포레스트, 트랜스포머, 가우시안 과정)과 화학 표현을 결합해 모델을 구성하였다. 본 연구는 같은 접근을 따르되, 실제 사용 시나리오인 미학습 기질 조건 추천에 맞춘 평가 설계와 지표에서 차별화된다.'),
  h2('2.2 기존 서비스'),
  ...table('관련 서비스', ['서비스', '주요 기능', 'ReactPilot과의 차이'], [
    ['Yoneda Labs', '클라우드 기반 반응 조건 추천 및 최적화', '가장 유사한 서비스. ReactPilot은 검증 지표와 추천 근거를 함께 제시'],
    ['Atinary SDLabs', '자율 실험실 연동 베이지안 최적화 플랫폼', '자동화 장비 전제. ReactPilot은 기존 HTE 플레이트 워크플로에 바로 적용'],
    ['SYNTHIA (Merck)', '역합성 경로 설계', '반응 경로 설계 도구. 조건 최적화는 범위 밖'],
    ['Reaxys, SciFinder', '문헌 반응 데이터베이스', '보고된 성공 조건 검색. 실패 데이터 기반 예측 없음'],
    ['IBM RXN', '생성물 예측, 역합성', '생성물 구조 예측. 수율 및 조건 순위는 다루지 않음'],
  ], [2.2, 3.6, 4.6]),

  h1('3. 연구 방법'),
  h2('3.1 용어 정의'),
  ...table('주요 용어', ['용어', '정의'], [
    ['HTE (High-Throughput Experimentation, 고처리량 실험)', '96-웰, 384-웰 플레이트에 조건 조합을 배치해 수백 개 반응을 병렬로 수행하는 실험 방식'],
    ['수율 (yield)', '투입한 기질 대비 목적 생성물로 전환된 비율(%)'],
    ['기질 (substrate)', '결합을 형성하는 두 반응물. BH는 할로젠화 아릴과 아민, SM은 친전자체와 친핵체'],
    ['리간드, 염기, 용매, 첨가제', '팔라듐 촉매에 배위해 반응성을 조절하는 분자, 반응을 촉진하는 염기, 반응 매질, 작용기 내성 평가용으로 투입한 분자'],
    ['분자 표현 (molecular representation)', '분자 구조를 수치 벡터로 변환한 것. 본 연구는 시약마다 RDKit 2D 기술자 12종과 Morgan 지문 64비트를 사용'],
    ['분자 기술자 (descriptor)', '분자량, 위상 극성 표면적(TPSA) 등 분자 구조로부터 계산한 물리화학적 수치'],
    ['Morgan 지문 (fingerprint)', '분자 내 부분 구조(substructure)의 존재 여부를 비트(0/1)로 나타낸 벡터. 반경 2(ECFP4에 해당)'],
    ['LightGBM', '그래디언트 부스팅 결정 트리(GBDT) 구현체. 특성 간 비선형 상호작용을 학습'],
    ['시드 앙상블 (seed ensemble)', '난수 시드만 달리해 학습한 동일 구조 모델 여러 개의 예측을 평균하는 방법. 예측 분산을 줄임'],
    ['MAE (평균 절대 오차)', '예측 수율과 실제 수율 차이의 절댓값 평균(%p)'],
    ['regret@k', '해당 반응의 최고 수율 − 추천 상위 k개 중 최고 수율(%p). 상위 k개만 실험했을 때의 수율 손실'],
    ['hit@k', '추천 상위 k개 안에 최적(또는 준최적) 조건이 포함될 확률'],
    ['준최적 조건 (near-optimal)', '해당 반응의 최고 수율과 5%p 이내의 조건'],
    ['그룹 교차검증 (Leave-One-Group-Out)', '특정 기질(쌍)의 모든 반응을 학습에서 제외하고 그 기질로 평가하는 방식을 모든 기질에 반복. 미학습 기질 상황을 재현'],
    ['부트스트랩 95% 신뢰구간', '평가 그룹을 복원 추출로 1,000회 재표집해 구한 성능 차이의 분포 범위. 구간이 0을 포함하지 않으면 유의한 차이로 판단'],
    ['SHAP (SHapley Additive exPlanations)', '개별 예측을 각 특성의 기여도로 분해하는 모델 해석 방법'],
  ], [3.2, 6.8]),
  h2('3.2 데이터'),
  p('두 데이터 모두 원 연구에서 시약 조합을 전수(full factorial) 실험한 HTE 데이터로, 수율 0%의 실패 반응까지 모두 포함한다. 성공 사례 위주로 보고되는 문헌 데이터와 달리 조건 공간 전체의 정답이 존재하므로, 조건 추천 모델의 학습과 평가에 적합하다.'),
  ...table('데이터 구성', ['데이터', '형성 결합', '변수', '반응 수'], [
    ['Buchwald–Hartwig (BH), Ahneman 등, 2018', 'C–N', '할로젠화 아릴 15종(아렌 5종 × Cl·Br·I), 리간드 4, 염기 3, 첨가제 22', '3,955'],
    ['Suzuki–Miyaura (SM), Perera 등, 2018', 'C–C', '친전자체 7, 친핵체 4(기질 쌍 15개), 리간드 12, 염기 8, 용매 4', '5,760'],
  ], [3.2, 1.4, 4.4, 1.2]),
  p('두 데이터는 실험 설계가 달라 서로 다른 평가에 적합하다. BH는 아민(p-톨루이딘)을 고정하고 할로젠화 아릴만 아렌 5종 × 이탈기 3종의 격자로 바꾸었으므로, 기질 구조 변화에 따른 수율 변화를 체계적으로 관찰할 수 있고 학습에 전혀 등장하지 않는 기질을 깔끔하게 분리할 수 있다. 대신 조건 공간은 12개(리간드 4 × 염기 3)로 작다. SM은 리간드·염기·용매를 모두 바꾼 384개 조건으로, 실제 384-웰 플레이트 스크리닝과 같은 규모이다. 이에 따라 미학습 기질의 수율 예측은 BH에서, 넓은 조건 공간에서의 조건 추천과 순차 최적화는 SM에서 주로 평가하였다.'),
  ...fig('substrates.png', 'BH 기질별 평균 수율. 이탈기(열)와 아렌(행)에 따라 평균 수율이 2%에서 59%까지 달라진다.', 470),
  p('수율 변동의 구조를 확인하기 위해 전체 데이터에 두 가지 선형 모델을 적합하고 결정계수(R²)를 비교하였다. 두 모델 모두 시약 정체를 원-핫 인코딩한 변수만 사용하였다.'),
  b('주효과 모델(main-effect model): 각 시약의 효과가 독립적으로 더해진다고 가정한다. 리간드의 효과는 기질과 무관하게 일정하다.'),
  b('2차 상호작용 모델(two-way interaction model): 주효과에 모든 시약 쌍의 상호작용 항을 추가한다. 예를 들어 특정 기질과 특정 리간드의 조합에 고유한 보정값이 존재한다.'),
  p('상호작용 항을 추가하면 R²가 BH 0.71에서 0.90으로, SM 0.51에서 0.78로 증가하였다. 즉 시약 간 상호작용이 수율 변동의 상당 부분을 설명하며, 비선형 상호작용을 학습할 수 있는 모델이 필요하다. 다만 원-핫 기반 상호작용 모델은 학습에 등장한 시약에만 계수를 가지므로, 학습에 한 번도 등장하지 않은 기질에는 적용할 수 없다. 따라서 시약을 정체가 아닌 구조 기반 분자 표현으로 입력해야 한다.'),
  ...fig('interaction.png', '주효과 모델과 2차 상호작용 모델의 결정계수(R²) 비교.', 400),
  h2('3.3 모델'),
  p('모델은 하나의 반응(시약 조합)을 입력받아 수율을 출력하는 회귀 모델이다. SM을 기준으로 구성은 다음과 같다.'),
  b('입력: 친전자체, 친핵체, 리간드, 염기, 용매 5종 시약 각각의 분자 표현. 시약마다 RDKit 2D 기술자 12종(분자량, TPSA, 회전 가능 결합 수 등)과 Morgan 지문 64비트(C–Cl, 피리딘 질소 등 부분 구조의 존재 여부)를 계산해 연결(concatenate)하고, 데이터 내 분산이 0인 특성을 제거하여 반응당 약 210차원 특성 벡터를 구성한다.'),
  b('출력: 예측 수율(%).'),
  p('시약을 원-핫 인코딩이 아닌 구조 기반 표현으로 입력하므로, 미학습 기질도 구조적으로 유사한 학습 시약의 정보를 활용해 예측할 수 있다. LightGBM은 "C–Cl 부분 구조가 있고 리간드가 XPhos이면 수율이 낮다"와 같은 특성 간 조건부 규칙을 트리 분기로 학습하므로, 3.2절에서 확인한 시약 간 상호작용을 표현할 수 있다.'),
  p('모델 선정을 위한 1차 비교에서 BH 미학습 기질의 MAE는 LightGBM 13.1%p, 다층 퍼셉트론(MLP) 12.8%p, 랜덤 포레스트 14.3%p로 신뢰구간이 크게 겹쳐 유의한 차이가 없었다. 정확도가 비슷하므로, 순차 최적화에서 수천 회 반복되는 재학습의 속도, 트리 모델에 대한 SHAP 해석의 용이성, 적은 튜닝으로 얻는 결과의 안정성을 고려해 LightGBM을 채택하였다.'),
  p('예측 모델은 동일 하이퍼파라미터에 난수 시드만 달리한 LightGBM 5개의 시드 앙상블이다. 중첩 교차검증(nested cross-validation)으로 하이퍼파라미터 후보 30개를 탐색한 결과 성능 차이가 0.2%p 수준에 그쳐, 고정 하이퍼파라미터를 최종 모델로 채택하였다.'),
  p('추천은 예측 수율을 기준으로 한다. 제로샷 추천에서는 대상 반응의 모든 조건에 대해 수율을 예측하고 내림차순으로 순위화한다. 순차 최적화는 순차 모델 기반 최적화(Sequential Model-Based Optimization, SMBO)로 구성하였다. SDL의 의사결정 계층과 같은 폐루프 구조(예측 → 실험 → 재학습)에서 대리 모델(surrogate model)로 LightGBM 앙상블을 사용하며, 매 배치마다 누적 결과로 재학습한 뒤 미실험 조건 중 예측 수율 상위 4개를 다음 배치로 선택한다(탐욕적 획득). 불확실성을 활용하는 UCB 획득(예측 평균 + 앙상블 표준편차)도 비교하였다.'),
  p('대리 모델로 가우시안 과정 대신 LightGBM을 사용한 것은, 다른 기질로 축적된 수천 건의 HTE 데이터로 사전 학습하기 위해서이다. 가우시안 과정은 학습 데이터 수의 세제곱에 비례해 계산량이 증가하여, 과거 데이터 5천여 건으로 학습할 때 대상 반응 하나의 순차 최적화에 수 분이 걸린 반면 LightGBM은 수 초였다. 실험은 연구자가 수행하고 결과를 입력하는 human-in-the-loop 방식으로, 자동화 장비 없이 기존 HTE 워크플로에 적용할 수 있다.'),
  ...table('최종 모델 구성', ['구성 요소', '내용'], [
    ['입력', '기질, 리간드, 염기, 용매(BH는 첨가제)의 분자 표현. 시약당 RDKit 2D 기술자 12 + Morgan 지문 64비트'],
    ['예측 모델', 'LightGBM 회귀, 시드 앙상블 5개'],
    ['출력', '조건별 예측 수율 및 순위'],
    ['순차 최적화', 'SMBO, 탐욕적 획득. 초기 4개 무작위 실험 후 배치 크기 4로 9회 반복(총 40회), 배치마다 재학습'],
  ], [2.5, 7.5]),
  h2('3.4 평가 방법'),
  p('실제 사용 시나리오는 미학습 기질에 대한 추천이므로, 모든 평가는 그룹 교차검증으로 수행하였다. BH는 기질 15종 각각을 시험 그룹으로 두고 나머지 14종으로 학습하였다(15-fold). SM은 기질 쌍 15개 각각을 시험 그룹으로 두고 나머지 14쌍(5,376건)으로 학습하였다. 방법 간 성능 차이는 시험 그룹 단위 부트스트랩(1,000회) 95% 신뢰구간으로 판단하였다.'),
  ...table('비교 기준 모델', ['모델', '설명'], [
    ['무작위 (random)', '조건을 무작위로 선택. 기댓값을 해석적으로 계산'],
    ['조건 평균 기준선', '학습 기질에서 같은 조건의 평균 수율로 예측. 기질 정보 미사용'],
    ['이력 기반 순위 (historical ranking)', '과거 데이터에서 평균 수율이 높았던 조건 순으로 추천'],
    ['능형 회귀 + 분자 표현', '최종 모델과 같은 분자 표현을 쓰는 선형 모델(Ridge)'],
    ['상호작용 선형 모델', '원-핫 인코딩한 시약과 모든 2차 상호작용 항을 쓰는 베이지안 능형 회귀'],
    ['무작위 특성 대조군 (random-feature control)', '분자 표현을 같은 차원의 고정된 무작위 비트로 대체한 LightGBM'],
    ['표준 베이지안 최적화 (GP + UCB)', '가우시안 과정(Matern 5/2 커널) 대리 모델과 UCB 획득 함수. 대상 반응의 실험 결과만으로 학습하는 일반적인 BO 설정. 순차 최적화에서 비교'],
  ], [3.4, 6.6]),

  h1('4. 결과'),
  h2('4.1 수율 예측'),
  p('학습에 등장한 시약의 새로운 조합(무작위 분할)에서 LightGBM의 MAE는 BH 4.0%p(R² 0.96), SM 6.8%p로, 조건 평균 기준선(BH 6.8, SM 10.8)보다 크게 낮았다.'),
  p('BH는 아렌 5종 × 이탈기 3종의 격자 구조이므로, 기질 하나를 제외해도 그 기질의 아렌과 이탈기는 각각 다른 기질로 학습 데이터에 등장한다. 이 조건에서 미학습 기질의 오차가 감소하였다. 조건 평균 기준선은 18.1%p, 동일 분자 표현의 능형 회귀는 15.8%p, LightGBM은 11.8%p였다. LightGBM과 능형 회귀의 차이는 −4.0%p(95% 신뢰구간 [−6.4, −1.3])로 유의하였다. 분자 표현과 비선형 상호작용을 학습하는 트리 모델이 모두 성능에 기여하였다.'),
  p('반면 SM의 친전자체 자리 시약 7종은 같은 퀴놀린 골격에 6번 위치 작용기(Cl, Br, I, OTf와 붕소 작용기 3종)만 다르므로, 하나를 제외하면 해당 작용기가 학습 데이터에 전혀 등장하지 않는다. 이 조건에서는 LightGBM과 상호작용 선형 모델의 MAE가 비슷하였다(19.0 vs 18.7%p, 차이 +0.3%p, 95% 신뢰구간 [−1.9, +2.2]). 즉 모델은 기질의 구조 요소가 학습 데이터에 포함된 범위에서 일반화한다. 4.2와 4.3의 SM 평가는 이 범위, 즉 시험 기질 쌍의 친전자체와 친핵체가 각각 다른 쌍으로 학습에 등장하는 조건에서 수행하였다.'),
  ...table('BH 미학습 기질의 수율 예측 오차', ['모델', '입력', 'MAE (%p)'], [
    ['조건 평균 기준선', '조건 정보만 사용', '18.1'],
    ['능형 회귀', '분자 표현', '15.8'],
    ['LightGBM', '분자 표현', '11.8'],
  ], [3, 5, 2]),
  h2('4.2 제로샷 조건 추천'),
  p('미학습 기질에 대해 대상 반응의 실험 데이터 없이 모든 조건의 수율을 예측하고, 예측 수율 순으로 조건을 순위화하는 제로샷 추천을 두 데이터에서 평가하였다. 조건 공간은 BH 12개, SM 384개이다.'),
  h3('(1) Buchwald–Hartwig: 조건 12개'),
  p('BH 미학습 기질마다 리간드 4종 × 염기 3종 = 12개 조건을 순위화하고, (기질, 첨가제) 330개 상황에서 추천 성능을 평가하였다. hit@3은 85%로 무작위(25%) 대비 크게 높았고, 준최적 조건 도달까지의 기대 실험 수는 무작위 3.9회에서 1.3회로 약 65% 감소하였다.'),
  p('Chuang과 Keiser(2018)의 지적에 따라 무작위 특성 대조군과 비교하였다. 대조군의 hit@3은 74%로 낮아졌고, regret@3 차이는 +0.54%p(95% 신뢰구간 [+0.18, +0.92])로 분자 표현이 유의하게 우수하였다. 모델이 무작위 식별자가 아닌 화학 구조 정보를 활용함을 확인하였다.'),
  ...fig('topk.png', 'BH 미학습 기질에서 추천 상위 k개 안에 최적 조건이 포함될 확률(hit@k).', 560),
  h3('(2) Suzuki–Miyaura: 조건 384개'),
  p('BH의 12개 조건은 무작위로 골라도 평균 3.9회 안에 준최적 조건에 도달하는 규모이다. 실제 HTE 스크리닝은 플레이트 단위로 수백 개 조건을 다루므로, 이 규모의 추천 성능은 SM에서 확인하였다. SM은 조건이 384개(리간드 12 × 염기 8 × 용매 4)로 32배 넓고, 주효과 모델의 설명력이 낮아(R² 0.51) 시약 간 상호작용이 더 중요하다. 미학습 기질 쌍마다 384개 조건을 순위화하였다.'),
  p('SM의 평가 단위는 기질 쌍이므로, 시험 쌍의 친전자체와 친핵체는 각각 다른 쌍의 반응으로 학습 데이터에 등장한다. 따라서 원-핫 기반 상호작용 선형 모델도 비교에 포함할 수 있다.'),
  ...table('SM 제로샷 조건 추천 성능 (기질 쌍 15개 평균)', ['모델', 'regret@1 (%p)', 'regret@8 (%p)', 'hit@8 (준최적)', 'hit@20 (준최적)'], [
    ['무작위', '50.0', '19.7', '19%', '38%'],
    ['이력 기반 순위', '23.5', '13.9', '47%', '60%'],
    ['능형 회귀 + 분자 표현', '24.3', '15.3', '40%', '53%'],
    ['상호작용 선형 모델', '14.8', '6.3', '60%', '80%'],
    ['LightGBM + 분자 표현', '10.6', '5.0', '60%', '87%'],
  ], [3.2, 1.6, 1.6, 1.9, 1.9]),
  p('LightGBM의 regret@8은 동일 분자 표현의 능형 회귀보다 평균 10.4%p 낮았다(95% 신뢰구간 [−20.3, −2.7]). 상호작용 선형 모델 대비 −1.3%p([−3.0, −0.3]), 이력 기반 순위 대비 −8.9%p([−17.3, −2.3]), 무작위 대비 −14.7%p([−19.8, −9.7])로 모든 비교에서 유의하게 우수하였다.'),
  ...fig('recommend.png', 'SM 미학습 기질 쌍에서의 hit@20(상위 20개 안에 준최적 조건 포함 비율).', 560),
  h2('4.3 SM 순차 최적화'),
  p('실제 스크리닝은 1차 플레이트 결과를 반영해 후속 플레이트를 설계한다. 이 과정을 나머지 14쌍의 과거 데이터가 주어진 조건(warm start)에서 재현하였다. 초기 4개 조건을 무작위로 실험한 뒤, 배치 크기 4로 9회(총 40회) SMBO를 수행하였다. BH는 배치 크기 4 기준 3회 만에 12개 조건이 모두 소진되어 반복 최적화의 여지가 없으므로, 순차 최적화는 SM에서만 평가하였다.'),
  p('준최적 조건 도달까지의 실험 수는 무작위 27.1회, 이력 기반 순위 16.5회, LightGBM 13.7회였다. 누적 20회 시점의 regret은 무작위 대비 8.3%p([−11.6, −5.1]), 이력 기반 순위 대비 6.5%p([−14.2, −0.6]) 낮았다. 이력 기반 순위는 기질과 무관하게 같은 조건을 우선 추천하므로, 최적 조건이 다른 염화 아릴 기질에서 큰 손실을 보였다. LightGBM은 초기 배치 결과를 반영한 직후(5회차)부터 regret을 빠르게 줄였다.'),
  p('초기 4개를 무작위 대신 제로샷 추천 상위 4개로 고르면, 준최적 조건까지의 실험 수가 11.1회로 2.6회 더 감소하였다(95% 신뢰구간 [−3.6, −1.4]). 누적 20회 시점의 regret은 차이가 없었다(−0.2%p, [−0.4, +0.1]). 즉 제로샷 추천으로 시작하면 좋은 조건을 더 일찍 찾고, 20회 이후의 최종 도달 수준은 같다. 6장의 프로토타입은 이 방식을 사용한다.'),
  p('표준 베이지안 최적화와도 비교하였다. 가우시안 과정과 UCB로 대상 반응의 실험 결과만 사용하면 준최적 조건까지 21.1회가 필요하였고, 축적 데이터로 사전 학습한 LightGBM은 이보다 7.4회(약 35%) 적었다(95% 신뢰구간 [−10.6, −4.0]). 같은 과거 데이터를 가우시안 과정에 제공하면 15.4회로, LightGBM과의 차이는 유의하지 않았다(−1.7회, [−5.1, +0.9]). 즉 순차 최적화의 성능 향상은 대리 모델의 종류보다 축적 HTE 데이터의 활용에서 비롯되었으며, LightGBM은 같은 수준의 성능을 짧은 학습 시간으로 제공한다. UCB와 탐욕적 획득의 차이는 가우시안 과정(21.1회 vs 21.4회, 유의하지 않음)과 LightGBM(13.2회 vs 13.7회) 모두에서 작아, 이 조건 공간에서는 불확실성 기반 탐색의 추가 이득이 작았다.'),
  ...table('SM 순차 최적화 방법 비교 (15쌍 × 시작 시드 10개 평균)', ['방법', '과거 데이터', '준최적 조건까지 실험 수', 'regret@20 (%p)'], [
    ['무작위', '–', '27.1', '11.6'],
    ['표준 베이지안 최적화 (GP + UCB)', '미사용', '21.1', '8.5'],
    ['이력 기반 순위', '사용', '16.5', '9.8'],
    ['가우시안 과정 + 탐욕적 획득', '사용', '15.4', '4.0'],
    ['LightGBM + 탐욕적 획득', '사용', '13.7', '3.3'],
  ], [3.8, 1.8, 2.4, 2.0]),
  ...fig('sequential.png', 'SM 순차 최적화에서 누적 실험 수에 따른 regret(15쌍 평균). 낮을수록 좋다.', 520),

  h1('5. 모델 해석'),
  p('추천 결과를 실험 현장에서 활용하려면 모델의 판단 근거가 화학적으로 타당해야 한다. SHAP으로 특성별 기여도를 계산하고, 시약별 평균 수율과 함께 알려진 반응 메커니즘과 비교하였다.'),
  h2('5.1 Buchwald–Hartwig'),
  ...fig('shap_bh.png', 'BH 모델의 SHAP 요약. 상위 특성일수록 예측에 대한 기여가 크다.', 560),
  ...fig('chem_bh.png', 'BH 시약별 평균 수율(이탈기, 리간드, 염기).', 620),
  ...table('BH 모델 해석과 화학적 근거', ['데이터 및 SHAP 관찰', '화학적 해석'], [
    ['기질 특성의 기여가 가장 큼. 상위 부분 구조는 방향족 C–Cl(평균 수율 6% 대 40%)과 메톡시기(20% 대 36%)', '촉매 순환의 첫 단계인 산화적 첨가(oxidative addition)는 C–Cl 결합에서 가장 느리고, 전자 공여성 메톡시기가 이를 더 늦춘다. 4-클로로아니솔의 평균 수율은 1.9%'],
    ['같은 C–Cl이라도 피리딘 고리는 수율이 높음(2-클로로피리딘 43.5%)', '전자 결핍성 헤테로아렌은 C–Cl 결합의 산화적 첨가를 촉진한다'],
    ['리간드 중 XPhos가 가장 낮음(20% 대 34~41%)', '이 반응에서는 tert-부틸기 등 입체적으로 크고 전자가 풍부한 포스핀 리간드가 유리하다'],
    ['에스터 치환 아이소옥사졸 첨가제가 가장 불리(9~20%)', '아이소옥사졸의 N–O 결합에 팔라듐이 산화적 첨가하며 촉매가 피독된다(Ahneman 등, 2018). 전자 끄는 에스터기가 이 부반응을 촉진한다'],
  ], [5, 5]),
  h2('5.2 Suzuki–Miyaura'),
  ...fig('shap_sm.png', 'SM 모델의 SHAP 요약.', 560),
  ...table('SM 모델 해석과 화학적 근거', ['데이터 및 SHAP 관찰', '화학적 해석'], [
    ['이탈기별 평균 수율 I 59.5% > OTf 44.6% > Br 41.5% > Cl 22.1%', '산화적 첨가의 반응성 순서(I > OTf > Br > Cl)와 일치한다'],
    ['SHAP 1위는 친핵체의 TPSA로, 붕소 종을 구분하는 역할. BF₃K 염(25.5%)이 보론산(52.2%)보다 크게 낮음', '트라이플루오로보레이트는 금속 교환(transmetalation) 전에 보론산으로 가수분해되어야 하므로 이 조건에서 느리다'],
    ['두자리(bidentate) 리간드(Xantphos, dppf)가 수율을 낮춤', '두자리 리간드는 팔라듐 착물의 배위 구조를 바꾼다. 이 조건에서는 한자리 리간드인 PPh₃(53%)가 가장 우수하고 Xantphos(16%)가 가장 불리하였다'],
  ], [5, 5]),
  p('모델이 의존한 특징은 산화적 첨가, 촉매 피독, 붕소 종 활성화라는 알려진 메커니즘과 일치하였다. 이는 모델의 추천이 우연한 상관이 아닌 화학적으로 타당한 정보에 기반함을 뒷받침하며, 서비스에서 추천 근거를 제시하는 기반이 된다.'),

  h1('6. 서비스 프로토타입: ReactPilot'),
  p('연구 결과를 실제 스크리닝 워크플로로 구현한 웹 프로토타입이다. 실제 플레이트 스크리닝과 같은 384개 조건 공간을 갖는 Pfizer 공개 SM 데이터 15쌍을 대상으로, 사용자가 반응을 선택하면 모델이 384개 조건의 순위를 제시하고 추천 순서대로 실험을 진행하는 과정을 재현한다. 실험 결과는 데이터에 기록된 실측 수율을 사용한다.'),
  ...fig('demo', 'ReactPilot 화면. 반응 선택, 시약 3D 구조와 추천 순위, 진행 현황과 검증 지표.', 600),
  ...table('ReactPilot 입출력', ['구분', '내용'], [
    ['사용자 입력', '반응(기질 쌍) 선택, 다음 배치 실행, 3D 구조 확인할 시약 선택'],
    ['추천 순위', '384개 조건(리간드 × 염기 × 용매)의 예측 수율 순위. 배치 결과 반영 시 재계산'],
    ['실험 기록 및 수렴 곡선', '실험 조건별 실측 수율, 현재 최고 수율, 무작위 선택과의 비교'],
    ['적용 영역 (applicability domain)', '학습 데이터에 동일 기질을 포함한 반응 수 표시'],
    ['검증 지표', '미학습 반응 15개에서 측정한 추천 성능(hit@20 87% 등)'],
  ], [2.8, 7.2]),
  p('예시 화면의 6-클로로퀴놀린 + 인다졸 보론산 반응에서는 첫 번째 추천 실험에서 최고 수율(93.7%) 대비 5%p 이내의 조건을 찾았다. 같은 반응에서 무작위 선택 시 기대 실험 수는 43회이다.'),

  h1('7. 차별점 및 활용 방안', false),
  h2('7.1 차별점'),
  b('검증 투명성: 미학습 반응 기준의 추천 성능과 신뢰구간을 사용자에게 공개한다. 기존 상용 서비스는 대체로 내부 검증 지표를 공개하지 않는다.'),
  b('설명 가능성: SHAP 기반 해석이 알려진 반응 메커니즘과 일치함을 확인하였으며, 이를 추천 근거로 제시할 수 있다.'),
  b('경량 구조: GBDT 앙상블로 일반 PC에서 학습과 추론이 가능하다. 데이터 보안이 중요한 제약사가 온프레미스로 설치해 자사 HTE 데이터로 재학습할 수 있다.'),
  b('워크플로 호환성: 자동화 장비 없이 기존 96-웰, 384-웰 플레이트 기반 HTE에 추천 순위만 추가하면 된다.'),
  h2('7.2 활용 방안'),
  p('주요 고객은 제약사와 위탁개발생산(CDMO) 기업의 공정 개발 조직이다. 신규 기질에 대해 1차 스크리닝 조건을 추천받고, 결과를 입력해 후속 배치를 추천받는 방식으로 활용한다. 본 연구 결과 기준으로 BH에서는 준최적 조건 도달까지의 실험이 약 65%, SM 순차 최적화에서는 약 50% 감소한다. 사내 축적 HTE 데이터로 재학습하면 각 기업의 주력 반응에 특화된 추천이 가능하다.'),
  p('추천에 필요한 학습 데이터는 신규 기질을 위해 새로 수행하는 실험이 아니라, 기업이 이전 프로젝트에서 이미 수행한 스크리닝 기록이다. 기록 규모에 따른 효과를 확인하기 위해, SM에서 학습에 사용하는 과거 스크리닝 기록(기질 쌍 하나의 384개 조건 전수 스크리닝 = 1건)을 1~14건으로 줄여 제로샷 추천을 반복하였다. 지표는 추천 순서대로 실험할 때 준최적 조건에 처음 도달하는 실험 수이다.'),
  ...table('과거 스크리닝 기록 규모에 따른 SM 제로샷 추천 성능 (기질 쌍 15개 평균)', ['과거 스크리닝 기록', '반응 수', '준최적 조건까지 실험 수', '무작위 대비 감소', 'hit@20'], [
    ['1건', '384', '59', '26%', '52%'],
    ['4건', '1,536', '52', '34%', '55%'],
    ['7건', '2,688', '43', '46%', '68%'],
    ['10건', '3,840', '41', '49%', '75%'],
    ['14건', '5,376', '30', '63%', '87%'],
  ], [2.2, 1.6, 2.6, 2.0, 1.6]),
  p('같은 지표에서 무작위 선택은 79회, 이력 기반 순위는 53회(14건 기준)였다. 기존 스크리닝 기록 14건을 재활용하면 신규 기질마다 전수 스크리닝(384회) 대비 약 92%, 이력 기반 순위 대비 약 44%의 실험을 줄일 수 있다. 무작위 대비 차이는 기록 7건부터, 이력 기반 순위 대비 차이는 14건에서 유의하였으며(15쌍 부트스트랩 95% 신뢰구간), 14건까지 성능이 계속 향상되어 기록이 쌓일수록 효과가 커질 것으로 기대된다. 이 결과는 동일한 시약 목록과 실험 플랫폼을 전제로 하므로, 기업별 데이터에서의 효과는 전향적 검증으로 확인해야 한다.'),

  h1('8. 결론 및 향후 계획', false),
  p('본 연구는 공개 HTE 데이터 두 종으로 미학습 기질의 수율 예측과 반응 조건 우선순위 추천 모델을 개발하고, 이를 웹 프로토타입으로 구현하였다. 분자 표현 기반 LightGBM은 동일 표현의 선형 모델 대비 수율 예측(BH MAE 15.8 → 11.8%p)과 제로샷 추천(SM regret@8 15.3 → 5.0%p)에서 유의하게 우수하였고, 무작위 대비 준최적 조건 도달까지의 실험 수를 BH 65%, SM 순차 최적화 약 50% 줄였다. 순차 최적화는 표준 베이지안 최적화보다도 약 35% 적은 실험으로 준최적 조건에 도달하여, SDL의 실험 선택 계층에 축적 HTE 데이터를 활용하는 방식의 효과를 보였다. 모델의 판단 근거는 알려진 반응 메커니즘과 일치하였다.'),
  p('향후 계획은 다음과 같다.'),
  b('학습에 없는 구조 요소(새로운 이탈기 등)를 가진 기질로의 일반화를 위해, 기질 다양성이 큰 데이터로 학습 범위 확장'),
  b('타 기관 HTE 데이터 및 다른 반응 유형(아마이드 결합 형성 등)으로 확장'),
  b('실제 스크리닝에 추천 순위를 적용한 전향적 검증(prospective validation)'),
  b('자동 합성·분석 장비와 연동해 SDL의 실험 선택 계층으로 확장'),
  b('사용자 데이터 업로드 및 재학습 기능 구현'),

  h1('참고문헌', false),
  ...[
    'Ahneman, D. T., Estrada, J. G., Lin, S., Dreher, S. D., Doyle, A. G. (2018). Predicting reaction performance in C–N cross-coupling using machine learning. Science, 360, 186–190.',
    'Chuang, K. V., Keiser, M. J. (2018). Comment on "Predicting reaction performance in C–N cross-coupling using machine learning". Science, 362, eaat8603.',
    'Perera, D., Tucker, J. W., Brahmbhatt, S., et al. (2018). A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science, 359, 429–434.',
    'Schwaller, P., Vaucher, A. C., Laino, T., Reymond, J.-L. (2021). Prediction of chemical reaction yields using deep learning. Machine Learning: Science and Technology, 2, 015016.',
    'Shields, B. J., Stevens, J., Li, J., et al. (2021). Bayesian reaction optimization as a tool for chemical synthesis. Nature, 590, 89–96.',
    'Strieth-Kalthoff, F., Sandfort, F., Kühnemund, M., et al. (2022). Machine learning for chemical reactivity: the importance of failed experiments. Angewandte Chemie International Edition, 61, e202204647.',
    'Coley, C. W., Thomas, D. A., Lummiss, J. A. M., et al. (2019). A robotic platform for flow synthesis of organic compounds informed by AI planning. Science, 365, eaax1566.',
    'Burger, B., Maffettone, P. M., Gusev, V. V., et al. (2020). A mobile robotic chemist. Nature, 583, 237–241.',
    'Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.',
    'Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.',
  ].map((s, i) => new Paragraph({ children: [t(`[${i + 1}] ${s}`, { size: 18 })], spacing: { after: 80 }, indent: { left: 360, hanging: 360 } })),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: '000000' }, paragraph: { spacing: { before: 360, after: 180 }, keepNext: true, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: '000000' }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 240 } } } }] }] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: [...cover, ...body],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT); });
