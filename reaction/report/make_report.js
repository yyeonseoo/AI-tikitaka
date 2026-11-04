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
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1600 }, children: [t('팔라듐 짝지음 반응의 수율 예측과 실험 우선순위 추천', { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t('팀명  윤멘토님과아이들', { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t('윤연서 (멘토) · 이경은 (멘티) · 김탁우 (멘티)', { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t('2026년 11월', { size: 22 })] }),
  br(),
  new Paragraph({ children: [t('목차', { size: 28, bold: true })], spacing: { after: 200 } }),
  new TableOfContents('목차', { hyperlink: true, headingStyleRange: '1-2' }),
];

const body = [
  h1('요약'),
  p('신약 공정 개발에서는 새 출발물질이 들어올 때마다 리간드, 염기, 용매 같은 반응 조건 수백 가지 중 어느 것이 잘 될지 몰라 많은 실험을 반복한다. 본 연구는 공개된 고속 대량 실험(HTE) 데이터 두 종(Buchwald–Hartwig 3,955건, Suzuki–Miyaura 5,760건)으로, 처음 보는 반응에 대해 수율을 예측하고 어떤 조건부터 실험할지 순위를 매기는 모델을 만들었다.'),
  p('모델은 시약마다 분자 특성(RDKit 기술자와 Morgan 지문)을 계산해 넣은 LightGBM 회귀 모델이다. 평가는 시험할 출발물질이 학습에 전혀 들어가지 않도록 그룹 단위로 나누어 진행했다. 주요 결과는 다음과 같다.'),
  b('Suzuki–Miyaura에서 실험을 하나도 하지 않은 상태로 384개 조건에 순위를 매겼을 때, 상위 20개 안에 최고 근처 조건이 들어 있는 비율은 87%였다(무작위 38%). 같은 분자 특성을 넣은 선형 회귀보다 상위 8개 추천의 수율 손실이 평균 10.4%p 작았다(95% 신뢰구간 [−20.3, −2.7]).'),
  b('Buchwald–Hartwig에서 처음 보는 출발물질의 최고 근처 조건을 찾는 데 필요한 실험 수는 무작위 3.9회에서 1.3회로 약 65% 줄었다.'),
  b('결과를 보며 다음 실험을 고르는 순차 추천에서는 최고 근처 조건까지의 실험 수가 무작위 27.1회에서 13.7회로 줄었다.'),
  b('모델이 중요하게 본 특징은 산화적 첨가 속도, 촉매 피독, 붕소 활성화 등 알려진 반응 메커니즘과 일치했다.'),
  p('이 모델을 바탕으로 반응을 고르면 조건 순위와 예측 수율, 근거를 보여주는 웹 프로토타입 ReactPilot을 구현했다.'),

  h1('1. 문제 정의'),
  h2('1.1 배경'),
  p('의약품 합성에서 탄소-질소, 탄소-탄소 결합을 만드는 팔라듐 촉매 짝지음 반응(cross-coupling)은 가장 많이 쓰이는 반응에 속한다. 같은 반응이라도 출발물질이 바뀌면 잘 듣는 리간드, 염기, 용매가 달라지고, 수율은 0%에서 90% 이상까지 크게 갈린다.'),
  p('조건의 가짓수는 곱으로 늘어난다. 이번에 다룬 Suzuki–Miyaura 데이터만 해도 리간드 12종, 염기 8종, 용매 4종을 조합하면 한 반응에 384가지 조건이 나온다. 제약회사는 이 문제를 HTE로 푼다. 96칸, 384칸 실험판에 조건을 깔아 한 번에 실험하는 방식이다. 하지만 출발물질을 확보하고 분석하는 비용이 들고, 새 출발물질이 들어올 때마다 이 과정을 다시 반복해야 한다.'),
  h2('1.2 해결하려는 문제'),
  p('본 연구가 답하려는 질문은 하나다. 처음 보는 반응이 들어왔을 때, 과거 실험 데이터만으로 어떤 조건부터 실험해야 하는가. 이를 세 가지 과제로 나누었다.'),
  ...table('세부 과제', ['과제', '입력', '출력', '실험실에서 쓰는 때'], [
    ['수율 예측', '출발물질과 조건 조합', '예상 수율(%)', '실험 전 성공 가능성 판단'],
    ['실험 전 조건 추천', '처음 보는 출발물질 쌍', '전체 조건의 순위', '첫 실험판에 깔 조건 선정'],
    ['순차 조건 추천', '지금까지의 실험 결과', '다음에 할 조건 4개', '결과를 보며 다음 실험판 구성'],
  ], [2, 3, 2.5, 3.5]),
  h2('1.3 성공 기준'),
  p('추천 모델의 가치는 실험을 얼마나 줄이느냐로 판단한다. 그래서 수율 예측 오차뿐 아니라 "추천 상위 k개만 실험했을 때 최고 조건을 얼마나 놓치는가"와 "최고 근처 조건을 찾을 때까지 몇 번 실험하는가"를 주 지표로 삼았다. 비교 대상은 무작위로 고르기, 과거에 잘 된 조건 순서대로 고르기, 선형 회귀 모델이다.'),

  h1('2. 관련 연구와 기존 서비스'),
  h2('2.1 선행 연구'),
  p('Ahneman 등(2018)은 Buchwald–Hartwig 반응 4,608건을 HTE로 측정하고, 시약의 양자화학 기술자를 랜덤 포레스트에 넣어 수율을 예측했다. 이 데이터는 이후 반응 수율 예측의 표준 벤치마크가 되었다. Chuang과 Keiser(2018)는 같은 데이터에서 기술자 대신 무작위 숫자를 넣어도 성능이 비슷하게 나온다는 점을 들어, 모델이 화학 정보를 실제로 쓰는지 확인해야 한다고 지적했다.'),
  p('Perera 등(2018)은 Pfizer에서 유동 반응기 기반 HTE로 Suzuki–Miyaura 반응 5,760건을 측정했다. Schwaller 등(2021)은 반응식을 문자열로 읽는 언어 모델(Yield-BERT)로 두 데이터의 수율을 예측했고, Shields 등(2021)은 베이지안 최적화로 실험 결과를 보며 다음 조건을 고르는 방식이 화학자보다 적은 실험으로 최적 조건에 도달함을 보였다. Strieth-Kalthoff 등(2022)은 실패한 반응이 기록되지 않는 문헌 데이터의 한계를 지적하며, 실패까지 모두 기록한 HTE 데이터의 가치를 강조했다.'),
  ...table('선행 연구와 본 연구의 비교', ['연구', '모델', '평가 방식', '본 연구와의 차이'], [
    ['Ahneman 등, 2018', '랜덤 포레스트 + 양자화학 기술자', '무작위 분할, 일부 첨가제 분할', '처음 보는 출발물질 단위 평가, 조건 추천 지표'],
    ['Schwaller 등, 2021', '언어 모델(Yield-BERT)', '무작위 분할 중심', '추천 성능과 실험 절감량을 직접 측정'],
    ['Shields 등, 2021', '베이지안 최적화', '순차 최적화', '과거 데이터를 활용한 실험 전 추천까지 포함'],
    ['본 연구', 'LightGBM + 분자 특성', '그룹 단위 분할, 부트스트랩 신뢰구간', '수율 예측, 실험 전 추천, 순차 추천을 한 모델로 평가'],
  ], [2.2, 2.6, 2.6, 3.4]),
  p('선행 연구도 대부분 기존 알고리즘(랜덤 포레스트, 언어 모델, 가우시안 과정)에 화학 표현을 결합하는 방식으로 모델을 만들었다. 본 연구 역시 같은 방식이며, 차이는 평가의 엄격함과 실제 사용 시나리오(처음 보는 반응에서의 추천)에 맞춘 지표에 있다.'),
  h2('2.2 기존 서비스'),
  ...table('관련 서비스', ['서비스', '제공 기능', 'ReactPilot과의 차이'], [
    ['Yoneda Labs', '클라우드 기반 반응 조건 추천과 최적화', '가장 가까운 서비스. ReactPilot은 검증 수치와 추천 근거를 함께 공개'],
    ['Atinary SDLabs', '자동화 실험실 연동 베이지안 최적화 플랫폼', '자동화 장비 전제. ReactPilot은 기존 HTE 실험판 운영에 바로 적용'],
    ['SYNTHIA (Merck)', '역합성 경로 설계', '어떤 반응을 쓸지 설계. 조건 최적화는 범위 밖'],
    ['Reaxys, SciFinder', '문헌 반응 데이터베이스 검색', '성공한 문헌 조건 검색. 실패 데이터 기반 예측 없음'],
    ['IBM RXN', '생성물 예측, 역합성', '무엇이 생기는지 예측. 수율과 조건 순위는 다루지 않음'],
  ], [2.2, 3.6, 4.6]),

  h1('3. 연구 방법'),
  h2('3.1 용어'),
  ...table('본 보고서에서 쓰는 용어', ['용어', '뜻'], [
    ['HTE (고속 대량 실험)', '여러 칸으로 나뉜 실험판에 조건 조합을 깔아 한 번에 수백 개 반응을 돌리는 실험 방식'],
    ['수율', '넣은 출발물질 중 원하는 생성물로 바뀐 비율(%)'],
    ['출발물질', '반응에서 결합을 만드는 두 분자. 이번 데이터에서는 할로젠 화합물과 아민(BH), 친전자체와 친핵체(SM)'],
    ['리간드, 염기, 용매, 첨가제', '팔라듐 촉매의 성질을 바꾸는 분자, 반응을 돕는 염기, 반응이 일어나는 액체, 방해 요소로 넣은 분자'],
    ['분자 특성', '분자 구조에서 계산한 숫자. 분자량, 극성 표면적 같은 RDKit 기술자 12개와 구조 조각 유무를 나타내는 Morgan 지문 64개를 시약마다 계산'],
    ['LightGBM', '결정 나무를 여러 개 이어 붙여 예측하는 모델. 시약 사이의 조합 효과를 잡아낼 수 있음'],
    ['선형 회귀', '각 특성에 가중치를 곱해 더하는 모델. 비교 기준으로 사용'],
    ['MAE', '예측 수율과 실제 수율 차이의 평균(%p). 낮을수록 좋음'],
    ['regret@k', '그 반응의 최고 수율 − 추천 상위 k개 안의 최고 수율(%p). 상위 k개만 실험했을 때 놓치는 수율'],
    ['최고 근처 조건', '그 반응의 최고 수율과 5%p 이내인 조건'],
    ['그룹 단위 평가', '시험할 출발물질(또는 출발물질 쌍)의 반응을 학습에서 통째로 빼고 평가하는 방식. 처음 보는 반응을 흉내 냄'],
    ['95% 신뢰구간', '시험 그룹을 다시 뽑는 부트스트랩을 1,000번 반복해 구한 범위. 두 방법 차이의 구간이 0을 넘지 않으면 차이가 확실하다고 판단'],
    ['SHAP', '예측값을 각 특성의 기여로 나누어 보여주는 설명 방법'],
  ], [2.6, 7.4]),
  h2('3.2 데이터'),
  p('두 데이터 모두 원 저자들이 시약 조합을 빠짐없이 실험한 HTE 데이터이고, 수율이 0%인 실패 반응도 모두 기록되어 있다. 문헌 데이터는 성공한 조건만 보고되는 경향이 있어, 조건 추천 모델을 학습하고 평가하기에는 이런 전수 실험 데이터가 적합하다.'),
  ...table('사용한 데이터', ['데이터', '반응', '바꾼 시약', '규모'], [
    ['Buchwald–Hartwig (BH), Ahneman 등, 2018', '탄소-질소 결합', '출발물질 15종(고리 5종 × 할로젠 Cl·Br·I), 리간드 4, 염기 3, 첨가제 22', '3,955건'],
    ['Suzuki–Miyaura (SM), Perera 등, 2018', '탄소-탄소 결합', '친전자체 7, 친핵체 4(출발물질 쌍 15개), 리간드 12, 염기 8, 용매 4', '5,760건'],
  ], [3.2, 1.8, 4, 1.2]),
  ...fig('substrates.png', 'BH 출발물질별 평균 수율. 할로젠(열)과 고리(행)가 바뀌면 같은 조건에서도 평균 수율이 2%에서 59%까지 달라진다.', 470),
  p('수율 차이가 어디서 오는지 보기 위해, 전체 데이터에 선형 회귀 두 개를 맞추고 각각이 수율의 흩어짐을 얼마나 설명하는지(R²) 비교했다. 두 모델 모두 시약 종류만 0과 1로 표시해 넣었고, 분자 특성은 쓰지 않았다.'),
  b('시약 효과를 더하기만 한 모델: 시약마다 점수를 하나씩 두고, 반응의 수율을 "출발물질 점수 + 리간드 점수 + 염기 점수 + …"로 계산한다. 어떤 리간드가 어느 출발물질과 만나도 같은 만큼 수율을 올린다고 가정한다.'),
  b('시약 짝 효과까지 넣은 모델: 위 점수에 두 시약의 조합마다 추가 점수를 더한다. 예를 들어 "이 출발물질과 이 리간드가 만났을 때"의 보정값이 따로 있다.'),
  p('짝 효과를 넣으면 설명 비율이 BH 71%에서 90%로, SM 51%에서 78%로 올랐다. 시약끼리의 궁합이 수율을 크게 좌우한다는 뜻이며, 조합 효과를 표현할 수 있는 모델이 필요한 이유다. 단, 짝 효과 모델은 각 조합의 점수를 데이터에서 직접 읽어 오므로, 학습에 없던 출발물질에는 쓸 수 없다. 처음 보는 반응을 다루려면 시약 이름 대신 분자 구조에서 계산한 특성을 입력으로 써야 한다.'),
  ...fig('interaction.png', '시약 효과를 더하기만 한 모델과 시약 짝 효과까지 넣은 모델의 설명 비율(R²).', 400),
  h2('3.3 모델'),
  p('모델은 반응 하나를 받아 예상 수율 하나를 내놓는다. SM 반응을 예로 들면 다음과 같다.'),
  b('입력: 친전자체, 친핵체, 리간드, 염기, 용매 다섯 가지 시약. 시약마다 RDKit으로 분자량, 극성 표면적, 회전 가능한 결합 수 등 기술자 12개와, 분자 안에 특정 구조 조각(C–Cl, 피리딘 질소 등)이 있는지를 0과 1로 표시한 Morgan 지문 64개를 계산한다. 다섯 시약의 숫자를 이어 붙이고, 이 데이터 안에서 모든 시약이 같은 값을 갖는 칸은 뺀다. 그 결과 한 반응은 약 210개의 숫자로 표현된다.'),
  b('출력: 그 반응의 예상 수율(%).'),
  p('시약을 이름이 아니라 구조에서 계산한 숫자로 표현하기 때문에, 학습에 없던 출발물질이 들어와도 구조가 비슷한 시약에서 배운 내용을 옮겨 쓸 수 있다. LightGBM은 "C–Cl 조각이 있고 리간드가 XPhos이면 수율이 낮다"처럼 여러 특성의 조합을 조건으로 나누어 배우므로, 3.2절에서 확인한 시약 사이의 궁합을 표현할 수 있다.'),
  p('추천은 이 예측을 그대로 활용한다. 새 반응이 들어오면 가능한 모든 조건에 대해 수율을 예측하고, 예측값이 높은 순서로 조건을 정렬해 추천한다. 순차 추천에서는 새로 얻은 실험 결과를 학습 데이터에 더해 모델을 다시 학습하고, 아직 실험하지 않은 조건 중 예측값이 가장 높은 4개를 다음 실험으로 고른다.'),
  p('예측기는 LightGBM 회귀 모델 5개를 서로 다른 난수 시드로 학습해 평균 낸 앙상블이다. 설정은 고정했고, 중첩 검증으로 설정 후보 30개를 탐색해도 성능 변화가 0.2%p 수준이어서 고정 설정을 최종 모델로 정했다.'),
  ...table('최종 모델 구성', ['구성 요소', '내용'], [
    ['입력', '출발물질, 리간드, 염기, 용매(또는 첨가제)의 분자 특성. 시약마다 RDKit 기술자 12 + Morgan 지문 64'],
    ['예측기', 'LightGBM 회귀, 시드 5개 앙상블'],
    ['출력', '조건별 예상 수율과 그 순위'],
    ['순차 추천', '첫 4개는 무작위로 실험, 이후 4개씩 9번. 매번 결과를 반영해 재학습하고 예측 상위 4개를 선택'],
  ], [2.5, 7.5]),
  h2('3.4 평가 방법'),
  p('실제 사용 상황은 처음 보는 반응에 대한 추천이므로, 모든 평가는 그룹 단위로 했다. BH는 출발물질 15종을 하나씩 시험으로 빼고 나머지 14종으로 학습했다. SM은 출발물질 쌍 15개를 하나씩 빼고 나머지 14쌍(5,376건)으로 학습했다. 두 방법의 차이는 시험 그룹 단위 부트스트랩 95% 신뢰구간으로 판단했다.'),

  h1('4. 결과'),
  h2('4.1 수율 예측'),
  p('이미 써 본 시약의 새로운 조합을 무작위로 시험에 뺐을 때, LightGBM의 예측 오차는 BH 4.0%p(R² 0.96), SM 6.8%p로, 같은 조건 평균을 쓰는 단순 방법(BH 6.8, SM 10.8)보다 크게 작았다.'),
  p('더 어려운 상황인 처음 보는 출발물질에서도 오차가 줄었다. 분자 정보를 쓰지 않는 단순 방법은 18.1%p, 같은 분자 특성을 넣은 선형 회귀는 15.8%p, LightGBM은 11.8%p였다. LightGBM과 선형 회귀의 차이는 −4.0%p, 95% 신뢰구간 [−6.4, −1.3]으로 0을 넘지 않았다. 분자 특성이라는 정보와, 그 정보를 조합으로 쓰는 트리 모델이 함께 성능을 냈다.'),
  ...table('처음 보는 출발물질의 수율 예측 오차 (BH, MAE %p)', ['방법', '입력 정보', 'MAE'], [
    ['조건 평균 (단순 방법)', '같은 조건의 다른 출발물질 평균. 분자 정보 없음', '18.1'],
    ['선형 회귀', '분자 특성', '15.8'],
    ['LightGBM', '분자 특성', '11.8'],
  ], [3, 5, 2]),
  h2('4.2 BH 조건 추천'),
  p('BH에서 처음 보는 출발물질마다 리간드 4종 × 염기 3종 = 12개 조건에 순위를 매기고, 첨가제별 상황 330개에서 추천 성능을 쟀다. 추천 상위 3개만 실험해도 최고 조건이 들어 있을 확률은 85%로, 무작위로 3개를 고를 때(25%)보다 훨씬 높았다. 최고 근처 조건을 찾을 때까지 필요한 실험 수는 무작위 3.9회에서 1.3회로 약 65% 줄었다.'),
  p('Chuang과 Keiser(2018)의 지적대로 모델이 화학 정보를 실제로 쓰는지 확인하기 위해, 분자 특성 대신 같은 개수의 무작위 숫자를 넣은 대조군도 학습했다. 대조군의 상위 3개 적중률은 74%로 떨어졌고, regret@3 차이는 +0.54%p [+0.18, +0.92]로 분자 특성 쪽이 확실히 나았다.'),
  ...fig('topk.png', 'BH 처음 보는 출발물질에서 추천 상위 k개 안에 최고 조건이 포함될 확률.', 560),
  h2('4.3 SM 실험 전 조건 추천'),
  p('SM은 고를 조건이 384개라 추천 능력의 차이가 더 크게 드러난다. 처음 보는 출발물질 쌍에 대해 실험을 하나도 하지 않은 상태로 384개 조건에 순위를 매겼다.'),
  ...table('SM 실험 전 추천 성능 (출발물질 쌍 15개 평균)', ['방법', 'regret@1 (%p)', 'regret@8 (%p)', '상위 8개 안 최고 근처', '상위 20개 안 최고 근처'], [
    ['무작위', '50.0', '19.7', '19%', '38%'],
    ['과거에 잘 된 순서', '23.5', '13.9', '47%', '60%'],
    ['선형 회귀 + 분자 특성', '24.3', '15.3', '40%', '53%'],
    ['선형 회귀 + 시약 짝 효과', '14.8', '6.3', '60%', '80%'],
    ['LightGBM + 분자 특성', '10.6', '5.0', '60%', '87%'],
  ], [3.2, 1.6, 1.6, 1.9, 1.9]),
  p('LightGBM의 상위 8개 추천은 같은 분자 특성을 넣은 선형 회귀보다 최고 수율에 평균 10.4%p 가까웠다(95% 신뢰구간 [−20.3, −2.7]). 시약 짝 효과를 직접 넣은 선형 모델과 비교해도 −1.3%p [−3.0, −0.3]로 LightGBM이 확실히 나았고, 과거에 잘 된 순서(−8.9%p [−17.3, −2.3])와 무작위(−14.7%p [−19.8, −9.7])보다도 나았다.'),
  ...fig('recommend.png', 'SM 처음 보는 출발물질 쌍에서 추천 상위 20개 안에 최고 근처 조건이 있는 비율.', 560),
  h2('4.4 SM 순차 조건 추천'),
  p('실제 실험실에서는 첫 실험판의 결과를 보고 다음 실험판을 구성한다. 이 과정을 다른 14쌍의 과거 데이터가 있는 상태에서 재현했다. 방식은 베이지안 최적화(Shields 등, 2021)와 같은 반복 구조(예측 → 실험 → 재학습)이고, 대리 모델로 LightGBM 앙상블을 썼다. 처음 4개 조건은 무작위로 실험하고, 이후 4개씩 9번, 총 40번까지 실험하며 매번 결과를 반영해 모델을 다시 학습했다.'),
  p('최고 근처 조건을 찾을 때까지 필요한 실험 수는 무작위 27.1회, 과거에 잘 된 순서 16.5회, LightGBM 13.7회였다. 20번 실험 시점의 regret은 무작위보다 8.3%p [−11.6, −5.1], 과거에 잘 된 순서보다 6.5%p [−14.2, −0.6] 작았다. 과거 순서는 출발물질이 바뀌어도 같은 조건을 먼저 추천하기 때문에, 염소 출발물질처럼 잘 듣는 조건이 다른 반응에서 크게 놓쳤다. LightGBM은 첫 결과를 반영한 직후(5번째 실험부터) 최고 수율과의 차이를 빠르게 줄였다.'),
  ...fig('sequential.png', 'SM 순차 추천에서 실험 수에 따른 최고 수율과의 차이(15쌍 평균). 낮을수록 좋다.', 520),

  h1('5. 모델 해석'),
  p('추천을 실험실에서 쓰려면 모델이 왜 그 조건을 골랐는지 화학적으로 납득할 수 있어야 한다. SHAP으로 각 특성이 예측에 기여한 정도를 계산하고, 시약별 평균 수율과 함께 알려진 반응 메커니즘과 비교했다.'),
  h2('5.1 Buchwald–Hartwig'),
  ...fig('shap_bh.png', 'BH 모델의 SHAP 요약. 위쪽 특성일수록 예측에 큰 영향을 준다.', 560),
  ...fig('chem_bh.png', 'BH 시약별 평균 수율. 반응 자리 할로젠, 리간드, 염기.', 620),
  ...table('BH에서 모델이 배운 것과 화학적 해석', ['데이터와 모델에서 보인 것', '화학적 해석'], [
    ['출발물질 특성이 가장 큰 영향. 상위 구조 조각은 벤젠 고리의 C–Cl(평균 수율 6% 대 40%)과 메톡시기(20% 대 36%)', '팔라듐이 탄소-할로젠 결합에 끼어드는 첫 단계(산화적 첨가)는 C–Cl에서 가장 느리고, 전자를 미는 메톡시기가 있으면 더 느려진다. 4-메톡시클로로벤젠의 평균 수율은 1.9%'],
    ['같은 C–Cl이라도 피리딘 고리는 수율이 높음(2-피리딜 43.5%)', '피리딘은 전자가 부족한 고리라 C–Cl 결합이 활성화된다'],
    ['리간드 중 XPhos가 가장 낮음(20% 대 34~41%)', '이 반응에는 인에 tert-부틸처럼 더 크고 전자가 풍부한 기가 붙은 리간드가 유리하다'],
    ['에스터가 붙은 아이소옥사졸 첨가제가 가장 나쁨(9~20%)', '아이소옥사졸의 N–O 결합에 팔라듐이 끼어들어 촉매를 소모시킨다(Ahneman 등, 2018). 전자를 당기는 에스터가 이 부반응을 쉽게 만든다'],
  ], [5, 5]),
  h2('5.2 Suzuki–Miyaura'),
  ...fig('shap_sm.png', 'SM 모델의 SHAP 요약.', 560),
  ...table('SM에서 모델이 배운 것과 화학적 해석', ['데이터와 모델에서 보인 것', '화학적 해석'], [
    ['반응 자리별 평균 수율 I 59.5% > OTf 44.6% > Br 41.5% > Cl 22.1%', '산화적 첨가의 반응성 순서(I > OTf > Br > Cl)와 일치한다'],
    ['SHAP 1위는 친핵체의 극성 표면적. 이 값이 붕소 종류를 구분하며, BF₃K 염(25.5%)이 보론산(52.2%)보다 크게 낮음', 'BF₃K 염은 반응 전에 물과 반응해 보론산으로 바뀌어야 해서 이 조건에서 느리다'],
    ['두 자리 리간드(Xantphos, dppf)가 수율을 낮춤', '두 자리 리간드는 팔라듐에 두 곳으로 결합해 착물 구조가 달라진다. 이 조건에서는 한 자리 인 리간드인 PPh₃(53%)가 가장 좋았고 Xantphos(16%)가 가장 나빴다'],
  ], [5, 5]),
  p('모델이 의존한 특징들이 산화적 첨가, 촉매 피독, 붕소 활성화라는 알려진 메커니즘과 맞아떨어졌다. 모델의 추천이 우연한 상관이 아니라 화학적으로 의미 있는 정보에 근거한다는 뜻이며, 추천 화면에서 근거를 함께 보여줄 수 있는 바탕이 된다.'),

  h1('6. 서비스 프로토타입: ReactPilot'),
  p('연구 결과를 실제 사용 흐름으로 옮긴 웹 프로토타입이다. Pfizer가 공개한 SM 데이터 15쌍을 대상으로, 사용자가 반응을 고르면 모델이 384개 조건의 순위를 보여주고, 추천 순서대로 실험을 진행하는 과정을 재현한다. 실제 수율은 데이터에 기록된 값을 꺼내 보여준다.'),
  ...fig('demo', 'ReactPilot 화면. 왼쪽부터 반응 선택, 시약 3D 구조와 AI 추천 순위, 진행 상황과 검증 성적.', 600),
  ...table('ReactPilot의 입력과 출력', ['구분', '내용'], [
    ['사용자 입력', '반응(출발물질 쌍) 선택, "다음 4개 실험하기" 버튼, 3D 구조로 볼 시약 선택'],
    ['AI 추천 순위', '384개 조건(리간드 × 염기 × 용매)의 예상 수율 순위. 실험 결과가 쌓일 때마다 다시 계산'],
    ['실험 기록과 진행 그래프', '실험한 조건의 실제 수율, 지금까지 찾은 최고 수율, 무작위로 고를 때와의 비교'],
    ['익숙도', '학습 데이터에 같은 출발물질을 쓴 반응이 몇 개 있었는지 표시'],
    ['검증 성적', '처음 보는 반응 15개에서 측정한 추천 성능(상위 20개 안 최고 근처 87% 등)'],
  ], [2.6, 7.4]),
  p('예시 화면의 6-클로로퀴놀린 + 인다졸 보론산 반응에서는 첫 번째 추천 실험에서 가능한 최고 수율(93.7%)과 5%p 이내인 조건을 찾았다. 같은 반응에서 무작위로 고르면 평균 43번의 실험이 필요하다.'),

  h1('7. 차별점과 활용 방안', false),
  h2('7.1 차별점'),
  b('검증의 투명성: 처음 보는 반응 단위로 평가한 추천 성능과 신뢰구간을 사용자에게 그대로 공개한다. 기존 상용 서비스는 대체로 내부 검증 수치를 공개하지 않는다.'),
  b('추천 근거 제시: SHAP 기반 해석이 알려진 반응 메커니즘과 일치함을 확인했고, 이를 화면에서 근거로 보여줄 수 있다.'),
  b('가벼운 구조: 트리 모델 앙상블이라 일반 PC에서 학습과 추천이 가능하다. 데이터 유출을 꺼리는 제약사가 사내 서버에 설치해 자사 HTE 데이터로 다시 학습하기 쉽다.'),
  b('기존 업무 흐름과의 호환: 자동화 장비 없이 기존 96칸, 384칸 HTE 실험판 운영에 추천 순서만 더하면 된다.'),
  h2('7.2 활용 방안'),
  p('제약사와 위탁개발생산(CDMO) 기업의 공정 개발 부서가 일차 고객이다. 새 출발물질이 들어오면 첫 실험판에 깔 조건을 추천받고, 결과를 입력하면 다음 실험판을 추천받는 방식으로 쓴다. 본 연구의 결과대로라면 BH에서는 최고 근처 조건까지의 실험이 약 65%, SM 순차 탐색에서는 약 절반 줄어든다. 사내에 쌓인 HTE 데이터로 모델을 다시 학습하면 회사가 주로 다루는 반응에 맞춘 추천이 가능하다.'),

  h1('8. 결론 및 향후 계획', false),
  p('본 연구는 공개 HTE 데이터 두 종으로 처음 보는 반응의 수율을 예측하고 실험 조건의 우선순위를 추천하는 모델을 만들고, 이를 웹 프로토타입으로 구현했다. 분자 특성을 넣은 LightGBM은 같은 특성의 선형 회귀보다 수율 예측(BH 오차 15.8 → 11.8%p)과 실험 전 추천(SM regret@8 15.3 → 5.0%p) 모두에서 확실히 나았고, 무작위 대비 최고 근처 조건까지의 실험 수를 BH 65%, SM 순차 탐색에서 약 절반 줄였다. 모델이 의존한 특징은 알려진 반응 메커니즘과 일치했다.'),
  p('다음 단계는 다음과 같다.'),
  b('다른 기관의 HTE 데이터와 다른 반응 종류(아마이드 결합 형성 등)로 대상 확장'),
  b('실제 실험실에서 추천 순서대로 실험해 절감 효과 확인'),
  b('사용자가 자사 데이터를 올려 모델을 다시 학습하는 기능 추가'),

  h1('참고문헌', false),
  ...[
    'Ahneman, D. T., Estrada, J. G., Lin, S., Dreher, S. D., Doyle, A. G. (2018). Predicting reaction performance in C–N cross-coupling using machine learning. Science, 360, 186–190.',
    'Chuang, K. V., Keiser, M. J. (2018). Comment on "Predicting reaction performance in C–N cross-coupling using machine learning". Science, 362, eaat8603.',
    'Perera, D., Tucker, J. W., Brahmbhatt, S., et al. (2018). A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science, 359, 429–434.',
    'Schwaller, P., Vaucher, A. C., Laino, T., Reymond, J.-L. (2021). Prediction of chemical reaction yields using deep learning. Machine Learning: Science and Technology, 2, 015016.',
    'Shields, B. J., Stevens, J., Li, J., et al. (2021). Bayesian reaction optimization as a tool for chemical synthesis. Nature, 590, 89–96.',
    'Strieth-Kalthoff, F., Sandfort, F., Kühnemund, M., et al. (2022). Machine learning for chemical reactivity: the importance of failed experiments. Angewandte Chemie International Edition, 61, e202204647.',
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
