# SAM-impute 프로젝트 (Claude Code용)

## 목적 / 무대 (확정)
순서형(ordinal) 불완전 행렬 imputation. OR 저널 재투고 리비전.
- 타겟: **JORS**(1순위) / EJOR(대안). 출판확률·속도 우선. (IS 저널 아님 — ISR/DSS/IJEC/Omega/JBR desk reject 이력)
- **기여 축(재정의 확정)**: **monotonization LP = 이식 가능한 ordinal-consistency 연산자**(Thm 1: 정수성, O(m)).
  SAM은 그 한 인스턴스. **RMSE 최고를 주장하지 않음.**
- 이론(추정유사도 bias-variance, gradedness G)은 **Lee-Shah / Breese(case amplification) / Herlocker 계승·특화**.
  related-work·설명으로만 위치. 새 정리로 내세우지 않음.
- 상세: `REVISION_PLAN.md`(계획·게이트), `THEORY_SKETCH.md`(이론), `COAUTHOR_MEMO.md`(공동저자 브리핑),
  `HANDOFF_2026-07.md`(인수인계).

## 채택 구성
probability scaling(normal3) + Pearson(`df.corr()`) + **α=1** + monotonization LP(Gurobi).
- 유사도는 **Pearson**으로 통일(논문 본문 "Spearman"은 오류 → 수정 대상).
- Gurobi: SKKU 학술 라이선스 **확보 완료**(ML-1M·큰 m 실행 가능).

### α 가중의 정확한 상태 (혼동 금지 — 두 겹)
- **원본 NQA(nested_quantile, 목적함수=train 자기재구성 SSE)는 영구 폐기.**
  목적함수가 틀려 α 오버슈팅(toy 강구조서 14.4 선택, 실제 최적 ~3.7). 어느 데이터에서도 권장 안 함.
- **α-tuning 일반은 이론적으로 지지됨**: 임계 G/(σ²+τ²(p)) > c. graded(이웃 유사도가 단계적) 데이터에서
  α>1이 표본외 RMSE 개선 — 이론 + toy 이중 확인.
- **우리 데이터는 저-G**: ML-100K/1M은 α=1 최적. **imputation_2026**(17유사도 × nested CV × inner_sim
  leak-free × α-grid 0~10.5)에서 **제대로 골랐는데도 α*≈1, 이득 미미** = 구조(저-G) 문제의 직접 증거.
  → "NQA만 고치면?" → 목적함수를 고쳐도 저-G에선 여전히 α=1. **두 문제는 독립.**
- **조건부 복원 규칙**:
  - 구조 스크리닝에서 graded 실데이터(α*>1 & 이득 유의) **발견 시** → α를 **leak-free inner-validation RMSE
    기준 적응 선택**으로 복원(NQA 아님, 목적함수 교체판).
  - **미발견 시** → α=1 유지, α 효용은 toy로만. **논문의 닻(LP)은 이와 무관하게 성립.**

## 원본 코드 (수정 금지, import·참조만)
- canonical 파이프라인: `Imputation/Run_algorithm(mono_impute, 2024.03).ipynb`
  (normal1/2/3, c_agg, rmse_cal2, nested_quantile, monotonization)
- 논문 Table 4(by data size) = `test_by_size/summary_20240215_table.xlsx`
- 논문 Table 2(scaling×aggregation) = `test_by_weighted_method/result_summary_all_with_nested_q.xlsx`
- fold 생성기: `test_by_size/k_fold_dataset_creating_process(2023_08_08).ipynb`
- **모든 새 작업은 `revision/` 에서. 원본 `Imputation/` 은 건드리지 말 것.**

## 정정된 프로토콜 (`revision/phase0/clean_pipeline.py` — 모든 실험의 기반)
원본의 두 결함이 정정됨:
- **(A) fold 비-disjoint**: 생성기가 루프 *안*에서 상수 seed 재시드 + prefix 슬라이스 → 관측치 ~66%만 커버.
  → 한 번만 shuffle 후 disjoint 분할(assert 검증).
- **(B) 유사도 누수**: `corr = df_orig.corr()`를 fold 루프 *밖*에서 전체행렬(테스트 포함)로 계산·재사용.
  → fold마다 `corr = df_train.corr()` (음수 클리핑·대각 0 유지).
- 정직한 baseline RMSE ≈ **0.96–1.00** (누수판 0.76–0.94).
- 원본 2025 = train/test + 10-fold(nested 아님). 2026 = nested CV + inner_sim(leak-free) = 더 성숙.

## Phase-1 결과 (완료 — 이 사실 위에서 작업할 것)
- **softImpute가 RMSE 12/12셀 1위**. SAM의 원래 RMSE 승리는 **방법론 아티팩트**.
- **SAM은 ACC 12/12셀 1위**이나, baseline 공정 튜닝 후 마진 +5~10 → **+2~3 p.p.**로 축소(유의는 유지),
  RMSE 격차는 +1.3% → **+5%**로 확대.
- **연속-정수 분해**: monotonization은 RMSE 거의 무손실(연속 대비 ML-100K +0.7%, ML-1M −0.5% 개선),
  naive 반올림 대비 3~5% 우수. **격차의 근원은 aggregation**(연속 레벨서 이미 softImpute 대비 +8~12% 열세).
- **이식성**: softImpute+mono ≈ SAM 동률(RMSE p=0.65, ACC p=0.94) → **일반성의 증거**.
- **조건성**: ΔACC(mono−round) = 가중-CF +0.069 > softImpute +0.050 > **KNN −0.034(악화)**.
  → mono는 **"매끄러운 순서"를 요구**하는 조건부 연산자.

## 현재 단계 / 다음 할 일
1. **미팅 노트북** `revision/meeting/coauthor_walkthrough.ipynb` — 공동저자 설득용.
   (문제점 코드 → 정정 → 축소 사다리 → NQA 두 겹 이유(+2026 증거) → 살아남는 것 → 향후)
2. **잔여 갭**: ML-1M + m=100 격자 재현 · **[21] Park-Kim-Zhu(2023) 정확 재구현**(현재 근사 대리) ·
   MissForest 표 완성(m=500·ML-1M).
3. **구조 스크리닝(시간-박스, 선택)**: 고-G 후보(BookCrossing 1~10, Jester-binned, FilmTrust/Epinions, 애니)
   3~4개를 2026 파이프라인에 태워 graded 실데이터 존재 확인 → α 부활 스위치.
4. **집필**: LP 연산자 전면 + 누수 없는 재평가 + 정직한 진단. 정오 수정(1000×500 **0.8821**, 표 번호,
   Spearman→Pearson).

## 보류 결정 (공동저자 미팅 사항)
- **imputation_2026 통합**: 2026의 **KNN-표준 aggregation을 논문 base로 채택**할지(권고 — 현재 bespoke
  전체-이웃 가중평균을 표준 KNN-CF로 대체하면 base가 방어 가능해지고 LP가 깨끗한 델타). 17유사도는
  "연산자가 유사도 선택에 강건" 재료.
- 범위(빠른 출하 vs 추가 투자), α 축 분량.

## 환경 / 규칙
- **git**(단일 컴퓨터, 브랜치 `revision-or-2026`): 작업 전 `git pull`, 후 `git add . && git commit && git push`.
- **imputation_2026 참조 시**: `results/inner_sim/` 만 유효(leak-free).
  유효 파일 = `results/inner_sim/combined/all_folds_grid_results_20260416_004853.csv`.
  **제외**: archived/overfitted, 비-inner_sim의 20260117/20260128/20260129, 20260312(lambda 혼재).
  2026 레포의 `CLAUDE.md` '사용 금지' 목록을 우선 확인할 것.
- 결과 보고: 실행 스크립트명 + 요약표/콘솔 + 에러 전문 + **예상과 다른 수치**.
- 새 실험은 `revision/phase2/` 등 `revision/` 하위에 추가(기존 프로토콜·결과 경로 재사용).
